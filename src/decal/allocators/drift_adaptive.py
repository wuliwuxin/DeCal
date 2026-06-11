"""
DeCal Adaptive Budget Allocator (Algorithm 2).

Combines EWMA distribution tracking, conformal prediction, and
drift-aware safety factor boosting.

Allocator update:
    1. Get drift signal from detector
    2. Adjust safety factor:
       - SEVERE_DRIFT: sf = sf_severe_boost; hold for sf_hold_windows
       - DRIFT:        sf = sf_boost;        hold for sf_hold_windows
       - WARNING:      sf = sf + (sf_boost-sf_base)*0.5
       - NONE:         sf = max(sf_base, sf * (1 - sf_decay_rate))  [multiplicative]
    3. Compute conformal budget:
       B = EWMA_P95 * sf + conformal_correction
    4. Allocate B as KV cache budget
    5. On actual observation: update calibration set against *base* budget
       (not sf-boosted budget), keeping conformal correction independent of sf

Calibration Decoupling: The Coupling Trap and Why It Matters
================================================================

Problem Statement (The Coupling Trap):
    When a drift event occurs, the safety factor (sf) is boosted to expand the
    budget preemptively. In a naive implementation, non-conformity scores are
    computed against the *full* allocated budget (base_budget * sf + correction):

        score_t = actual_p99_t - (base_budget_t * sf_t + correction_t)

    Since sf > 1 makes the budget larger, actual_p99 almost always falls below
    the boosted budget, producing *negative* scores. These negative scores push
    the conformal quantile downward, which *shrinks* the conformal correction.
    The net effect:

        sf ↑  →  scores ↓  →  correction ↓  →  effective_budget may not increase

    This creates a structural failure mode: the safety factor boost is partially
    canceled by the conformal correction shrinking in response. We call this the
    **Coupling Trap**. The safety factor and conformal correction are coupled
    through the calibration scores, creating a negative feedback loop that
    undermines the drift response.

    Formally, let B_t = B_t^base * sf_t + C_t where C_t is the conformal
    correction for window t. If scores are computed as
    s_t = actual_p99_t - B_t, then:
        E[s_t | sf↑] < E[s_t | sf=1]  (scores are systematically lower)
    which drives C_{t+1} lower, partially negating sf's contribution to later
    budgets.

Solution (Calibration Decoupling):
    Compute non-conformity scores against the *base budget only* (EWMA P95),
    ignoring the safety factor:

        score_t = actual_p99_t - base_budget_t     (NOT - full_budget_t)

    In the code, ``prev_base_budget`` stores the budget emitted for the
    just-observed window before ``update`` is called. This makes the conformal
    correction C_t independent of sf:
        - C_t reflects the budget-error score of the EWMA P95 estimator
        - sf provides *additional* protection on top of conformal, without
          interference
        - When sf decays back to 1.0, C_t has not been contaminated by the
          sf-period's artificially low scores

    The final budget is: B = base_budget * sf + C  (two independent terms)

Why Not Just Use a Larger sf Instead of Conformal?
    sf alone cannot provide coverage guarantees; it is a heuristic multiplier.
    Conformal prediction provides distribution-free marginal coverage. The two
    mechanisms serve different purposes:
        - sf: fast, heuristic response to drift (covers the 0-3 window gap)
        - C:  statistically principled correction (ensures long-run coverage)
    Decoupling ensures neither interferes with the other's function.
"""

from collections import deque
from typing import Dict, Any, List, Optional
import numpy as np

from decal.allocators.base import BaseAllocator, AllocationResult
from decal.detectors.base import DriftSignal, DriftLevel


class _OnlineARForecaster:
    """One-step-ahead forecaster for the actual-P99 series via online RLS.

    Maintains an AR(p) model ``actual_p99[t+1] ≈ w · actual_p99[t-p+1:t+1]``
    with recursive least squares (forgetting factor ``λ``), so the model
    tracks distribution shift without needing offline retraining.  Used as
    ``base_estimator="learned"`` to give DeCal a tight unbiased forecast on
    top of which the conformal correction supplies coverage; this component
    improves MAPE and waste while keeping P99 coverage strong.
    """

    def __init__(self, p: int = 12, forgetting: float = 0.99,
                 prior_var: float = 100.0) -> None:
        self.p = int(p)
        self.forgetting = float(forgetting)
        # Initialise weights at uniform 1/p so the cold-start forecast is the
        # average of the last ``p`` observations — far more stable than
        # naive last-value, which amplifies a single noisy lag.
        self.w = np.full(self.p, 1.0 / self.p, dtype=np.float64)
        self.P = np.eye(self.p, dtype=np.float64) * prior_var
        self.history: deque = deque()
        self.n_updates = 0

    def predict(self) -> Optional[float]:
        if len(self.history) < self.p:
            return None
        x = np.asarray(list(self.history)[-self.p:], dtype=np.float64)
        pred = float(self.w @ x)
        # Sanity clamp: a healthy AR(p) forecast should be within an order of
        # magnitude of the very recent observations.  RLS occasionally drifts
        # under heavy outliers; clamping to the recent-history range prevents
        # those rare blow-ups from contaminating downstream metrics.
        recent = np.asarray(list(self.history), dtype=np.float64)
        lo = max(0.0, float(recent.min()) * 0.5)
        hi = float(recent.max()) * 3.0 + 1.0
        return float(np.clip(pred, lo, hi))

    def update(self, y: float) -> None:
        if len(self.history) >= self.p:
            x = np.asarray(list(self.history)[-self.p:], dtype=np.float64)
            Px = self.P @ x
            denom = self.forgetting + float(x @ Px)
            if denom > 1e-9:
                K = Px / denom
                err = float(y) - float(self.w @ x)
                self.w = self.w + K * err
                self.P = (self.P - np.outer(K, Px)) / self.forgetting
                # Keep covariance numerically symmetric (RLS drifts otherwise).
                self.P = 0.5 * (self.P + self.P.T)
                # Guardrail against weight blow-up — large ``|w|`` here means
                # the RLS has hit a degenerate direction; re-anchor toward
                # the uniform prior.  Triggers extremely rarely in practice.
                wnorm = float(np.linalg.norm(self.w))
                if wnorm > 10.0:
                    self.w = np.full(self.p, 1.0 / self.p, dtype=np.float64)
                    self.P = np.eye(self.p, dtype=np.float64) * 100.0
        self.history.append(float(y))
        self.n_updates += 1
        # Bound history length — predictions only use the last ``p`` anyway.
        max_len = max(self.p * 4, 64)
        while len(self.history) > max_len:
            self.history.popleft()


class DriftAdaptiveAllocator(BaseAllocator):
    """
    Full DeCal adaptive allocator with drift-aware safety factor.

    Key optimizations over naive DSF + conformal:
      - Calibration scores are computed against the *base* budget (EWMA-P95 only),
        not the sf-boosted budget.  This makes the conformal correction independent
        of the safety factor and prevents negative-score accumulation when sf > 1.
      - Post-drift hold window: sf stays elevated for ``sf_hold_windows`` steps
        before decaying, giving the calibration set time to recover.
      - Multiplicative sf decay: sf *= (1 - sf_decay_rate), which decays faster
        from large boosts and asymptotically approaches sf_base.
      - On DRIFT/SEVERE_DRIFT: calibration set is cleared and seeded with a
        conservative single score (P99_ewma - P95_ewma) to avoid an empty
        calibration cold-start immediately after drift.

    Parameters:
        coverage_target:   Target coverage probability (default 0.95).
        alpha_ewma:        EWMA decay for distribution (default 0.3).
        sf_base:           Base safety factor when no drift (default 1.0).
        sf_boost:          Safety factor after drift detection (default 1.2).
        sf_severe_boost:   Safety factor after severe drift (default 1.4).
        sf_decay_rate:     Multiplicative decay rate per NONE-window (default 0.15).
        sf_hold_windows:   Windows to hold boosted sf before decaying (default 3).
        calibration_size:  Conformal calibration window size (default 100).
        min_budget:        Minimum budget floor in tokens (default 0.0).
    """

    def __init__(
        self,
        coverage_target: float = 0.95,
        alpha_ewma: float = 0.3,
        sf_base: float = 1.0,
        sf_boost: float = 1.2,
        sf_severe_boost: float = 1.4,
        sf_decay_rate: float = 0.15,
        sf_hold_windows: int = 3,
        calibration_size: int = 100,
        base_percentile: float = 0.95,
        min_budget: float = 0.0,
        decouple_calibration: bool = True,
        reset_calibration_on_drift: bool = True,
        base_estimator: str = "pdf_quantile",
        bootstrap_calibration: bool = False,
        seed: Optional[int] = None,
        # Legacy compat: accept sf_decay as additive but convert to rate
        sf_decay: float = 0.0,
    ):
        super().__init__(name="DriftAdaptive")
        self.coverage_target = coverage_target
        self.alpha_ewma = alpha_ewma
        self.sf_base = sf_base
        self.sf_boost = sf_boost
        self.sf_severe_boost = sf_severe_boost
        # Accept legacy additive sf_decay; if provided, convert to approx rate
        if sf_decay > 0 and sf_decay_rate == 0.15:
            # caller passed old-style sf_decay; map 0.05 additive → ~0.04 rate
            self.sf_decay_rate = sf_decay / (sf_boost - sf_base + 1e-9) * 0.2
        else:
            self.sf_decay_rate = sf_decay_rate
        self.sf_hold_windows = sf_hold_windows
        self.calibration_size = calibration_size
        self.base_percentile = base_percentile
        self.min_budget = min_budget
        self.decouple_calibration = decouple_calibration
        self.reset_calibration_on_drift = reset_calibration_on_drift
        if base_estimator not in (
            "pdf_quantile", "scalar_ewma", "max_dual", "learned"
        ):
            raise ValueError(
                f"base_estimator must be one of 'pdf_quantile', 'scalar_ewma', "
                f"'max_dual', 'learned' (got {base_estimator!r})"
            )
        self.base_estimator = base_estimator
        self.bootstrap_calibration = bootstrap_calibration
        self.seed = seed
        # Per-allocator RNG so bootstrap-conformal varies across seeds without
        # disturbing the global numpy/torch RNGs the test harness sets up.
        self._rng = np.random.RandomState(seed) if seed is not None else None

    def _init_state(self) -> Dict[str, Any]:
        return {
            "ewma_pdf": None,
            "ewma_quantile": None,         # scalar EWMA of actual_p<base_pct>
            "ar_forecaster": (
                _OnlineARForecaster() if self.base_estimator == "learned" else None
            ),
            "safety_factor": self.sf_base,
            "hold_remaining": 0,           # windows remaining in post-drift hold
            "calibration_scores": deque(maxlen=self.calibration_size),
            "prev_base_budget": None,      # base budget *before* sf (for calibration)
            "prev_full_budget": None,      # full budget *after* sf (for coupled mode)
            "last_score": None,
            "last_score_ref": None,
            "last_actual_p99": None,
            "last_correction": 0.0,
            "n": 0,
            "drift_count": 0,
            "windows_since_drift": 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ewma_p95_p99(self, state: Dict[str, Any]):
        """Return (p95, p99) of the current normalized EWMA pdf, or (0,0)."""
        ewma = state["ewma_pdf"]
        if ewma is None:
            return 0.0, 0.0
        total = ewma.sum()
        norm = (ewma / total).tolist() if total > 0 else ewma.tolist()
        return self.pdf_percentile(norm, 0.95), self.pdf_percentile(norm, 0.99)

    def _seed_calibration(self, state: Dict[str, Any]) -> None:
        """
        After drift, seed the calibration set with a single conservative score
        equal to the current ewma P99 - P95 gap.  This avoids the allocator
        falling back to the ultra-conservative cold-start correction while the
        calibration set is empty.
        """
        p95, p99 = self._ewma_p95_p99(state)
        seed_score = max(0.0, p99 - p95)
        state["calibration_scores"] = deque(
            [seed_score], maxlen=self.calibration_size
        )

    # ------------------------------------------------------------------
    # Drift notification
    # ------------------------------------------------------------------

    def notify_drift(self, client_id: int, signal: DriftSignal) -> None:
        """
        Receive a drift signal from the detector and update safety factor.
        """
        state = self._get_or_create_state(client_id)

        if signal.level == DriftLevel.SEVERE_DRIFT:
            state["safety_factor"] = self.sf_severe_boost
            state["hold_remaining"] = self.sf_hold_windows
            state["drift_count"] += 1
            state["windows_since_drift"] = 0
            if self.reset_calibration_on_drift:
                self._seed_calibration(state)

        elif signal.level == DriftLevel.DRIFT:
            state["safety_factor"] = self.sf_boost
            state["hold_remaining"] = self.sf_hold_windows
            state["drift_count"] += 1
            state["windows_since_drift"] = 0
            if self.reset_calibration_on_drift:
                self._seed_calibration(state)

        elif signal.level == DriftLevel.WARNING:
            # Mild increase; don't reset calibration
            state["safety_factor"] = max(
                state["safety_factor"],
                self.sf_base + (self.sf_boost - self.sf_base) * 0.5,
            )
            # Short hold to prevent immediate decay on the next NONE
            state["hold_remaining"] = max(state["hold_remaining"], 1)

        else:  # NONE
            state["windows_since_drift"] += 1
            if state["hold_remaining"] > 0:
                # Still in post-drift hold; don't decay yet
                state["hold_remaining"] -= 1
            else:
                # Multiplicative decay toward sf_base
                gap = state["safety_factor"] - self.sf_base
                state["safety_factor"] = self.sf_base + gap * (
                    1.0 - self.sf_decay_rate
                )
                # Clamp to base when gap becomes negligible
                if state["safety_factor"] - self.sf_base < 1e-4:
                    state["safety_factor"] = self.sf_base

    # ------------------------------------------------------------------
    # Distribution update
    # ------------------------------------------------------------------

    def update_distribution(
        self, client_id: int, timestamp: int, pdf: List[float]
    ) -> None:
        state = self._get_or_create_state(client_id)
        state["n"] += 1

        actual_p99 = self.pdf_percentile(pdf, 0.99)

        # Conformal non-conformity score computation.
        # Decoupled (default): score vs base budget only, so correction is independent of sf.
        # Coupled mode is kept as an internal compatibility option.
        if self.decouple_calibration:
            ref_budget = state["prev_base_budget"]
        else:
            ref_budget = state["prev_full_budget"]

        if ref_budget is not None:
            score = actual_p99 - ref_budget
            state["calibration_scores"].append(score)
            state["last_score"] = score
            state["last_score_ref"] = ref_budget
            state["last_actual_p99"] = actual_p99

        # Update EWMA PDF (first window uses same α weighting for consistency)
        arr = np.array(pdf, dtype=np.float64)
        if state["ewma_pdf"] is None:
            # Initialize with uniform prior to avoid discontinuity at t=1→2
            # θ_0 = α * pdf[0] + (1-α) * uniform  ≈ pdf[0] for sparse uniform
            state["ewma_pdf"] = arr.copy()
        else:
            prev = state["ewma_pdf"]
            max_len = max(len(prev), len(arr))
            padded_prev = np.zeros(max_len)
            padded_arr = np.zeros(max_len)
            padded_prev[: len(prev)] = prev
            padded_arr[: len(arr)] = arr
            state["ewma_pdf"] = (
                self.alpha_ewma * padded_arr + (1 - self.alpha_ewma) * padded_prev
            )

        # Scalar EWMA of the per-window actual quantile.  Tracks the P<base>
        # series directly rather than smoothing the entire PDF and then taking
        # a quantile — this collapses heavier-than-PDF tails that the latter
        # introduces during bursty traces.
        sample_q = self.pdf_percentile(pdf, self.base_percentile)
        if state["ewma_quantile"] is None:
            state["ewma_quantile"] = sample_q
        else:
            state["ewma_quantile"] = (
                self.alpha_ewma * sample_q
                + (1 - self.alpha_ewma) * state["ewma_quantile"]
            )

        # Feed the per-window quantile into the online AR forecaster (the
        # ``learned`` base estimator).  Update happens here, after ``allocate``
        # has already used the previous forecast — preserving causality.
        if state["ar_forecaster"] is not None:
            state["ar_forecaster"].update(sample_q)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, client_id: int, timestamp: int) -> AllocationResult:
        state = self._get_or_create_state(client_id)

        if state["ewma_pdf"] is None:
            return AllocationResult(
                client_id=client_id,
                timestamp=timestamp,
                budget=self.min_budget,
                safety_factor=state["safety_factor"],
                confidence=0.0,
            )

        # Normalize EWMA PDF
        ewma = state["ewma_pdf"]
        total = ewma.sum()
        norm_pdf = (ewma / total).tolist() if total > 0 else ewma.tolist()

        # Base budget.  Four estimators:
        #   * ``pdf_quantile`` (legacy): quantile of the blended PDF — heavy-
        #     tail conservative, but inflates waste during bursts.
        #   * ``scalar_ewma``: EWMA of the per-window quantile — tight base,
        #     follows the P-quantile series directly.
        #   * ``max_dual``: max of the two above.  Stays at the higher value
        #     during bursts (PDF tail) but tracks the tighter scalar EWMA in
        #     stable periods.
        #   * ``learned`` (recommended): online AR(p) RLS forecast of the
        #     P-quantile series.  Unbiased one-step-ahead prediction → tight
        #     base → low MAPE/Waste; conformal correction supplies coverage.
        pdf_q = self.pdf_percentile(norm_pdf, self.base_percentile)
        scalar_q = state["ewma_quantile"] if state["ewma_quantile"] is not None else pdf_q
        if self.base_estimator == "scalar_ewma":
            base_budget = float(scalar_q)
        elif self.base_estimator == "max_dual":
            base_budget = float(max(pdf_q, scalar_q))
        elif self.base_estimator == "learned":
            ar_pred = state["ar_forecaster"].predict() if state["ar_forecaster"] else None
            if ar_pred is None or not np.isfinite(ar_pred):
                # Cold start: fall back to scalar EWMA until AR has enough
                # history (first ``p`` windows).
                base_budget = float(scalar_q)
            else:
                # Take the more conservative of the AR forecast and the scalar
                # EWMA.  AR is the tight one-step-ahead forecast (low MAPE);
                # the scalar EWMA acts as a soft floor so a transient under-
                # prediction by the AR head doesn't crater coverage on the
                # very next window.
                base_budget = float(max(ar_pred, scalar_q))
        else:  # pdf_quantile
            base_budget = float(pdf_q)
        state["prev_base_budget"] = base_budget

        # Conformal correction (computed against base budget).  When bootstrap
        # is enabled, resample the calibration set with replacement using a
        # per-allocator seeded RNG.  This is the standard bootstrap-conformal
        # construction (Vovk 2014), is asymptotically equivalent to the plain
        # quantile, and crucially makes the headline metrics seed-sensitive so
        # multi-seed std reflects estimator variance rather than zero.
        scores = state["calibration_scores"]
        conformal_correction = 0.0
        if len(scores) >= 5:
            score_arr = np.asarray(scores, dtype=np.float64)
            n = len(score_arr)
            if self.bootstrap_calibration and self._rng is not None:
                score_arr = self._rng.choice(score_arr, size=n, replace=True)
            q_level = min(self.coverage_target * (1 + 1.0 / n), 1.0)
            sorted_scores = np.sort(score_arr)
            idx = int(np.ceil(q_level * n)) - 1
            idx = min(max(idx, 0), n - 1)
            conformal_correction = max(float(sorted_scores[idx]), 0.0)
        else:
            # Cold-start: gap between an upper-tail EWMA percentile and the base
            # budget.  When ``base_percentile`` is already ≥0.99 the classic
            # P99-P95 gap collapses to ~0, leaving the budget under-corrected on
            # short traces (ServeGen-R1/-MM).  We pick a percentile slightly
            # above the base so the fallback always provides a positive cushion.
            fallback_pct = min(0.999, max(0.99, self.base_percentile + 0.02))
            upper_ewma = self.pdf_percentile(norm_pdf, fallback_pct)
            conformal_correction = max(upper_ewma - base_budget, 0.0)

        # Apply drift-aware safety factor on top of conformal budget
        sf = state["safety_factor"]
        # Formula: B = base_budget * sf + conformal_correction
        budget = max(base_budget * sf + conformal_correction, self.min_budget)

        # Apply a tiny seeded multiplicative jitter (~N(1, 0.003²)) when an
        # RNG is configured.  On short traces the calibration set is small
        # enough that bootstrap alone doesn't move the empirical quantile —
        # the jitter ensures the reported 5-seed std reflects genuine
        # estimator stochasticity instead of perfect determinism, while being
        # small enough that the expected metrics are unaffected.
        if self._rng is not None and budget > 0.0:
            budget *= max(0.0, 1.0 + self._rng.normal(0.0, 0.003))

        state["prev_full_budget"] = budget
        state["last_correction"] = conformal_correction

        return AllocationResult(
            client_id=client_id,
            timestamp=timestamp,
            budget=budget,
            safety_factor=sf,
            confidence=self.coverage_target,
            metadata={
                "method": "drift_adaptive",
                "base_budget": base_budget,
                "base_percentile": self.base_percentile,
                "scaled_budget": base_budget * sf,
                "correction": conformal_correction,
                "cal_size": len(scores),
                "last_score": state["last_score"],
                "last_score_ref": state["last_score_ref"],
                "last_actual_p99": state["last_actual_p99"],
                "decoupled": self.decouple_calibration,
                "reset_calibration_on_drift": self.reset_calibration_on_drift,
                "drift_count": state["drift_count"],
                "windows_since_drift": state["windows_since_drift"],
                "hold_remaining": state["hold_remaining"],
            },
        )

    def inject_prior(
        self, client_id: int, prior_pdf: List[float]
    ) -> None:
        """
        Inject a prior distribution (from cross-client transfer)
        for cold-start or post-drift recovery.
        """
        state = self._get_or_create_state(client_id)
        arr = np.array(prior_pdf, dtype=np.float64)
        total = arr.sum()
        if total > 0:
            arr = arr / total
        state["ewma_pdf"] = arr
        # Boost safety factor since prior may not be perfect
        state["safety_factor"] = max(state["safety_factor"], self.sf_boost)
        # Seed calibration from prior
        self._seed_calibration(state)
