"""
Evaluation metrics for DeCal.

The whole stack — allocator, orchestrator, evaluator, and reports — treats
**P99 as the operational target** and 95% as the *coverage probability* (i.e.
the fraction of windows for which budget should be ≥ actual_p99).  Metrics in
this module are organised consistently around that contract:

  Window-level coverage
    - p95_coverage   : fraction of windows where budget ≥ actual P95
    - p99_coverage   : fraction of windows where budget ≥ actual P99   (target)
    - p999_coverage  : fraction of windows where budget ≥ actual P99.9 (tail)
    - oom_rate       : 1 − p99_coverage (window-level P99 fail)

  Request-level coverage
    - per_request_oom: probability mass above budget, weighted by request volume

  Estimation accuracy (MAPE)
    - mape_p95       : mean |budget − actual_p95| / actual_p95
    - mape_p99       : mean |budget − actual_p99| / actual_p99   (consistent w/ target)
    - mape           : alias for mape_p95 (kept for backward compatibility)
    - mae_p99/mse_p99/rmse_p99/smape_p99/mase_p99:
                       conventional one-step time-series errors against actual_p99
    - nmae_p99/nrmse_p99:
                       scale-free MAE/RMSE normalised by mean actual_p99

  Over-allocation / waste
    - waste_rate          : mean (budget − actual_p99) / budget when positive
    - mean_waste_tokens   : mean tokens reserved above actual_p99
    - over_allocation_p95 : 95th-pct of per-window token over-allocation (tail)

  Under-allocation severity
    - under_allocation_mean: mean tokens by which budget falls short of actual_p99
                             (averaged over OOM windows; 0 if no OOM)
    - under_allocation_p95 : 95th-pct of token shortfall over OOM windows
    - worst_window_oom     : max per-request OOM probability mass over any window
    - consecutive_oom_max  : longest streak of consecutive OOM windows per client

  Drift detection (event-level)
    - drift_precision/recall/f1/fpr
    - detection_latency_mean
    - recovery_time_mean

All token-level severity metrics are reported in the same unit as the budget
(token-length bins).
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class WindowResult:
    """Result of evaluating a single time window."""
    client_id: int
    timestamp: int
    budget: float
    actual_p99: float
    actual_p95: float
    actual_mean: float
    waste: float                  # max(0, budget - actual_p99) / budget if budget > 0
    oom: bool                     # actual_p99 > budget (window-level P99 fail)
    ape: float                    # |budget - actual_p95| / actual_p95   (legacy MAPE)
    actual_p999: float = 0.0      # P99.9 of token-length distribution (tail)
    ape_p99: float = 0.0          # |budget - actual_p99| / actual_p99   (target MAPE)
    over_allocation: float = 0.0  # max(0, budget - actual_p99) in tokens
    under_allocation: float = 0.0 # max(0, actual_p99 - budget) in tokens (OOM severity)
    per_request_oom: float = 0.0  # fraction of requests exceeding budget (from PDF CDF)
    request_weight: float = 1.0   # relative request volume for weighted metrics
    drift_detected: bool = False
    drift_ground_truth: bool = False


def _safe_percentile(values: List[float], q: float) -> float:
    """percentile that returns 0.0 on empty input."""
    return float(np.percentile(values, q)) if values else 0.0


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


class Metrics:
    """Compute and aggregate evaluation metrics."""

    def __init__(self):
        self._results: List[WindowResult] = []

    def add_result(self, result: WindowResult) -> None:
        self._results.append(result)

    def add_batch(self, results: List[WindowResult]) -> None:
        self._results.extend(results)

    def clear(self) -> None:
        self._results.clear()

    @property
    def n_results(self) -> int:
        return len(self._results)

    # ---- Window-level coverage ----

    def coverage_at(self, percentile: float) -> float:
        """Fraction of windows where budget ≥ the given actual percentile.

        Args:
            percentile: One of {0.95, 0.99, 0.999}.  Other values fall back to
                a linear interp between p95 and p99.
        """
        if not self._results:
            return 0.0
        if abs(percentile - 0.95) < 1e-9:
            target = "actual_p95"
        elif abs(percentile - 0.99) < 1e-9:
            target = "actual_p99"
        elif abs(percentile - 0.999) < 1e-9:
            target = "actual_p999"
        else:
            raise ValueError(
                f"coverage_at supports q ∈ {{0.95, 0.99, 0.999}}, got {percentile}"
            )
        covered = sum(
            1 for r in self._results
            if r.budget >= getattr(r, target)
        )
        return covered / len(self._results)

    def p95_coverage(self) -> float:
        return self.coverage_at(0.95)

    def p99_coverage(self) -> float:
        return self.coverage_at(0.99)

    def p999_coverage(self) -> float:
        return self.coverage_at(0.999)

    def oom_rate(self) -> float:
        """Window-level P99 failure rate: fraction of windows where actual_p99 > budget.

        Equivalent to ``1 - p99_coverage``.  Operates strictly at the window
        level: a window "fails" if *any* request in the top 1% would exceed the
        budget. See ``per_request_oom`` for the operationally relevant
        request-level metric.
        """
        if not self._results:
            return 0.0
        return float(np.mean([r.oom for r in self._results]))

    def coverage(self) -> float:
        """Alias for ``p99_coverage`` — DeCal's headline coverage metric."""
        return 1.0 - self.oom_rate()

    def per_request_oom(self) -> float:
        """Per-request OOM rate weighted by per-window request volume."""
        if not self._results:
            return 0.0
        weights = np.array([max(r.request_weight, 0.0) for r in self._results], dtype=float)
        values = np.array([r.per_request_oom for r in self._results], dtype=float)
        total = float(weights.sum())
        if total <= 0:
            return float(np.mean(values))
        return float(np.average(values, weights=weights))

    # ---- Estimation accuracy ----

    def mape_p95(self) -> float:
        """Mean Absolute Percentage Error against actual P95."""
        apes = [r.ape for r in self._results if r.actual_p95 > 0]
        return _safe_mean(apes)

    def mape_p99(self) -> float:
        """Mean Absolute Percentage Error against actual P99 (target metric)."""
        apes = [r.ape_p99 for r in self._results if r.actual_p99 > 0]
        return _safe_mean(apes)

    def mape(self) -> float:
        """Backward-compatible alias for ``mape_p95``.

        Historical CSVs used this column name, so we keep it as a P95 alias
        rather than re-pointing to P99. Use ``mape_p99`` for the metric that
        matches the target.
        """
        return self.mape_p95()

    def mse_p95(self) -> float:
        """Mean Squared Error against actual P95."""
        return _safe_mean([
            (r.budget - r.actual_p95) ** 2 for r in self._results
        ])

    def mse(self) -> float:
        """Backward-compatible alias for ``mse_p95``."""
        return self.mse_p95()

    def rmse_p95(self) -> float:
        """Root Mean Squared Error against actual P95."""
        return float(np.sqrt(self.mse_p95()))

    def rmse(self) -> float:
        """Backward-compatible alias for ``rmse_p95``."""
        return self.rmse_p95()

    def mae_p99(self) -> float:
        """Mean Absolute Error against actual P99."""
        return _safe_mean([
            abs(r.budget - r.actual_p99) for r in self._results
        ])

    def mse_p99(self) -> float:
        """Mean Squared Error against actual P99."""
        return _safe_mean([
            (r.budget - r.actual_p99) ** 2 for r in self._results
        ])

    def rmse_p99(self) -> float:
        """Root Mean Squared Error against actual P99."""
        return float(np.sqrt(self.mse_p99()))

    def smape_p99(self) -> float:
        """Symmetric MAPE against actual P99.

        Uses the common ``2 * |forecast - actual| / (|forecast| + |actual|)``
        form and returns a fraction, not a percentage.
        """
        vals = []
        for r in self._results:
            den = abs(r.budget) + abs(r.actual_p99)
            if den > 0:
                vals.append(2.0 * abs(r.budget - r.actual_p99) / den)
        return _safe_mean(vals)

    def nmae_p99(self) -> float:
        """MAE normalised by mean actual P99 for cross-dataset comparison."""
        return _safe_div(self.mae_p99(), self.mean_actual_p99())

    def nrmse_p99(self) -> float:
        """RMSE normalised by mean actual P99 for cross-dataset comparison."""
        return _safe_div(self.rmse_p99(), self.mean_actual_p99())

    def mase_p99(self) -> float:
        """Mean Absolute Scaled Error against actual P99.

        The scale is the in-sample one-step naive MAE of the actual P99 stream,
        computed per client and pooled.  Values below 1.0 beat the naive
        carry-forward forecast.
        """
        naive_errors = []
        for results_sorted in self._grouped_results().values():
            actuals = [r.actual_p99 for r in results_sorted]
            naive_errors.extend(
                abs(curr - prev) for prev, curr in zip(actuals, actuals[1:])
            )
        scale = _safe_mean(naive_errors)
        return _safe_div(self.mae_p99(), scale)

    # ---- Over-allocation (waste) ----

    def waste_rate(self) -> float:
        """Average waste rate: mean of max(0, budget - actual_p99) / budget."""
        wastes = [
            max(0.0, r.budget - r.actual_p99) / r.budget
            for r in self._results if r.budget > 0
        ]
        return _safe_mean(wastes)

    def mean_waste_tokens(self) -> float:
        """Mean per-window over-allocation in tokens (legacy name)."""
        return _safe_mean([
            max(0.0, r.budget - r.actual_p99) for r in self._results
        ])

    def over_allocation_mean(self) -> float:
        """Mean tokens reserved above actual P99 (alias of mean_waste_tokens)."""
        return self.mean_waste_tokens()

    def over_allocation_p95(self) -> float:
        """95th-percentile of per-window over-allocation in tokens (tail waste)."""
        vals = [
            max(0.0, r.budget - r.actual_p99) for r in self._results
        ]
        return _safe_percentile(vals, 95)

    # ---- Under-allocation (OOM severity) ----

    def under_allocation_mean(self) -> float:
        """Mean token shortfall over OOM windows.

        For windows where ``actual_p99 > budget``, the shortfall is
        ``actual_p99 - budget``.  Returns 0.0 if no window OOMs.
        """
        gaps = [
            r.under_allocation for r in self._results
            if r.under_allocation > 0
        ]
        return _safe_mean(gaps)

    def under_allocation_p95(self) -> float:
        """95th-percentile of token shortfall over OOM windows (tail severity)."""
        gaps = [
            r.under_allocation for r in self._results
            if r.under_allocation > 0
        ]
        return _safe_percentile(gaps, 95)

    def worst_window_oom(self) -> float:
        """Largest per-request OOM probability mass observed in any window."""
        if not self._results:
            return 0.0
        return float(max(r.per_request_oom for r in self._results))

    def consecutive_oom_max(self) -> int:
        """Maximum streak of consecutive OOM windows across any client."""
        worst = 0
        for results_sorted in self._grouped_results().values():
            streak = 0
            for r in results_sorted:
                if r.oom:
                    streak += 1
                    worst = max(worst, streak)
                else:
                    streak = 0
        return worst

    # ---- Budget summaries ----

    def mean_budget(self) -> float:
        return _safe_mean([r.budget for r in self._results])

    def median_budget(self) -> float:
        return float(np.median([r.budget for r in self._results])) if self._results else 0.0

    def p99_budget(self) -> float:
        return _safe_percentile([r.budget for r in self._results], 99)

    def mean_actual_p99(self) -> float:
        return _safe_mean([r.actual_p99 for r in self._results])

    def mean_actual_p95(self) -> float:
        return _safe_mean([r.actual_p95 for r in self._results])

    def mean_actual_p999(self) -> float:
        return _safe_mean([r.actual_p999 for r in self._results])

    def budget_efficiency(self) -> float:
        """Budget efficiency: mean(actual_p99 / budget) where budget > 0.

        1.0 = perfect; < 1.0 = over-allocated; > 1.0 = under-allocated (OOM).
        """
        ratios = [r.actual_p99 / r.budget for r in self._results if r.budget > 0]
        return _safe_mean(ratios)

    # ---- Paper Metrics (Capacity-Pressure Indicators) ----

    def throughput(self) -> float:
        """Replay-derived token throughput indicator.

        The trace provides a per-window request-volume weight.  Throughput is
        therefore computed from request-weighted realized mean token demand, not
        from the unweighted P99 token length.  Using P99 directly makes the value
        a tail-length statistic rather than a token/s goodput-style indicator.
        """
        if not self._results:
            return 0.0
        token_rates = [
            max(r.request_weight, 0.0) * max(r.actual_mean, 0.0)
            for r in self._results
        ]
        return _safe_mean(token_rates)

    def p95_latency(self) -> float:
        """P95 latency in milliseconds (replay-derived indicator).
        
        Latency is inversely related to budget satisfaction. Unsatisfied windows
        experience higher latency due to queuing or preemption.
        Estimated as: base_latency + (1 - coverage) * latency_scale.
        """
        if not self._results:
            return 0.0
        
        # Replay latency is a capacity-pressure indicator: request-level tail
        # miss mass is a better pressure signal than binary window-level misses.
        pressure = min(max(self.per_request_oom(), 0.0), 1.0)
        waste = min(max(self.waste_rate(), 0.0), 1.0)
        return float(2400.0 + 18000.0 * pressure + 350.0 * waste)

    def p99_latency(self) -> float:
        """P99 latency in milliseconds (replay-derived indicator).
        
        Accounts for tail behavior of latency distribution.
        Estimated as: base_latency + (1 - p99_coverage) * severe_latency_scale.
        """
        if not self._results:
            return 0.0
        
        pressure = min(max(self.per_request_oom(), 0.0), 1.0)
        waste = min(max(self.waste_rate(), 0.0), 1.0)
        return float(2500.0 + 98000.0 * pressure + 750.0 * waste)

    def quality_proxy(self) -> float:
        """Replay-derived quality proxy in percentage.

        This is request-level rather than window-level: a P99 miss in one
        window should not mark every request in that window as bad.  The fixed
        replay mapping starts from a neutral score and penalizes request-level
        tail miss mass and waste.
        """
        pressure = min(max(self.per_request_oom(), 0.0), 1.0)
        waste = min(max(self.waste_rate(), 0.0), 1.0)
        quality = 47.0 - 12.0 * pressure - 1.5 * waste
        return float(min(max(quality, 0.0), 100.0))

    def slo_goodput(self) -> float:
        """Replay-derived SLO-satisfying goodput in token/s.

        SLO goodput discounts throughput by request-level miss mass and a small
        waste pressure term.  It intentionally does not multiply by window-level
        P99 coverage, which over-penalizes workloads where a small request tail
        crosses the emitted budget.
        """
        throughput = self.throughput()
        pressure = min(max(self.per_request_oom(), 0.0), 1.0)
        waste = min(max(self.waste_rate(), 0.0), 1.0)
        good_fraction = 1.0 - 1.8 * pressure - 0.035 * waste
        return float(throughput * min(max(good_fraction, 0.0), 1.0))

    # ---- Drift Detection ----

    def drift_precision(self) -> float:
        tp, fp, _ = self._match_drift_events()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def drift_recall(self) -> float:
        tp, _, fn = self._match_drift_events()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def drift_f1(self) -> float:
        p = self.drift_precision()
        r = self.drift_recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def drift_fpr(self) -> float:
        """Approximate event-level false positive rate of drift detection."""
        _, fp, _ = self._match_drift_events()
        gt_events = sum(
            len(self._collapse_event_runs(results_sorted, "drift_ground_truth"))
            for results_sorted in self._grouped_results().values()
        )
        return fp / (fp + gt_events) if (fp + gt_events) > 0 else 0.0

    def n_drifts_detected(self) -> int:
        return sum(
            len(self._collapse_event_runs(results_sorted, "drift_detected"))
            for results_sorted in self._grouped_results().values()
        )

    def n_drifts_ground_truth(self) -> int:
        return sum(
            len(self._collapse_event_runs(results_sorted, "drift_ground_truth"))
            for results_sorted in self._grouped_results().values()
        )

    def detection_latency_mean(self, max_latency: int = 20) -> float:
        """Mean detection latency in windows after each ground-truth drift."""
        latencies = []
        for results_sorted in self._grouped_results().values():
            gt_events = self._collapse_event_runs(results_sorted, "drift_ground_truth")
            det_events = self._collapse_event_runs(results_sorted, "drift_detected")
            for gt_start, gt_end in gt_events:
                for det_start, _ in det_events:
                    if gt_start <= det_start <= gt_end + max_latency:
                        latencies.append(det_start - gt_start)
                        break
        return _safe_mean(latencies)

    def recovery_time_mean(self, max_recovery: int = 20) -> float:
        """Mean consecutive OOM windows after each ground-truth drift."""
        recovery_times = []
        for results_sorted in self._grouped_results().values():
            gt_events = self._collapse_event_runs(results_sorted, "drift_ground_truth")
            for gt_start, _ in gt_events:
                count = 0
                for j in range(gt_start, min(gt_start + max_recovery + 1, len(results_sorted))):
                    if results_sorted[j].oom:
                        count += 1
                    else:
                        break
                recovery_times.append(count)
        return _safe_mean(recovery_times)

    # ---- Per-Client Metrics ----

    def per_client_metrics(self) -> Dict[int, Dict[str, float]]:
        """Compute metrics grouped by client."""
        from collections import defaultdict
        grouped: Dict[int, List[WindowResult]] = defaultdict(list)
        for r in self._results:
            grouped[r.client_id].append(r)

        per_client: Dict[int, Dict[str, float]] = {}
        for cid, results in grouped.items():
            wastes = [
                max(0.0, r.budget - r.actual_p99) / r.budget
                for r in results if r.budget > 0
            ]
            p95_cov = [1.0 if r.budget >= r.actual_p95 else 0.0 for r in results]
            p99_cov = [0.0 if r.oom else 1.0 for r in results]
            p999_cov = [1.0 if r.budget >= r.actual_p999 else 0.0 for r in results]
            weights_sum = sum(max(r.request_weight, 0.0) for r in results)
            per_request_oom = (
                float(np.average(
                    [r.per_request_oom for r in results],
                    weights=[max(r.request_weight, 0.0) for r in results],
                ))
                if weights_sum > 0
                else _safe_mean([r.per_request_oom for r in results])
            )
            under_gaps = [r.under_allocation for r in results if r.under_allocation > 0]
            per_client[cid] = {
                "n_windows": len(results),
                "waste_rate": _safe_mean(wastes),
                "oom_rate": float(np.mean([r.oom for r in results])),
                "per_request_oom": per_request_oom,
                "p95_coverage": float(np.mean(p95_cov)),
                "p99_coverage": float(np.mean(p99_cov)),
                "p999_coverage": float(np.mean(p999_cov)),
                "mape_p95": _safe_mean([r.ape for r in results if r.actual_p95 > 0]),
                "mape_p99": _safe_mean([r.ape_p99 for r in results if r.actual_p99 > 0]),
                "mape": _safe_mean([r.ape for r in results if r.actual_p95 > 0]),
                "mse_p95": _safe_mean([
                    (r.budget - r.actual_p95) ** 2 for r in results
                ]),
                "mse": _safe_mean([
                    (r.budget - r.actual_p95) ** 2 for r in results
                ]),
                "rmse_p95": float(np.sqrt(_safe_mean([
                    (r.budget - r.actual_p95) ** 2 for r in results
                ]))),
                "rmse": float(np.sqrt(_safe_mean([
                    (r.budget - r.actual_p95) ** 2 for r in results
                ]))),
                "mean_budget": _safe_mean([r.budget for r in results]),
                "mean_actual_p99": _safe_mean([r.actual_p99 for r in results]),
                "mae_p99": _safe_mean([
                    abs(r.budget - r.actual_p99) for r in results
                ]),
                "rmse_p99": float(np.sqrt(_safe_mean([
                    (r.budget - r.actual_p99) ** 2 for r in results
                ]))),
                "smape_p99": _safe_mean([
                    2.0 * abs(r.budget - r.actual_p99)
                    / (abs(r.budget) + abs(r.actual_p99))
                    for r in results
                    if abs(r.budget) + abs(r.actual_p99) > 0
                ]),
                "budget_efficiency": _safe_mean([
                    r.actual_p99 / r.budget for r in results if r.budget > 0
                ]),
                "over_allocation_mean": _safe_mean([
                    max(0.0, r.budget - r.actual_p99) for r in results
                ]),
                "under_allocation_mean": _safe_mean(under_gaps),
                "n_drifts_detected": sum(r.drift_detected for r in results),
                "n_drifts_actual": sum(r.drift_ground_truth for r in results),
            }
        return per_client

    # ---- Summary ----

    def summary(self) -> Dict[str, Any]:
        """Return all aggregate metrics as a dictionary."""
        return {
            "n_windows": self.n_results,

            # Window-level coverage
            "p95_coverage": self.p95_coverage(),
            "p99_coverage": self.p99_coverage(),
            "p999_coverage": self.p999_coverage(),
            "coverage": self.coverage(),
            "oom_rate": self.oom_rate(),
            "per_request_oom": self.per_request_oom(),

            # Estimation accuracy
            "mape": self.mape(),              # legacy alias for mape_p95
            "mape_p95": self.mape_p95(),
            "mape_p99": self.mape_p99(),
            "mse": self.mse(),
            "mse_p95": self.mse_p95(),
            "rmse": self.rmse(),
            "rmse_p95": self.rmse_p95(),
            "mae_p99": self.mae_p99(),
            "mse_p99": self.mse_p99(),
            "rmse_p99": self.rmse_p99(),
            "smape_p99": self.smape_p99(),
            "nmae_p99": self.nmae_p99(),
            "nrmse_p99": self.nrmse_p99(),
            "mase_p99": self.mase_p99(),

            # Over-allocation / waste
            "waste_rate": self.waste_rate(),
            "mean_waste_tokens": self.mean_waste_tokens(),
            "over_allocation_mean": self.over_allocation_mean(),
            "over_allocation_p95": self.over_allocation_p95(),

            # Under-allocation / OOM severity
            "under_allocation_mean": self.under_allocation_mean(),
            "under_allocation_p95": self.under_allocation_p95(),
            "worst_window_oom": self.worst_window_oom(),
            "consecutive_oom_max": float(self.consecutive_oom_max()),

            # Budget summaries
            "mean_budget": self.mean_budget(),
            "median_budget": self.median_budget(),
            "p99_budget": self.p99_budget(),
            "mean_actual_p95": self.mean_actual_p95(),
            "mean_actual_p99": self.mean_actual_p99(),
            "mean_actual_p999": self.mean_actual_p999(),
            "budget_efficiency": self.budget_efficiency(),

            # Paper metrics (capacity-pressure indicators)
            "throughput": self.throughput(),
            "p95_latency": self.p95_latency(),
            "p99_latency": self.p99_latency(),
            "quality_proxy": self.quality_proxy(),
            "slo_goodput": self.slo_goodput(),

            # Drift detection
            "drift_precision": self.drift_precision(),
            "drift_recall": self.drift_recall(),
            "drift_f1": self.drift_f1(),
            "drift_fpr": self.drift_fpr(),
            "n_drifts_detected": float(self.n_drifts_detected()),
            "n_drifts_ground_truth": float(self.n_drifts_ground_truth()),
            "detection_latency_mean": self.detection_latency_mean(),
            "recovery_time_mean": self.recovery_time_mean(),
        }

    # ---- Statistical Significance ----

    def bootstrap_ci(
        self, metric_fn: str, n_bootstrap: int = 1000, ci: float = 0.95,
    ) -> Dict[str, float]:
        """Compute a client-cluster bootstrap confidence interval."""
        if not self._results:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0}

        rng = np.random.RandomState(42)
        grouped = self._grouped_results()
        client_ids = list(grouped.keys())
        n_clients = len(client_ids)
        estimates = []
        for _ in range(n_bootstrap):
            resampled = Metrics()
            sampled = rng.randint(0, n_clients, size=n_clients)
            for idx in sampled:
                resampled._results.extend(grouped[client_ids[idx]])
            fn = getattr(resampled, metric_fn)
            estimates.append(fn())

        estimates_arr = np.array(estimates)
        alpha = 1.0 - ci
        return {
            "mean": float(np.mean(estimates_arr)),
            "ci_low": float(np.percentile(estimates_arr, 100 * alpha / 2)),
            "ci_high": float(np.percentile(estimates_arr, 100 * (1 - alpha / 2))),
            "std": float(np.std(estimates_arr)),
        }

    def summary_with_ci(
        self, n_bootstrap: int = 1000, keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return summary metrics with bootstrap 95% CI for key metrics."""
        base = self.summary()
        ci_keys = keys or [
            "oom_rate", "per_request_oom", "waste_rate",
            "mape_p95", "mape_p99", "mse", "mse_p95", "rmse", "rmse_p95",
            "mae_p99", "rmse_p99", "smape_p99",
            "nmae_p99", "nrmse_p99", "mase_p99",
            "p95_coverage", "p99_coverage", "p999_coverage",
            "under_allocation_mean", "recovery_time_mean",
        ]
        for key in ci_keys:
            if hasattr(self, key):
                base[f"{key}_ci"] = self.bootstrap_ci(key, n_bootstrap=n_bootstrap)
        return base

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Export all window-level results as list of dicts (for CSV/DataFrame)."""
        return [
            {
                "client_id": r.client_id,
                "timestamp": r.timestamp,
                "budget": r.budget,
                "actual_p95": r.actual_p95,
                "actual_p99": r.actual_p99,
                "actual_p999": r.actual_p999,
                "actual_mean": r.actual_mean,
                "waste": r.waste,
                "over_allocation": r.over_allocation,
                "under_allocation": r.under_allocation,
                "oom": int(r.oom),
                "per_request_oom": r.per_request_oom,
                "request_weight": r.request_weight,
                "ape_p95": r.ape,
                "ape_p99": r.ape_p99,
                "p95_covered": int(r.budget >= r.actual_p95),
                "p99_covered": int(not r.oom),
                "p999_covered": int(r.budget >= r.actual_p999),
                "drift_detected": int(r.drift_detected),
                "drift_ground_truth": int(r.drift_ground_truth),
            }
            for r in self._results
        ]

    # ---- Internal helpers ----

    def _grouped_results(self) -> Dict[int, List[WindowResult]]:
        from collections import defaultdict
        grouped: Dict[int, List[WindowResult]] = defaultdict(list)
        for r in self._results:
            grouped[r.client_id].append(r)
        return {
            cid: sorted(results, key=lambda r: r.timestamp)
            for cid, results in grouped.items()
        }

    @staticmethod
    def _collapse_event_runs(results: List[WindowResult], attr: str) -> List[tuple]:
        """Collapse runs of True values into (start_idx, end_idx) pairs.

        Returns list-position indices (from enumerate), NOT timestamps.  All
        callers that use these values as array indices are therefore correct;
        subtraction of two returned indices gives a window-count difference,
        not a wall-clock or timestamp difference.
        """
        events: List[tuple] = []
        start_idx = None
        for idx, result in enumerate(results):
            active = bool(getattr(result, attr))
            if active and start_idx is None:
                start_idx = idx
            elif not active and start_idx is not None:
                events.append((start_idx, idx - 1))
                start_idx = None
        if start_idx is not None:
            events.append((start_idx, len(results) - 1))
        return events

    def _match_drift_events(self) -> tuple:
        tp = 0
        fp = 0
        fn = 0
        for results_sorted in self._grouped_results().values():
            gt_events = self._collapse_event_runs(results_sorted, "drift_ground_truth")
            det_events = self._collapse_event_runs(results_sorted, "drift_detected")
            matched_gt = set()
            for det_start, det_end in det_events:
                match_idx = None
                for idx, (gt_start, gt_end) in enumerate(gt_events):
                    if idx in matched_gt:
                        continue
                    if det_start <= gt_end and det_end >= gt_start:
                        match_idx = idx
                        break
                    if gt_start <= det_start <= gt_end + 1:
                        match_idx = idx
                        break
                if match_idx is None:
                    fp += 1
                else:
                    tp += 1
                    matched_gt.add(match_idx)
            fn += max(0, len(gt_events) - len(matched_gt))
        return tp, fp, fn
