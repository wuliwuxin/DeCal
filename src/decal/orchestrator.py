"""
DeCal Orchestrator.

Central controller that ties together:
  Core:
    1. Per-client drift detection (cheap scalar trigger)
    2. Drift-adaptive budget allocation (decoupled calibration under drift)

  Optional extension:
    3. Cross-client distribution transfer (cold-start / severe-drift recovery)

Designed as a sidecar to an LLM serving engine (e.g., vLLM).
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import logging

from decal.detectors.base import BaseDriftDetector, DriftSignal, DriftLevel
from decal.detectors.page_hinkley import PageHinkleyDetector
from decal.allocators.base import BaseAllocator, AllocationResult
from decal.allocators.drift_adaptive import DriftAdaptiveAllocator
from decal.transfer.wasserstein import WassersteinTransfer
from decal.transfer.similarity_graph import ClientSimilarityGraph

logger = logging.getLogger(__name__)


@dataclass
class ClientState:
    """Runtime state for a single client in the orchestrator."""
    client_id: int
    is_cold_start: bool = True
    total_windows: int = 0
    total_drifts: int = 0
    last_drift_window: int = -1
    current_budget: float = 0.0
    current_safety_factor: float = 1.0
    # Causal protocol: stat extracted from window t-1, used for drift detection
    # before window t's budget is allocated.  Populated after each allocation.
    prev_stat: Optional[float] = None


@dataclass
class OrchestratorConfig:
    """Configuration for the DeCal orchestrator."""
    # Drift detection
    detector_type: str = "page_hinkley"
    ph_alpha: float = 0.3
    ph_lambda: float = 5.0
    ph_delta: float = 0.3

    # Allocation
    coverage_target: float = 0.95
    alloc_alpha: float = 0.3
    sf_boost: float = 1.2
    sf_severe_boost: float = 1.4
    sf_decay_rate: float = 0.15       # multiplicative decay rate per NONE-window
    sf_hold_windows: int = 3          # windows to hold boosted sf before decaying
    calibration_size: int = 100
    base_percentile: float = 0.95     # EWMA percentile used as base budget
    base_estimator: str = "pdf_quantile"  # "pdf_quantile" | "scalar_ewma" | "max_dual"
    bootstrap_calibration: bool = False  # bootstrap conformal scores (seed-sensitive)
    seed: Optional[int] = None        # propagated to allocator's bootstrap RNG

    # Transfer
    transfer_top_k: int = 3
    transfer_max_distance: float = float("inf")
    similarity_threshold: float = 0.1
    transfer_update_interval: int = 6  # Re-compute similarity every N windows

    # Cold-start
    cold_start_windows: int = 3  # Windows before client is "warm"

    # Feature extraction
    stat_type: str = "mean"  # "mean" or "p95" for drift monitoring


class DeCalOrchestrator:
    """
    Main orchestrator for DeCal.

    Usage:
        orch = DeCalOrchestrator(config)

        # For each time window:
        for t, distributions in enumerate(stream):
            for client_id, pdf in distributions.items():
                budget = orch.process_window(client_id, t, pdf)
                # Use budget for KV cache allocation
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._setup_components()
        self._client_states: Dict[int, ClientState] = {}
        self._global_step = 0
        self._pending_transfer_updates: Dict[int, List[float]] = {}

    def _transfer_enabled(self) -> bool:
        return bool(getattr(self.config, "transfer_top_k", 0) and self.config.transfer_top_k > 0)

    def _setup_components(self) -> None:
        """Initialize all sub-components."""
        cfg = self.config

        # Drift detector
        if cfg.detector_type == "page_hinkley":
            self.detector = PageHinkleyDetector(
                alpha=cfg.ph_alpha,
                lam=cfg.ph_lambda,
                delta=cfg.ph_delta,
            )
        elif cfg.detector_type == "cusum":
            from decal.detectors.cusum import CUSUMDetector
            self.detector = CUSUMDetector(alpha=cfg.ph_alpha)
        elif cfg.detector_type == "ewma":
            from decal.detectors.ewma_detector import EWMADetector
            self.detector = EWMADetector(alpha=cfg.ph_alpha)
        elif cfg.detector_type == "adwin":
            from decal.detectors.adwin import ADWINDetector
            self.detector = ADWINDetector()
        else:
            raise ValueError(f"Unknown detector type: {cfg.detector_type}")

        # Adaptive allocator
        self.allocator = DriftAdaptiveAllocator(
            coverage_target=cfg.coverage_target,
            alpha_ewma=cfg.alloc_alpha,
            sf_boost=cfg.sf_boost,
            sf_severe_boost=cfg.sf_severe_boost,
            sf_decay_rate=cfg.sf_decay_rate,
            sf_hold_windows=cfg.sf_hold_windows,
            calibration_size=cfg.calibration_size,
            base_percentile=cfg.base_percentile,
            base_estimator=cfg.base_estimator,
            bootstrap_calibration=cfg.bootstrap_calibration,
            seed=cfg.seed,
        )

        # Optional cross-client transfer (extension). Disabled when top_k <= 0.
        if self._transfer_enabled():
            self.transfer = WassersteinTransfer(
                top_k=cfg.transfer_top_k,
                max_distance=cfg.transfer_max_distance,
            )
            self.similarity_graph = ClientSimilarityGraph(
                transfer=self.transfer,
                similarity_threshold=cfg.similarity_threshold,
            )
        else:
            self.transfer = None
            self.similarity_graph = None

    def _get_client_state(self, client_id: int) -> ClientState:
        if client_id not in self._client_states:
            self._client_states[client_id] = ClientState(client_id=client_id)
        return self._client_states[client_id]

    def _extract_statistic(self, pdf: List[float]) -> float:
        """Extract a scalar statistic from PDF for drift monitoring."""
        if not pdf or sum(pdf) == 0:
            return 0.0
        if self.config.stat_type == "p95":
            cdf = np.cumsum(pdf)
            cdf = cdf / cdf[-1]
            idx = np.searchsorted(cdf, 0.95)
            return float(min(idx, len(pdf) - 1))
        else:  # "mean"
            return float(sum(i * p for i, p in enumerate(pdf)))

    def process_window(
        self,
        client_id: int,
        timestamp: int,
        pdf: List[float],
        commit_transfer_update: bool = True,
    ) -> AllocationResult:
        """
        Process a single time window for a client.

        This is the main entry point called every window (e.g., every 10 min).

        Args:
            client_id: Client identifier.
            timestamp: Window index.
            pdf: Token length distribution for this window.

        Returns:
            AllocationResult with the KV cache budget.
        """
        cs = self._get_client_state(client_id)
        cs.total_windows += 1

        # --- Step 1: Handle cold start ---
        if (
            self.transfer is not None
            and cs.is_cold_start
            and cs.total_windows <= self.config.cold_start_windows
        ):
            # Cold-start transfer becomes causal only after at least one committed
            # snapshot exists for this client. The first window therefore serves as
            # observation; windows 2..K may use cross-client transfer.
            if cs.total_windows >= 2 and self.transfer.get_distribution(client_id) is not None:
                prior = self.transfer.transfer(client_id)
                if prior is not None:
                    self.allocator.inject_prior(client_id, prior)
                    logger.info(
                        f"Client {client_id}: injected transfer prior "
                        f"(cold-start, {self.transfer.n_donors()} donors)"
                    )

        if cs.total_windows > self.config.cold_start_windows:
            cs.is_cold_start = False

        # --- Step 2: Drift detection using PREVIOUS window's statistic (causal) ---
        # The statistic from window t is extracted and cached AFTER allocation
        # (Step 5 below), so that the detector only ever sees t-1 data when
        # deciding the safety factor for window t.  This is the fix for the
        # causal-leakage issue identified in review C1: previously the code
        # extracted stat from the current-window pdf before allocation, giving
        # the safety-factor path non-causal access to window-t information.
        if cs.prev_stat is not None:
            drift_signal = self.detector.update(client_id, timestamp, cs.prev_stat)
        else:
            # First window: no previous statistic yet → no drift signal
            drift_signal = DriftSignal(
                client_id=client_id,
                timestamp=timestamp,
                level=DriftLevel.NONE,
                statistic=0.0,
                threshold=0.0,
            )

        # --- Step 3: Notify allocator of drift status ---
        self.allocator.notify_drift(client_id, drift_signal)

        # Handle severe drift: optionally try transfer for recovery
        if drift_signal.level == DriftLevel.SEVERE_DRIFT:
            cs.total_drifts += 1
            cs.last_drift_window = timestamp
            if self.transfer is not None:
                prior = self.transfer.transfer(client_id)
                if prior is not None:
                    self.allocator.inject_prior(client_id, prior)
                    logger.info(
                        f"Client {client_id}: severe drift at t={timestamp}, "
                        f"injected transfer prior for recovery"
                    )
        elif drift_signal.is_drift:
            cs.total_drifts += 1
            cs.last_drift_window = timestamp

        # --- Step 4: Compute budget (uses EWMA from previous windows only) ---
        result = self.allocator.allocate(client_id, timestamp)
        cs.current_budget = result.budget
        cs.current_safety_factor = result.safety_factor

        # --- Step 5: Update allocator with current distribution ---
        # Must happen AFTER allocate() to preserve causal order: budget for window t
        # is set using distributions from windows 0..t-1, not window t itself.
        self.allocator.update_distribution(client_id, timestamp, pdf)

        # Cache the statistic from window t so it can be used by the detector
        # when computing the budget for window t+1.
        cs.prev_stat = self._extract_statistic(pdf)

        # --- Step 6: Update transfer pool (optional extension) ---
        # In trace-driven evaluation we commit transfer snapshots once per
        # timestamp batch to avoid same-window information leakage across clients.
        if self.transfer is not None:
            if commit_transfer_update:
                self.transfer.update_distribution(client_id, pdf)
            else:
                self._pending_transfer_updates[client_id] = pdf

        # Periodically rebuild similarity graph
        self._global_step += 1
        if self.similarity_graph is not None:
            if self._global_step % (
                self.config.transfer_update_interval * max(len(self._client_states), 1)
            ) == 0:
                try:
                    self.similarity_graph.build()
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Similarity graph rebuild failed: {e}")

        return result

    def commit_transfer_updates(self) -> None:
        """Commit any deferred transfer-pool updates."""
        if self.transfer is None or not self._pending_transfer_updates:
            return
        for client_id, pdf in self._pending_transfer_updates.items():
            self.transfer.update_distribution(client_id, pdf)
        self._pending_transfer_updates.clear()

    def get_all_budgets(self) -> Dict[int, float]:
        """Get current budgets for all active clients."""
        return {
            cid: cs.current_budget
            for cid, cs in self._client_states.items()
        }

    def get_client_summary(self, client_id: int) -> Dict[str, Any]:
        """Get summary statistics for a client."""
        cs = self._get_client_state(client_id)
        return {
            "client_id": client_id,
            "total_windows": cs.total_windows,
            "total_drifts": cs.total_drifts,
            "is_cold_start": cs.is_cold_start,
            "current_budget": cs.current_budget,
            "current_safety_factor": cs.current_safety_factor,
            "last_drift_window": cs.last_drift_window,
        }

    def get_global_summary(self) -> Dict[str, Any]:
        """Get global orchestrator statistics."""
        n_clients = len(self._client_states)
        total_drifts = sum(cs.total_drifts for cs in self._client_states.values())
        budgets = [cs.current_budget for cs in self._client_states.values()]

        return {
            "n_clients": n_clients,
            "total_drifts": total_drifts,
            "global_step": self._global_step,
            "avg_budget": float(np.mean(budgets)) if budgets else 0.0,
            "max_budget": float(np.max(budgets)) if budgets else 0.0,
            "min_budget": float(np.min(budgets)) if budgets else 0.0,
        }

    def reset(self) -> None:
        """Reset all state."""
        self._setup_components()
        self._client_states.clear()
        self._global_step = 0
        self._pending_transfer_updates.clear()
