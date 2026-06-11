"""
ADWIN (Adaptive Windowing) Drift Detector.

Bifet & Gavalda (SDM'07). Maintains a variable-length window of recent
observations and detects drift by comparing sub-window means.
Simplified implementation for LLM serving workloads.
"""

from typing import Dict, Any, List
import numpy as np

from decal.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    DriftLevel,
)


class ADWINDetector(BaseDriftDetector):
    """
    Simplified ADWIN detector.

    Parameters:
        delta: Confidence parameter (default 0.01). Lower = fewer false positives.
        max_window: Maximum window size to maintain (default 100).
        min_window: Minimum sub-window size for comparison (default 5).
        min_samples: Minimum observations before detection (default 5).
    """

    def __init__(
        self,
        delta: float = 0.01,
        max_window: int = 100,
        min_window: int = 5,
        min_samples: int = 5,
    ):
        super().__init__(name="ADWIN")
        self.delta = delta
        self.max_window = max_window
        self.min_window = min_window
        self.min_samples = min_samples

    def _init_state(self) -> Dict[str, Any]:
        return {
            "window": [],
            "n": 0,
        }

    def _compute_threshold(self, n1: int, n2: int) -> float:
        """Hoeffding bound threshold for two sub-windows."""
        n = n1 + n2
        m = 1.0 / (1.0 / n1 + 1.0 / n2)
        eps = np.sqrt(np.log(4.0 / self.delta) / (2.0 * m))
        return eps

    def update(self, client_id: int, timestamp: int, value: float) -> DriftSignal:
        state = self._get_or_create_state(client_id)
        state["n"] += 1
        window: List[float] = state["window"]
        window.append(value)

        # Trim window
        if len(window) > self.max_window:
            window.pop(0)

        # Check for drift by comparing all valid splits
        level = DriftLevel.NONE
        best_stat = 0.0
        best_threshold = 0.0

        if state["n"] >= self.min_samples and len(window) >= 2 * self.min_window:
            for split in range(self.min_window, len(window) - self.min_window + 1):
                w0 = window[:split]
                w1 = window[split:]
                mu0 = np.mean(w0)
                mu1 = np.mean(w1)
                diff = abs(mu0 - mu1)
                eps = self._compute_threshold(len(w0), len(w1))

                if diff > eps and diff > best_stat:
                    best_stat = diff
                    best_threshold = eps

            if best_stat > 0:
                ratio = best_stat / best_threshold if best_threshold > 0 else 0
                if ratio > 2.0:
                    level = DriftLevel.SEVERE_DRIFT
                elif ratio > 1.0:
                    level = DriftLevel.DRIFT
                elif ratio > 0.7:
                    level = DriftLevel.WARNING

        # Shrink window after drift
        if level in (DriftLevel.DRIFT, DriftLevel.SEVERE_DRIFT):
            # Keep only the recent sub-window
            mid = len(window) // 2
            state["window"] = window[mid:]
        else:
            state["window"] = window

        signal = DriftSignal(
            timestamp=timestamp,
            client_id=client_id,
            level=level,
            statistic=best_stat,
            threshold=best_threshold,
            metadata={"window_size": len(state["window"])},
        )
        self._history.append(signal)
        return signal
