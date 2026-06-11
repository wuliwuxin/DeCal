"""
CUSUM (Cumulative Sum) Drift Detector.

Classic sequential change-point detection method. Maintains both
positive and negative cumulative sums to detect upward and downward shifts.
"""

from typing import Dict, Any
import numpy as np

from decal.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    DriftLevel,
)


class CUSUMDetector(BaseDriftDetector):
    """
    Two-sided CUSUM detector with EWMA baseline.

    Parameters:
        alpha: EWMA decay factor for baseline mean (default 0.3).
        threshold: Detection threshold h (default 8.0).
        drift_magnitude: Expected shift magnitude k (default 0.5).
        min_samples: Minimum observations before detection (default 3).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        threshold: float = 8.0,
        drift_magnitude: float = 0.5,
        min_samples: int = 3,
    ):
        super().__init__(name="CUSUM")
        self.alpha = alpha
        self.threshold = threshold
        self.drift_magnitude = drift_magnitude
        self.min_samples = min_samples

    def _init_state(self) -> Dict[str, Any]:
        return {
            "mu_hat": 0.0,
            "sigma_hat": 1.0,
            "s_pos": 0.0,
            "s_neg": 0.0,
            "n": 0,
            "values": [],
        }

    def update(self, client_id: int, timestamp: int, value: float) -> DriftSignal:
        state = self._get_or_create_state(client_id)
        state["n"] += 1
        n = state["n"]

        # Update EWMA baseline
        if n == 1:
            state["mu_hat"] = value
            state["sigma_hat"] = 1.0
        else:
            prev_mu = state["mu_hat"]
            state["mu_hat"] = self.alpha * value + (1 - self.alpha) * prev_mu
            diff = value - prev_mu
            state["sigma_hat"] = np.sqrt(
                self.alpha * diff ** 2 + (1 - self.alpha) * state["sigma_hat"] ** 2
            )

        state["values"].append(value)
        if len(state["values"]) > 50:
            state["values"] = state["values"][-50:]

        # Normalize
        sigma = max(state["sigma_hat"], 1e-6)
        z = (value - state["mu_hat"]) / sigma

        # Two-sided CUSUM
        k = self.drift_magnitude
        state["s_pos"] = max(0.0, state["s_pos"] + z - k)
        state["s_neg"] = max(0.0, state["s_neg"] - z - k)
        cusum_stat = max(state["s_pos"], state["s_neg"])

        # Detection
        level = DriftLevel.NONE
        if n >= self.min_samples:
            if cusum_stat > self.threshold * 2:
                level = DriftLevel.SEVERE_DRIFT
            elif cusum_stat > self.threshold:
                level = DriftLevel.DRIFT
            elif cusum_stat > self.threshold * 0.7:
                level = DriftLevel.WARNING

        signal = DriftSignal(
            timestamp=timestamp,
            client_id=client_id,
            level=level,
            statistic=cusum_stat,
            threshold=self.threshold,
            metadata={
                "s_pos": state["s_pos"],
                "s_neg": state["s_neg"],
                "mu_hat": state["mu_hat"],
            },
        )
        self._history.append(signal)

        if level in (DriftLevel.DRIFT, DriftLevel.SEVERE_DRIFT):
            state["s_pos"] = 0.0
            state["s_neg"] = 0.0

        return signal
