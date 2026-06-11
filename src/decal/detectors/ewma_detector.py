"""
EWMA-based Drift Detector.

Detects drift when the current observation deviates significantly from
the EWMA estimate, using a z-score approach with adaptive variance.
"""

from typing import Dict, Any
import numpy as np

from decal.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    DriftLevel,
)


class EWMADetector(BaseDriftDetector):
    """
    EWMA control-chart-style drift detector.

    Parameters:
        alpha: EWMA decay factor (default 0.3).
        z_warning: Z-score threshold for warning (default 2.0).
        z_drift: Z-score threshold for drift (default 3.0).
        z_severe: Z-score threshold for severe drift (default 4.5).
        min_samples: Minimum observations before detection (default 5).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        z_warning: float = 2.0,
        z_drift: float = 3.0,
        z_severe: float = 4.5,
        min_samples: int = 5,
    ):
        super().__init__(name="EWMA")
        self.alpha = alpha
        self.z_warning = z_warning
        self.z_drift = z_drift
        self.z_severe = z_severe
        self.min_samples = min_samples

    def _init_state(self) -> Dict[str, Any]:
        return {
            "mu_hat": 0.0,
            "sigma_hat": 1.0,
            "n": 0,
        }

    def update(self, client_id: int, timestamp: int, value: float) -> DriftSignal:
        state = self._get_or_create_state(client_id)
        state["n"] += 1
        n = state["n"]

        if n == 1:
            state["mu_hat"] = value
            state["sigma_hat"] = 1.0
            z = 0.0
        else:
            prev_mu = state["mu_hat"]
            prev_sigma = state["sigma_hat"]
            diff = value - prev_mu
            state["mu_hat"] = self.alpha * value + (1 - self.alpha) * prev_mu
            var = self.alpha * diff ** 2 + (1 - self.alpha) * prev_sigma ** 2
            state["sigma_hat"] = np.sqrt(max(var, 1e-10))
            z = abs(diff) / max(prev_sigma, 1e-10)

        # Detection
        level = DriftLevel.NONE
        if n >= self.min_samples:
            if z > self.z_severe:
                level = DriftLevel.SEVERE_DRIFT
            elif z > self.z_drift:
                level = DriftLevel.DRIFT
            elif z > self.z_warning:
                level = DriftLevel.WARNING

        # Reset after drift detection (consistent with PH/CUSUM)
        if level in (DriftLevel.DRIFT, DriftLevel.SEVERE_DRIFT):
            state["mu_hat"] = value
            state["sigma_hat"] = 1.0

        signal = DriftSignal(
            timestamp=timestamp,
            client_id=client_id,
            level=level,
            statistic=z,
            threshold=self.z_drift,
            metadata={"mu_hat": state["mu_hat"], "sigma_hat": state["sigma_hat"]},
        )
        self._history.append(signal)
        return signal
