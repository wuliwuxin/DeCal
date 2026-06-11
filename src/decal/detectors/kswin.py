"""
KSWIN-style drift detector.

Implements a lightweight version of the Kolmogorov-Smirnov Windowing idea used
in streaming drift libraries such as river. The detector compares a recent
window against an older reference window using the two-sample KS statistic.
"""

from typing import Dict, Any, List
import math

from decal.detectors.base import BaseDriftDetector, DriftSignal, DriftLevel


class KSWINDetector(BaseDriftDetector):
    """Streaming KS-window drift detector."""

    def __init__(
        self,
        alpha: float = 0.01,
        window_size: int = 50,
        stat_size: int = 20,
        min_samples: int = 20,
    ):
        super().__init__(name="KSWIN")
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.min_samples = min_samples

    def _init_state(self) -> Dict[str, Any]:
        return {"window": []}

    @staticmethod
    def _ks_statistic(sample_a: List[float], sample_b: List[float]) -> float:
        a = sorted(sample_a)
        b = sorted(sample_b)
        if not a or not b:
            return 0.0
        values = sorted(set(a + b))
        i = 0
        j = 0
        d = 0.0
        for v in values:
            while i < len(a) and a[i] <= v:
                i += 1
            while j < len(b) and b[j] <= v:
                j += 1
            d = max(d, abs(i / len(a) - j / len(b)))
        return d

    def _critical_value(self, n_ref: int, n_cur: int) -> float:
        effective = math.sqrt((n_ref + n_cur) / (n_ref * n_cur))
        return math.sqrt(-0.5 * math.log(max(self.alpha / 2.0, 1e-12))) * effective

    def update(self, client_id: int, timestamp: int, value: float) -> DriftSignal:
        state = self._get_or_create_state(client_id)
        window = state["window"]
        window.append(value)
        if len(window) > self.window_size:
            window.pop(0)

        level = DriftLevel.NONE
        stat = 0.0
        threshold = 0.0
        if len(window) >= max(self.min_samples, self.stat_size * 2):
            recent = window[-self.stat_size:]
            reference = window[:-self.stat_size][-self.stat_size:]
            stat = self._ks_statistic(reference, recent)
            threshold = self._critical_value(len(reference), len(recent))
            ratio = stat / threshold if threshold > 0 else 0.0
            if ratio > 1.5:
                level = DriftLevel.SEVERE_DRIFT
            elif ratio > 1.0:
                level = DriftLevel.DRIFT
            elif ratio > 0.7:
                level = DriftLevel.WARNING

        signal = DriftSignal(
            timestamp=timestamp,
            client_id=client_id,
            level=level,
            statistic=stat,
            threshold=threshold,
            metadata={"window_size": len(window), "stat_size": self.stat_size},
        )
        self._history.append(signal)
        return signal
