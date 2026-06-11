"""Base class for all drift detectors."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Dict, Any
from enum import Enum


class DriftLevel(Enum):
    """Severity of detected drift."""
    NONE = "none"
    WARNING = "warning"       # Mild shift detected
    DRIFT = "drift"           # Significant distribution change
    SEVERE_DRIFT = "severe"   # Abrupt, large-magnitude change


@dataclass
class DriftSignal:
    """Output signal from a drift detector."""
    timestamp: int
    client_id: int
    level: DriftLevel
    statistic: float          # Detector-specific test statistic
    threshold: float          # Threshold that was exceeded (if any)
    kl_estimate: float = 0.0  # Estimated KL divergence at detection
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_drift(self) -> bool:
        return self.level in (DriftLevel.DRIFT, DriftLevel.SEVERE_DRIFT)

    @property
    def is_warning(self) -> bool:
        return self.level == DriftLevel.WARNING


class BaseDriftDetector:
    """
    Abstract base class for online drift detectors.

    Each detector maintains per-client state and processes a stream
    of scalar distribution summaries (e.g., mean token length, P95)
    to detect changes over time. This is intentionally lighter than
    full online distribution testing; the evaluator separately scores
    alarms against divergence labels computed from consecutive PDFs.
    """

    _MAX_HISTORY = 10_000  # Cap per-instance history to prevent unbounded growth

    def __init__(self, name: str = "BaseDriftDetector"):
        self.name = name
        self._client_states: Dict[int, Dict[str, Any]] = {}
        self._history: Deque[DriftSignal] = deque(maxlen=self._MAX_HISTORY)

    def _get_or_create_state(self, client_id: int) -> Dict[str, Any]:
        """Get or initialize per-client state."""
        if client_id not in self._client_states:
            self._client_states[client_id] = self._init_state()
        return self._client_states[client_id]

    def _init_state(self) -> Dict[str, Any]:
        """Initialize detector state for a new client. Override in subclasses."""
        raise NotImplementedError

    def update(self, client_id: int, timestamp: int, value: float) -> DriftSignal:
        """
        Feed a new observation and return a drift signal.

        Args:
            client_id: Identifier of the client.
            timestamp: Current time window index.
            value: Observed statistic (e.g., mean input tokens).

        Returns:
            DriftSignal indicating drift status.
        """
        raise NotImplementedError

    def reset(self, client_id: int) -> None:
        """Reset detector state for a specific client after drift is handled."""
        if client_id in self._client_states:
            self._client_states[client_id] = self._init_state()

    def reset_all(self) -> None:
        """Reset all client states."""
        self._client_states.clear()
        self._history = deque(maxlen=self._MAX_HISTORY)

    def get_history(self, client_id: Optional[int] = None) -> List[DriftSignal]:
        """Return detection history, optionally filtered by client."""
        if client_id is None:
            return list(self._history)
        return [s for s in self._history if s.client_id == client_id]

    def get_drift_count(self, client_id: Optional[int] = None) -> int:
        """Count the number of drift detections."""
        return sum(1 for s in self.get_history(client_id) if s.is_drift)
