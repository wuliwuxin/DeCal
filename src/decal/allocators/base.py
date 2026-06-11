"""Base class for KV cache budget allocators."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class AllocationResult:
    """Result of a budget allocation decision."""
    client_id: int
    timestamp: int
    budget: float             # Allocated KV cache budget (in tokens)
    safety_factor: float      # Applied safety factor
    confidence: float         # Confidence level of the allocation
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAllocator:
    """
    Abstract base class for KV cache budget allocators.

    Each allocator maintains per-client state and determines how much
    KV cache to pre-allocate for incoming requests.
    """

    def __init__(self, name: str = "BaseAllocator"):
        self.name = name
        self._client_states: Dict[int, Dict[str, Any]] = {}

    def _get_or_create_state(self, client_id: int) -> Dict[str, Any]:
        if client_id not in self._client_states:
            self._client_states[client_id] = self._init_state()
        return self._client_states[client_id]

    def _init_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def update_distribution(
        self,
        client_id: int,
        timestamp: int,
        pdf: List[float],
    ) -> None:
        """
        Update the allocator with a new observed distribution.

        Args:
            client_id: Client identifier.
            timestamp: Time window index.
            pdf: Probability density function (histogram) of token lengths.
        """
        raise NotImplementedError

    def allocate(
        self,
        client_id: int,
        timestamp: int,
    ) -> AllocationResult:
        """
        Determine the KV cache budget for the next request.

        Args:
            client_id: Client identifier.
            timestamp: Current time window index.

        Returns:
            AllocationResult with the budget decision.
        """
        raise NotImplementedError

    def reset(self, client_id: int) -> None:
        if client_id in self._client_states:
            self._client_states[client_id] = self._init_state()

    def reset_all(self) -> None:
        self._client_states.clear()

    # ---- Utility methods for PDF operations ----

    @staticmethod
    def pdf_percentile(pdf: List[float], q: float) -> float:
        """Compute the q-th percentile from a PDF histogram."""
        if not pdf or sum(pdf) == 0:
            return 0.0
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]  # Normalize
        idx = np.searchsorted(cdf, q)
        return float(min(idx, len(pdf) - 1))

    @staticmethod
    def pdf_mean(pdf: List[float]) -> float:
        """Compute the mean from a PDF histogram."""
        if not pdf or sum(pdf) == 0:
            return 0.0
        return float(sum(i * p for i, p in enumerate(pdf)))

    @staticmethod
    def pdf_to_cdf(pdf: List[float]) -> np.ndarray:
        """Convert PDF to CDF."""
        arr = np.array(pdf, dtype=np.float64)
        total = arr.sum()
        if total == 0:
            return np.zeros_like(arr)
        return np.cumsum(arr / total)
