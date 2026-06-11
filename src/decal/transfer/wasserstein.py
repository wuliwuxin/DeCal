"""
Wasserstein-based Distribution Transfer (Algorithm 3).

Uses the Wasserstein-1 (earth mover's) distance to measure similarity
between client distributions and transfer knowledge from similar
clients for cold-start and post-drift recovery.

Transfer procedure:
    Periodically:
      1. Compute pairwise W-1 distances for all client pairs
      2. Build similarity graph with edge weights = 1/(1+W)
    On cold-start or severe drift for client c:
      3. Find Top-K nearest clients by W-1 distance
      4. Transfer: P̂_c = Σ w_k · P_k / Σ w_k  where w_k = 1/(1+W_k)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def wasserstein_1d(pdf_a: List[float], pdf_b: List[float]) -> float:
    """
    Compute the Wasserstein-1 (earth mover's) distance between two
    1D discrete distributions represented as PDF histograms.

    This is equivalent to the L1 distance between their CDFs.

    Args:
        pdf_a: First PDF histogram.
        pdf_b: Second PDF histogram.

    Returns:
        W-1 distance in bin units.
    """
    max_len = max(len(pdf_a), len(pdf_b))
    if max_len == 0:
        return 0.0

    a = np.zeros(max_len)
    b = np.zeros(max_len)
    a[: len(pdf_a)] = pdf_a
    b[: len(pdf_b)] = pdf_b

    # Normalize
    sum_a = a.sum()
    sum_b = b.sum()
    if sum_a > 0:
        a /= sum_a
    if sum_b > 0:
        b /= sum_b

    # W-1 = integral of |CDF_a - CDF_b|
    cdf_a = np.cumsum(a)
    cdf_b = np.cumsum(b)
    return float(np.sum(np.abs(cdf_a - cdf_b)))


def kl_divergence(pdf_a: List[float], pdf_b: List[float], eps: float = 1e-10) -> float:
    """
    Compute KL(pdf_a || pdf_b) for discrete distributions.

    Args:
        pdf_a: True distribution.
        pdf_b: Approximate distribution.
        eps: Smoothing constant for numerical stability.

    Returns:
        KL divergence in nats.
    """
    max_len = max(len(pdf_a), len(pdf_b))
    if max_len == 0:
        return 0.0

    a = np.zeros(max_len) + eps
    b = np.zeros(max_len) + eps
    a[: len(pdf_a)] += np.array(pdf_a)
    b[: len(pdf_b)] += np.array(pdf_b)

    a /= a.sum()
    b /= b.sum()

    return float(np.sum(a * np.log(a / b)))


def js_divergence(pdf_a: List[float], pdf_b: List[float], eps: float = 1e-10) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric) for discrete distributions.

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5*(P+Q).
    Range: [0, ln(2)] nats. Symmetric and satisfies triangle inequality (sqrt).
    """
    max_len = max(len(pdf_a), len(pdf_b))
    if max_len == 0:
        return 0.0

    a = np.zeros(max_len) + eps
    b = np.zeros(max_len) + eps
    a[: len(pdf_a)] += np.array(pdf_a)
    b[: len(pdf_b)] += np.array(pdf_b)

    a /= a.sum()
    b /= b.sum()
    m = 0.5 * (a + b)

    return float(0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m)))


class WassersteinTransfer:
    """
    Cross-client distribution transfer using Wasserstein distance.

    Parameters:
        top_k: Number of nearest clients to use for transfer (default 3).
        max_distance: Maximum W-1 distance to consider (default inf).
        field_name: Which field to transfer (default 'input_tokens').
    """

    def __init__(
        self,
        top_k: int = 3,
        max_distance: float = float("inf"),
        field_name: str = "input_tokens",
    ):
        self.top_k = top_k
        self.max_distance = max_distance
        self.field_name = field_name

        # Current distribution snapshots: client_id -> pdf
        self._distributions: Dict[int, List[float]] = {}
        # Cached pairwise distances: (i,j) -> distance
        self._pairwise_cache: Dict[Tuple[int, int], float] = {}
        self._cache_valid = False

    def update_distribution(self, client_id: int, pdf: List[float]) -> None:
        """Update the current distribution snapshot for a client."""
        self._distributions[client_id] = pdf
        self._cache_valid = False

    def get_distribution(self, client_id: int) -> Optional[List[float]]:
        """Return the currently committed snapshot for a client, if any."""
        return self._distributions.get(client_id)

    def n_donors(self) -> int:
        """Return the number of clients in the transfer pool."""
        return len(self._distributions)

    def remove_client(self, client_id: int) -> None:
        """Remove a client from the transfer pool."""
        self._distributions.pop(client_id, None)
        self._cache_valid = False

    def compute_pairwise_distances(self) -> Dict[Tuple[int, int], float]:
        """
        Compute pairwise W-1 distances between all clients.
        Results are cached until distributions are updated.

        Returns:
            Dictionary mapping (client_i, client_j) -> W-1 distance.
        """
        if self._cache_valid:
            return self._pairwise_cache

        client_ids = sorted(self._distributions.keys())
        self._pairwise_cache = {}

        for i, cid_a in enumerate(client_ids):
            for j in range(i + 1, len(client_ids)):
                cid_b = client_ids[j]
                dist = wasserstein_1d(
                    self._distributions[cid_a],
                    self._distributions[cid_b],
                )
                self._pairwise_cache[(cid_a, cid_b)] = dist
                self._pairwise_cache[(cid_b, cid_a)] = dist

        self._cache_valid = True
        return self._pairwise_cache

    def get_neighbors(self, client_id: int) -> List[Tuple[int, float]]:
        """
        Get the Top-K nearest clients by W-1 distance.

        Args:
            client_id: Target client.

        Returns:
            List of (neighbor_id, distance) sorted by distance.
        """
        self.compute_pairwise_distances()

        distances = []
        for other_id in self._distributions:
            if other_id == client_id:
                continue
            key = (client_id, other_id)
            if key in self._pairwise_cache:
                d = self._pairwise_cache[key]
                if d <= self.max_distance:
                    distances.append((other_id, d))

        distances.sort(key=lambda x: x[1])
        return distances[: self.top_k]

    def transfer(
        self,
        target_client_id: int,
        exclude_ids: Optional[List[int]] = None,
    ) -> Optional[List[float]]:
        """
        Generate a prior distribution for the target client by
        weighted averaging of its nearest neighbors.

        Args:
            target_client_id: Client needing a prior.
            exclude_ids: Client IDs to exclude (e.g., for leave-one-out eval).

        Returns:
            Transferred PDF, or None if no suitable neighbors found.
        """
        self.compute_pairwise_distances()

        candidates = []
        for other_id in self._distributions:
            if other_id == target_client_id:
                continue
            if exclude_ids and other_id in exclude_ids:
                continue
            key = (target_client_id, other_id)
            if key in self._pairwise_cache:
                d = self._pairwise_cache[key]
                if d <= self.max_distance:
                    candidates.append((other_id, d))

        candidates.sort(key=lambda x: x[1])
        neighbors = candidates[: self.top_k]

        if not neighbors:
            return None

        # Weighted average: w_k = 1 / (1 + W_k)
        return self._weighted_average_from_neighbors(neighbors)

    def transfer_from_reference(
        self,
        reference_pdf: List[float],
        exclude_ids: Optional[List[int]] = None,
    ) -> Optional[List[float]]:
        """Generate a prior using an explicit reference PDF.

        This is intended for offline evaluation such as leave-one-out transfer,
        where the target client's reference distribution is available for
        matching but excluded from the donor pool.
        """
        candidates = []
        for other_id, donor_pdf in self._distributions.items():
            if exclude_ids and other_id in exclude_ids:
                continue
            dist = wasserstein_1d(reference_pdf, donor_pdf)
            if dist <= self.max_distance:
                candidates.append((other_id, dist))

        candidates.sort(key=lambda x: x[1])
        neighbors = candidates[: self.top_k]
        return self._weighted_average_from_neighbors(neighbors)

    def _weighted_average_from_neighbors(
        self,
        neighbors: List[Tuple[int, float]],
    ) -> Optional[List[float]]:
        """Weighted average helper shared by online and offline transfer."""
        max_len = 0
        weighted_pdfs = []
        weights = []
        for neighbor_id, dist in neighbors:
            pdf = self._distributions[neighbor_id]
            max_len = max(max_len, len(pdf))
            weighted_pdfs.append(pdf)
            weights.append(1.0 / (1.0 + dist))

        result = np.zeros(max_len)
        total_weight = sum(weights)
        for pdf, w in zip(weighted_pdfs, weights):
            arr = np.zeros(max_len)
            arr[: len(pdf)] = pdf
            result += (w / total_weight) * arr

        # Normalize
        total = result.sum()
        if total > 0:
            result /= total

        return result.tolist() if neighbors else None

    def get_distance_matrix(self) -> Tuple[List[int], np.ndarray]:
        """
        Return the full pairwise distance matrix.

        Returns:
            Tuple of (client_ids, distance_matrix).
        """
        self.compute_pairwise_distances()
        client_ids = sorted(self._distributions.keys())
        n = len(client_ids)
        matrix = np.zeros((n, n))

        for i, cid_a in enumerate(client_ids):
            for j, cid_b in enumerate(client_ids):
                if i == j:
                    continue
                key = (cid_a, cid_b)
                if key in self._pairwise_cache:
                    matrix[i, j] = self._pairwise_cache[key]

        return client_ids, matrix
