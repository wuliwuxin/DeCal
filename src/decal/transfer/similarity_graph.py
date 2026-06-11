"""
Client Similarity Graph.

Builds and maintains a graph of client similarities based on
Wasserstein distances between their token length distributions.
Supports clustering and archetype identification.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from decal.transfer.wasserstein import WassersteinTransfer


class ClientSimilarityGraph:
    """
    Graph structure for client distribution similarity.

    Nodes = clients, edges weighted by similarity = 1/(1+W).

    Parameters:
        transfer: WassersteinTransfer instance for distance computation.
        similarity_threshold: Minimum similarity to create edge (default 0.1).
    """

    def __init__(
        self,
        transfer: WassersteinTransfer,
        similarity_threshold: float = 0.1,
    ):
        self.transfer = transfer
        self.similarity_threshold = similarity_threshold
        self._adjacency: Dict[int, Dict[int, float]] = {}

    def build(self) -> None:
        """Build the similarity graph from current pairwise distances."""
        client_ids, dist_matrix = self.transfer.get_distance_matrix()
        self._adjacency = {cid: {} for cid in client_ids}

        for i, cid_a in enumerate(client_ids):
            for j, cid_b in enumerate(client_ids):
                if i >= j:
                    continue
                dist = dist_matrix[i, j]
                sim = 1.0 / (1.0 + dist)
                if sim >= self.similarity_threshold:
                    self._adjacency[cid_a][cid_b] = sim
                    self._adjacency[cid_b][cid_a] = sim

    def get_neighbors(self, client_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get top-K most similar clients."""
        if client_id not in self._adjacency:
            return []
        neighbors = list(self._adjacency[client_id].items())
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]

    def get_clusters(self, min_similarity: float = 0.3) -> List[Set[int]]:
        """
        Find clusters of similar clients using connected components.

        Uses depth-first search on edges with similarity >= min_similarity.

        Returns:
            List of sets, each containing client IDs in a cluster.
        """
        visited: Set[int] = set()
        clusters: List[Set[int]] = []

        for cid in self._adjacency:
            if cid in visited:
                continue
            cluster: Set[int] = set()
            stack = [cid]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.add(node)
                for neighbor, sim in self._adjacency.get(node, {}).items():
                    if neighbor not in visited and sim >= min_similarity:
                        stack.append(neighbor)
            if cluster:
                clusters.append(cluster)

        return clusters

    def get_similarity_matrix(self) -> Tuple[List[int], np.ndarray]:
        """Return the full similarity matrix."""
        client_ids = sorted(self._adjacency.keys())
        n = len(client_ids)
        matrix = np.eye(n)

        id_to_idx = {cid: i for i, cid in enumerate(client_ids)}
        for cid_a, neighbors in self._adjacency.items():
            i = id_to_idx[cid_a]
            for cid_b, sim in neighbors.items():
                j = id_to_idx[cid_b]
                matrix[i, j] = sim

        return client_ids, matrix
