"""System-level interpretation helpers for DeCal budgets.

The core evaluator reports token-length-bin budgets.  These helpers map the
same scalar to approximate KV-cache bytes under a fixed model geometry. The
mapping is intentionally explicit so generated reports can state the assumptions.
"""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class KVModelSpec:
    """Model geometry used to convert token budgets into KV-cache bytes."""

    n_layers: int = 32
    n_kv_heads: int = 32
    head_dim: int = 128
    bytes_per_element: int = 2

    @property
    def bytes_per_token(self) -> int:
        """Bytes needed to store K and V for one token."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.bytes_per_element

    def to_dict(self) -> Dict[str, int]:
        data = asdict(self)
        data["bytes_per_token"] = self.bytes_per_token
        return data


def tokens_to_kv_bytes(tokens: float, spec: KVModelSpec) -> float:
    """Convert a token-length budget into approximate KV-cache bytes."""
    return float(max(tokens, 0.0)) * float(spec.bytes_per_token)


def bytes_to_gib(num_bytes: float) -> float:
    """Convert bytes to GiB."""
    return float(num_bytes) / float(1024 ** 3)


def tokens_to_gib(tokens: float, spec: KVModelSpec) -> float:
    """Convert a token-length budget into approximate KV-cache GiB."""
    return bytes_to_gib(tokens_to_kv_bytes(tokens, spec))
