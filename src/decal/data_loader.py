"""
DeCal Data Loader & Preprocessor.

Supports two dataset families:

1. ServeGen (Alibaba Cloud LLM serving traces)
   Data root: data/ServeGen/
   Subdirs: language/{m-small,m-mid,m-large}, reason/deepseek-r1,
            multimodal/mm-image
   Files per client: chunk-{id}-trace.csv + chunk-{id}-dataset.json

2. BurstGPT (real ChatGPT/GPT-4 production traces, ~10.6M requests, 231 days)
   Data root: data/BurstGPT/
   Files: BurstGPT_1.csv, BurstGPT_2.csv, BurstGPT_3.csv
   Reference: https://github.com/HPMLL/BurstGPT

Raw Data Format (ServeGen):
===========================================================================
1. Trace CSV: chunk-{id}-trace.csv
   - Each row: timestamp, rate, cv, pat_name, param1, param2
   - timestamp: seconds since start (0, 600, 1200, ...) in 10-min windows
   - rate: request rate (req/sec) for this client in this window
   - cv: coefficient of variation of inter-arrival times
   - pat_name: "Gamma" or "Weibull" (arrival pattern)
   - param1, param2: distribution parameters (shape, scale)
   - Empty rows (rate=0) mean the client was inactive in that window

2. Dataset JSON: chunk-{id}-dataset.json
   - Structure: { "timestamp": { "field_name": "{bin: prob, ...}" } }
   - Each timestamp maps to a dict of fields (input_tokens, output_tokens, etc.)
   - Each field is a *string* encoding a Python dict: "{163: 0.256, 165: 0.013, ...}"
   - The dict maps token-length bins -> probability mass
   - Probabilities sum to ~1.0 (it's a PDF histogram)
   - Bins are sparse (only non-zero bins are stored)

Raw Data Format (BurstGPT):
===========================================================================
BurstGPT_1/2.csv columns (6):
  - Timestamp:       seconds from 00:00:00 day-1 (continuous across files)
  - Model:           "ChatGPT" | "GPT-4"
  - Request tokens:  input token count  → primary field
  - Response tokens: output token count
  - Total tokens:    request + response
  - Log Type:        "Conversation log" | "API log"

BurstGPT_3.csv columns (8, adds):
  - Session ID:      conversation session UUID
  - Elapsed time:    end-to-end latency (seconds)

DeCal Input Format:
===========================================================================
DeCal algorithms expect:
  - Per client, per timestamp: a dense PDF array (List[float])
    where pdf[i] = P(token_length == i)
  - The Orchestrator's process_window(client_id, timestamp, pdf) takes these

Preprocessing Pipeline:
===========================================================================
  ServeGen: Raw JSON string -> Parse dict -> Sparse-to-Dense -> Normalize -> PDF array
  BurstGPT: BurstGPT_*.csv -> Group by (Model×LogType, 1-hour window)
            -> Histogram of Request tokens -> Normalize -> PDF array
"""

import ast
import os
import json
import csv
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    tqdm = None

logger = logging.getLogger(__name__)


@dataclass
class TraceWindow:
    """A single time window from a trace CSV."""
    timestamp: int
    rate: float
    cv: float
    pat_name: str        # "Gamma" or "Weibull"
    param1: float        # shape
    param2: float        # scale


@dataclass
class ClientTimeSeries:
    """Complete preprocessed data for one client."""
    client_id: int
    model_name: str
    # Trace data: timestamp -> TraceWindow
    trace: Dict[int, TraceWindow]
    # Distribution data: timestamp -> {field_name -> dense PDF array}
    distributions: Dict[int, Dict[str, List[float]]]
    # Sorted active timestamps (where both trace and distribution exist)
    active_timestamps: List[int]

    @property
    def n_windows(self) -> int:
        return len(self.active_timestamps)

    def get_pdf(self, timestamp: int, field: str = "input_tokens") -> Optional[List[float]]:
        """Get the PDF for a specific timestamp and field."""
        if timestamp in self.distributions:
            return self.distributions[timestamp].get(field)
        return None


class ServeGenDataLoader:
    """
    Load and preprocess ServeGen data for DeCal.

    Usage:
        loader = ServeGenDataLoader("data/ServeGen/language/m-large")
        loader.load()

        # Access preprocessed data
        for client in loader.clients.values():
            for ts in client.active_timestamps:
                pdf = client.get_pdf(ts, "input_tokens")
                # Feed to DeCal: orchestrator.process_window(client.client_id, ts, pdf)

        # Or get the simplified format for the Evaluator
        client_data = loader.get_client_distributions(field="input_tokens")
        # client_data: Dict[int, Dict[int, List[float]]]
    """

    def __init__(
        self,
        data_dir: str,
        max_pdf_length: int = 0,       # 0 = auto (use max observed bin)
        normalize: bool = True,         # Normalize PDFs to sum to 1
        min_rate: float = 0.0,          # Skip windows with rate below this
        fields: Optional[List[str]] = None,  # Fields to load (None = all)
    ):
        self.data_dir = data_dir
        self.max_pdf_length = max_pdf_length
        self.normalize = normalize
        self.min_rate = min_rate
        self.fields = fields

        self.clients: Dict[int, ClientTimeSeries] = {}
        self.model_name: str = os.path.basename(data_dir)
        self._all_timestamps: List[int] = []

    def load(self) -> None:
        """Load all data from the data directory."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Discover chunks
        trace_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.endswith("-trace.csv") and not f.startswith(".")
        ])

        logger.info(f"Loading {len(trace_files)} chunks from {self.data_dir}")

        all_ts_set = set()

        trace_iter = (
            tqdm(trace_files, desc=f"ServeGen {self.model_name} chunks", dynamic_ncols=True)
            if tqdm is not None else trace_files
        )
        for trace_file in trace_iter:
            chunk_id = int(trace_file.split("-")[1])
            dataset_file = f"chunk-{chunk_id}-dataset.json"
            dataset_path = os.path.join(self.data_dir, dataset_file)

            if not os.path.exists(dataset_path):
                logger.warning(f"Missing dataset file for chunk {chunk_id}, skipping")
                continue

            # Load trace
            trace = self._load_trace(
                os.path.join(self.data_dir, trace_file)
            )

            # Load distributions
            distributions = self._load_dataset(dataset_path)

            # Find active timestamps (both trace and distribution exist)
            active_ts = sorted(
                set(trace.keys()) & set(distributions.keys())
            )
            # Filter by minimum rate
            if self.min_rate > 0:
                active_ts = [
                    ts for ts in active_ts
                    if trace[ts].rate >= self.min_rate
                ]

            if not active_ts:
                continue

            all_ts_set.update(active_ts)

            self.clients[chunk_id] = ClientTimeSeries(
                client_id=chunk_id,
                model_name=self.model_name,
                trace=trace,
                distributions=distributions,
                active_timestamps=active_ts,
            )

        self._all_timestamps = sorted(all_ts_set)

        logger.info(
            f"Loaded {len(self.clients)} clients, "
            f"{len(self._all_timestamps)} unique timestamps"
        )

    def _load_trace(self, filepath: str) -> Dict[int, TraceWindow]:
        """
        Load trace CSV.

        Format: timestamp, rate, cv, pat_name, param1, param2
        - Rows with rate=0 or empty pat_name are inactive (skipped).
        """
        trace: Dict[int, TraceWindow] = {}

        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue

                ts = int(row[0])
                rate_str = row[1].strip()
                cv_str = row[2].strip()
                pat_name = row[3].strip()
                p1_str = row[4].strip()
                p2_str = row[5].strip()

                # Skip inactive windows
                if not rate_str or not pat_name:
                    continue
                rate = float(rate_str)
                if rate <= 0:
                    continue

                trace[ts] = TraceWindow(
                    timestamp=ts,
                    rate=rate,
                    cv=float(cv_str) if cv_str else 0.0,
                    pat_name=pat_name,
                    param1=float(p1_str) if p1_str else 0.0,
                    param2=float(p2_str) if p2_str else 0.0,
                )

        return trace

    def _load_dataset(
        self, filepath: str
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        Load and preprocess dataset JSON.

        Raw format:
            { "timestamp_str": { "field": "{bin: prob, bin: prob, ...}" } }

        Preprocessing steps:
            1. Parse the string-encoded dict into actual dict
            2. Convert sparse {bin: prob} to dense array pdf[i]
            3. Normalize to sum to 1.0
        """
        with open(filepath, "r") as f:
            raw = json.load(f)

        distributions: Dict[int, Dict[str, List[float]]] = {}

        for ts_str, fields_dict in raw.items():
            ts = int(ts_str)
            field_pdfs: Dict[str, List[float]] = {}

            for field_name, pdf_str in fields_dict.items():
                # Filter fields if specified
                if self.fields and field_name not in self.fields:
                    continue

                # Step 1: Parse string -> dict
                pdf_dict = self._parse_pdf_string(pdf_str)
                if not pdf_dict:
                    continue

                # Step 2: Sparse -> Dense
                pdf_dense = self._sparse_to_dense(pdf_dict)

                # Step 3: Normalize
                if self.normalize:
                    pdf_dense = self._normalize_pdf(pdf_dense)

                field_pdfs[field_name] = pdf_dense

            if field_pdfs:
                distributions[ts] = field_pdfs

        return distributions

    @staticmethod
    def _parse_pdf_string(pdf_str: str) -> Dict[int, float]:
        """
        Parse the string-encoded PDF dict.

        Input:  "{163: 0.256, 165: 0.013, ...}"  (Python dict literal)
        Output: {163: 0.256, 165: 0.013, ...}     (actual dict)

        Uses ast.literal_eval() for safe parsing of Python dict literals.
        """
        if not pdf_str or pdf_str.strip() in ("", "{}"):
            return {}

        try:
            parsed = ast.literal_eval(pdf_str)
            if isinstance(parsed, dict):
                return {int(k): float(v) for k, v in parsed.items()}
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse PDF string: {e!r} (string starts with: {pdf_str[:60]!r})")
        return {}

    @staticmethod
    def _sparse_to_dense(
        sparse_pdf: Dict[int, float],
        max_length: int = 0,
    ) -> List[float]:
        """
        Convert sparse {bin: prob} dict to dense array.

        Input:  {9: 0.007, 163: 0.257, 165: 0.014, ...}
        Output: [0.0, 0.0, ..., 0.007, ..., 0.257, 0.0, 0.014, ...]
                 ^0    ^1       ^9             ^163       ^165

        The array length = max(bin) + 1 (or max_length if specified).
        """
        if not sparse_pdf:
            return []

        max_bin = max(sparse_pdf.keys())
        length = max(max_bin + 1, max_length)
        dense = [0.0] * length

        for bin_idx, prob in sparse_pdf.items():
            if 0 <= bin_idx < length:
                dense[bin_idx] = prob

        return dense

    @staticmethod
    def _normalize_pdf(pdf: List[float]) -> List[float]:
        """Normalize PDF to sum to 1.0."""
        total = sum(pdf)
        if total > 0 and abs(total - 1.0) > 1e-6:
            return [p / total for p in pdf]
        return pdf

    # ---- Convenience methods ----

    def get_client_distributions(
        self, field: str = "input_tokens"
    ) -> Dict[int, Dict[int, List[float]]]:
        """
        Get distributions in the format expected by Evaluator.

        Returns:
            { client_id: { timestamp: pdf_list } }
        """
        result: Dict[int, Dict[int, List[float]]] = {}
        for cid, client in self.clients.items():
            client_data: Dict[int, List[float]] = {}
            for ts in client.active_timestamps:
                pdf = client.get_pdf(ts, field)
                if pdf is not None:
                    client_data[ts] = pdf
            if client_data:
                result[cid] = client_data
        return result

    def get_all_timestamps(self) -> List[int]:
        """Get all unique sorted timestamps across all clients."""
        return list(self._all_timestamps)

    def get_trace_statistics(self, client_id: int) -> Dict[str, Any]:
        """Get summary statistics from trace data for a client."""
        if client_id not in self.clients:
            return {}
        client = self.clients[client_id]
        rates = [client.trace[ts].rate for ts in client.active_timestamps
                 if ts in client.trace]
        return {
            "n_active_windows": len(rates),
            "mean_rate": float(np.mean(rates)) if rates else 0.0,
            "max_rate": float(np.max(rates)) if rates else 0.0,
            "rate_cv": float(np.std(rates) / np.mean(rates)) if rates and np.mean(rates) > 0 else 0.0,
        }

    def compute_drift_labels(
        self,
        field: str = "input_tokens",
        kl_threshold: float = 10.0,
    ) -> Dict[int, Dict[int, bool]]:
        """
        Compute ground-truth drift labels based on KL divergence.

        Returns:
            { client_id: { timestamp: is_drift } }
        """
        from decal.transfer.wasserstein import kl_divergence

        labels: Dict[int, Dict[int, bool]] = {}
        for cid, client in self.clients.items():
            client_labels: Dict[int, bool] = {}
            sorted_ts = client.active_timestamps
            for i in range(len(sorted_ts)):
                ts = sorted_ts[i]
                if i == 0:
                    client_labels[ts] = False
                    continue
                prev_ts = sorted_ts[i - 1]
                prev_pdf = client.get_pdf(prev_ts, field)
                curr_pdf = client.get_pdf(ts, field)
                if prev_pdf is not None and curr_pdf is not None:
                    kl = kl_divergence(curr_pdf, prev_pdf)
                    client_labels[ts] = kl > kl_threshold
                else:
                    client_labels[ts] = False
            labels[cid] = client_labels
        return labels

    def summary(self) -> str:
        """Return a human-readable summary of the loaded data."""
        lines = [
            f"ServeGen Dataset: {self.model_name}",
            f"  Data directory: {self.data_dir}",
            f"  Clients: {len(self.clients)}",
            f"  Timestamps: {len(self._all_timestamps)}",
        ]
        if self._all_timestamps:
            duration_s = self._all_timestamps[-1] - self._all_timestamps[0]
            duration_h = duration_s / 3600
            window_s = (self._all_timestamps[1] - self._all_timestamps[0]
                        if len(self._all_timestamps) > 1 else 600)
            lines.append(f"  Time range: {duration_h:.1f} hours ({duration_s}s)")
            lines.append(f"  Window size: {window_s}s ({window_s//60}min)")

        total_windows = sum(c.n_windows for c in self.clients.values())
        lines.append(f"  Total active windows: {total_windows}")

        # Distribution stats
        all_lengths = []
        for client in self.clients.values():
            for ts in client.active_timestamps[:5]:  # Sample first 5
                pdf = client.get_pdf(ts, "input_tokens")
                if pdf:
                    all_lengths.append(len(pdf))
        if all_lengths:
            lines.append(f"  PDF lengths (sample): "
                         f"mean={np.mean(all_lengths):.0f}, "
                         f"max={max(all_lengths)}")

        return "\n".join(lines)


def preprocess_for_decal(
    data_dir: str,
    field: str = "input_tokens",
    min_rate: float = 0.0,
) -> Tuple[Dict[int, Dict[int, List[float]]], List[int]]:
    """
    One-call convenience function: load + preprocess ServeGen data.

    Args:
        data_dir: Path to model data (e.g., "data/ServeGen/language/m-large")
        field: Distribution field to extract
        min_rate: Minimum request rate to include a window

    Returns:
        (client_distributions, timestamps)
        client_distributions: { client_id: { timestamp: pdf } }
        timestamps: sorted list of all timestamps
    """
    loader = ServeGenDataLoader(
        data_dir=data_dir,
        fields=[field],
        min_rate=min_rate,
    )
    loader.load()
    return loader.get_client_distributions(field), loader.get_all_timestamps()


# ---------------------------------------------------------------------------
# BurstGPT Data Loader
# ---------------------------------------------------------------------------

class BurstGPTDataLoader:
    """
    Load and preprocess the BurstGPT dataset for DeCal.

    BurstGPT is a real-world LLM serving trace dataset collected from
    production ChatGPT and GPT-4 traffic over ~231 days (~10.6M requests).
    Reference: https://github.com/HPMLL/BurstGPT

    Data files (all placed in data_dir):
        BurstGPT_1.csv  - days   0-61   (~1.43M rows, 6 columns)
        BurstGPT_2.csv  - days  61-122  (~3.86M rows, 6 columns)
        BurstGPT_3.csv  - days 225-335  (~5.34M rows, 8 columns, adds Session ID
                                         and Elapsed time)

    CSV schema (all files):
        Timestamp       seconds from 00:00:00 day-1 (continuous across files)
        Model           "ChatGPT" | "GPT-4"
        Request tokens  input token count            → primary field
        Response tokens output token count
        Total tokens    request + response
        Log Type        "Conversation log" | "API log"
    Additional columns in BurstGPT_3.csv only:
        Session ID      conversation session UUID
        Elapsed time    end-to-end latency (seconds)

    Client segmentation — Model × Log Type → 4 deterministic integer IDs:
        0: ChatGPT × Conversation log
        1: ChatGPT × API log
        2: GPT-4   × Conversation log
        3: GPT-4   × API log

    Supported distribution fields:
        "request_tokens"   input token length  (primary, analogous to input_tokens)
        "response_tokens"  output token length
        "total_tokens"     request + response

    Usage:
        loader = BurstGPTDataLoader("data/BurstGPT")
        loader.load()

        client_data = loader.get_client_distributions(field="request_tokens")
        # { client_id: { timestamp: pdf_list } }   ← same interface as ServeGenDataLoader
    """

    # Default time window (1 hour).  BurstGPT has ~10.6M requests over 231 days
    # across 4 client streams, giving ~250 requests/window – adequate for PDF
    # estimation.  Use 600 to match ServeGen exactly if desired.
    WINDOW_SIZE: int = 3600

    # Deterministic client mapping: (Model, Log Type) → int ID
    CLIENT_MAP: Dict[Tuple[str, str], int] = {
        ("ChatGPT", "Conversation log"): 0,
        ("ChatGPT", "API log"):          1,
        ("GPT-4",   "Conversation log"): 2,
        ("GPT-4",   "API log"):          3,
    }
    CLIENT_NAMES: Dict[int, str] = {v: f"{m}+{lt}" for (m, lt), v in CLIENT_MAP.items()}

    _SUPPORTED_FIELDS = {"request_tokens", "response_tokens", "total_tokens"}

    def __init__(
        self,
        data_dir: str,
        window_size: int = WINDOW_SIZE,
        normalize: bool = True,
        min_requests: int = 10,   # minimum requests per window for a valid PDF
        fields: Optional[List[str]] = None,
        max_tokens: int = 0,      # clip token bins at this value (0 = no clip)
        log_type_filter: Optional[List[str]] = None,  # e.g. ["API log"]
    ):
        """
        Args:
            data_dir: Directory containing BurstGPT_*.csv files.
                      Note: the directory name may have a trailing space on some
                      systems — pass the exact path (e.g. "data/BurstGPT ").
            window_size: Aggregation window in seconds (default: 3600).
            normalize: Normalize PDFs to sum to 1.
            min_requests: Discard windows with fewer than this many requests.
            fields: Fields to build distributions for (default: all three).
            max_tokens: If > 0, clip token counts to [1, max_tokens] before
                        binning to keep PDF arrays small.
            log_type_filter: Keep only these Log Types (None = keep all).
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.normalize = normalize
        self.min_requests = min_requests
        self.fields = fields or ["request_tokens", "response_tokens", "total_tokens"]
        self.max_tokens = max_tokens
        self.log_type_filter = set(log_type_filter) if log_type_filter else None

        self.clients: Dict[int, ClientTimeSeries] = {}
        self.model_name: str = "burstgpt"
        self._all_timestamps: List[int] = []
        self._total_requests: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all BurstGPT_*.csv files and build per-client time-series."""
        csv_files = self._discover_files()
        if not csv_files:
            raise FileNotFoundError(
                f"No BurstGPT_*.csv files found in: {self.data_dir}\n"
                f"Download from https://github.com/HPMLL/BurstGPT"
            )

        # client_id -> window_ts -> field -> list[int token counts]
        raw: Dict[int, Dict[int, Dict[str, List[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # client_id -> window_ts -> request count
        counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for path in csv_files:
            self._read_file(path, raw, counts)

        if not raw:
            logger.warning(f"No usable rows found in {self.data_dir}")
            return

        all_ts_set: set = set()

        for client_id in sorted(raw.keys()):
            window_dict = raw[client_id]
            trace: Dict[int, TraceWindow] = {}
            distributions: Dict[int, Dict[str, List[float]]] = {}

            for window_ts in sorted(window_dict.keys()):
                n_req = counts[client_id][window_ts]
                if n_req < self.min_requests:
                    continue

                trace[window_ts] = TraceWindow(
                    timestamp=window_ts,
                    rate=n_req / self.window_size,
                    cv=0.0,
                    pat_name="Gamma",
                    param1=0.0,
                    param2=0.0,
                )

                field_pdfs: Dict[str, List[float]] = {}
                for field_name, token_list in window_dict[window_ts].items():
                    if not token_list:
                        continue
                    pdf = self._build_pdf(token_list)
                    if pdf:
                        field_pdfs[field_name] = pdf

                if field_pdfs:
                    distributions[window_ts] = field_pdfs
                    all_ts_set.add(window_ts)

            active_ts = sorted(set(trace.keys()) & set(distributions.keys()))
            if not active_ts:
                continue

            self.clients[client_id] = ClientTimeSeries(
                client_id=client_id,
                model_name=self.model_name,
                trace=trace,
                distributions=distributions,
                active_timestamps=active_ts,
            )

        self._all_timestamps = sorted(all_ts_set)
        logger.info(
            f"BurstGPT: loaded {len(self.clients)} clients, "
            f"{len(self._all_timestamps)} timestamps, "
            f"{self._total_requests:,} total requests"
        )

    def get_client_distributions(
        self, field: str = "request_tokens"
    ) -> Dict[int, Dict[int, List[float]]]:
        """Return { client_id: { timestamp: pdf } } — same interface as ServeGenDataLoader."""
        result: Dict[int, Dict[int, List[float]]] = {}
        for cid, client in self.clients.items():
            client_data: Dict[int, List[float]] = {}
            for ts in client.active_timestamps:
                pdf = client.get_pdf(ts, field)
                if pdf is not None:
                    client_data[ts] = pdf
            if client_data:
                result[cid] = client_data
        return result

    def get_all_timestamps(self) -> List[int]:
        return list(self._all_timestamps)

    def compute_drift_labels(
        self,
        field: str = "request_tokens",
        kl_threshold: float = 10.0,
    ) -> Dict[int, Dict[int, bool]]:
        """Ground-truth drift labels based on KL divergence between consecutive windows."""
        from decal.transfer.wasserstein import kl_divergence
        labels: Dict[int, Dict[int, bool]] = {}
        for cid, client in self.clients.items():
            client_labels: Dict[int, bool] = {}
            for i, ts in enumerate(client.active_timestamps):
                if i == 0:
                    client_labels[ts] = False
                    continue
                prev_pdf = client.get_pdf(client.active_timestamps[i - 1], field)
                curr_pdf = client.get_pdf(ts, field)
                if prev_pdf is not None and curr_pdf is not None:
                    client_labels[ts] = kl_divergence(curr_pdf, prev_pdf) > kl_threshold
                else:
                    client_labels[ts] = False
            labels[cid] = client_labels
        return labels

    def summary(self) -> str:
        lines = [
            "BurstGPT Dataset (ChatGPT/GPT-4 production traces)",
            f"  Data directory : {self.data_dir}",
            f"  Clients        : {len(self.clients)} "
            f"({', '.join(f'{cid}={self.CLIENT_NAMES[cid]}' for cid in sorted(self.clients))})",
            f"  Timestamps     : {len(self._all_timestamps)}",
            f"  Total requests : {self._total_requests:,}",
        ]
        if self._all_timestamps:
            duration_s = self._all_timestamps[-1] - self._all_timestamps[0]
            lines.append(
                f"  Time range     : {duration_s / 86400:.1f} days ({duration_s:,}s)"
            )
            lines.append(
                f"  Window size    : {self.window_size}s ({self.window_size // 60}min)"
            )
        total_windows = sum(c.n_windows for c in self.clients.values())
        lines.append(f"  Active windows : {total_windows}")

        # Sample PDF statistics
        sample_field = "request_tokens"
        lengths = []
        for client in self.clients.values():
            for ts in client.active_timestamps[:3]:
                pdf = client.get_pdf(ts, sample_field)
                if pdf:
                    lengths.append(len(pdf))
        if lengths:
            lines.append(
                f"  PDF lengths ({sample_field}): "
                f"mean={np.mean(lengths):.0f}, max={max(lengths)}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_files(self) -> List[str]:
        """Return sorted list of BurstGPT_*.csv paths in data_dir."""
        if not os.path.exists(self.data_dir):
            return []
        files = sorted(
            f for f in os.listdir(self.data_dir)
            if f.startswith("BurstGPT_") and f.endswith(".csv") and not f.startswith(".")
        )
        return [os.path.join(self.data_dir, f) for f in files]

    def _read_file(
        self,
        filepath: str,
        raw: Dict[int, Dict[int, Dict[str, List[int]]]],
        counts: Dict[int, Dict[int, int]],
    ) -> None:
        """Parse one BurstGPT CSV file and accumulate into raw / counts."""
        logger.info(f"Reading {os.path.basename(filepath)} ...")
        n_rows = 0

        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse timestamp
                ts_str = row.get("Timestamp", "").strip()
                if not ts_str:
                    continue
                try:
                    ts_raw = float(ts_str)
                except ValueError:
                    continue
                window_ts = int(ts_raw // self.window_size) * self.window_size

                model = row.get("Model", "").strip()
                log_type = row.get("Log Type", "").strip()

                # Log type filter
                if self.log_type_filter and log_type not in self.log_type_filter:
                    continue

                client_id = self.CLIENT_MAP.get((model, log_type))
                if client_id is None:
                    continue

                counts[client_id][window_ts] += 1
                n_rows += 1

                # Collect token values for requested fields
                for field_name in self.fields:
                    val = self._parse_token_field(row, field_name)
                    if val is not None:
                        raw[client_id][window_ts][field_name].append(val)

        self._total_requests += n_rows
        logger.info(f"  → {n_rows:,} rows")

    def _parse_token_field(self, row: Dict[str, str], field_name: str) -> Optional[int]:
        """Parse a token-count column from a CSV row."""
        col_map = {
            "request_tokens":  "Request tokens",
            "response_tokens": "Response tokens",
            "total_tokens":    "Total tokens",
        }
        col = col_map.get(field_name)
        if col is None:
            return None
        v = row.get(col, "").strip()
        if not v:
            return None
        try:
            val = int(float(v))
        except ValueError:
            return None
        if val <= 0:
            return None
        if self.max_tokens > 0:
            val = min(val, self.max_tokens)
        return val

    def _build_pdf(self, token_counts: List[int]) -> List[float]:
        """Build a normalized PDF histogram from integer token counts."""
        if not token_counts:
            return []
        max_bin = max(token_counts)
        if max_bin <= 0:
            return []
        pdf = [0.0] * (max_bin + 1)
        for v in token_counts:
            pdf[v] += 1.0
        if self.normalize:
            total = sum(pdf)
            if total > 0:
                pdf = [p / total for p in pdf]
        return pdf


def preprocess_burstgpt_for_decal(
    data_dir: str,
    field: str = "request_tokens",
    window_size: int = BurstGPTDataLoader.WINDOW_SIZE,
    min_requests: int = 10,
) -> Tuple[Dict[int, Dict[int, List[float]]], List[int]]:
    """
    One-call convenience function: load + preprocess BurstGPT data.

    Args:
        data_dir: Path to BurstGPT data directory (e.g., "data/BurstGPT").
        field: Distribution field to extract (default: "request_tokens").
        window_size: Time window in seconds (default: 3600).
        min_requests: Minimum requests per window to include.

    Returns:
        (client_distributions, timestamps)
        client_distributions: { client_id: { timestamp: pdf } }
        timestamps: sorted list of all timestamps
    """
    loader = BurstGPTDataLoader(
        data_dir=data_dir,
        window_size=window_size,
        fields=[field],
        min_requests=min_requests,
    )
    loader.load()
    return loader.get_client_distributions(field), loader.get_all_timestamps()
