"""
Evaluator: Trace-driven simulation for DeCal allocation.

Replays ServeGen or BurstGPT trace data through DeCal and computes
comprehensive metrics.
"""

import ast
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
import os
import json
import zlib

from decal.detectors.base import BaseDriftDetector, DriftLevel
from decal.allocators.base import BaseAllocator
from decal.allocators.drift_adaptive import DriftAdaptiveAllocator
from decal.evaluation.metrics import Metrics, WindowResult
from decal.transfer.wasserstein import kl_divergence, js_divergence
from decal.orchestrator import DeCalOrchestrator, OrchestratorConfig

logger = logging.getLogger(__name__)


def _compute_per_request_oom(pdf: List[float], budget: float) -> float:
    """Compute fraction of requests exceeding budget: 1 - CDF(budget).

    Given a discrete PDF (pdf[i] = probability of token length i),
    returns the probability mass above the budget threshold.
    """
    total = sum(pdf)
    if total <= 0 or budget < 0:
        return 0.0
    cutoff = int(budget)
    if cutoff >= len(pdf) - 1:
        return 0.0
    tail_mass = sum(pdf[cutoff + 1:])
    return tail_mass / total


class Evaluator:
    """
    Trace-driven evaluator for KV cache allocation.

    Loads trace data (ServeGen or BurstGPT) and replays it through DeCal,
    computing waste, OOM, MAPE, and drift detection metrics.

    Usage (ServeGen):
        evaluator = Evaluator(data_dir="data/ServeGen/language/m-large")
        evaluator.load_data()

    Usage (BurstGPT):
        loader = BurstGPTDataLoader("data/BurstGPT")
        loader.load()
        evaluator = Evaluator(field_name="request_tokens")
        evaluator.load_from_loader(loader)

        metrics = evaluator.evaluate_orchestrator(decal_orch)
    """

    def __init__(
        self,
        data_dir: str = "",
        kl_drift_threshold: float = None,
        field_name: str = "input_tokens",
        drift_divergence: str = "js",
        eval_seed: int = 0,
        metric_sample_prob: float = 1.0,
        shuffle_orchestrator_clients: bool = False,
    ):
        """
        Args:
            data_dir: Path to data directory.
                ServeGen:  e.g. "data/ServeGen/language/m-large".
                BurstGPT:  pass "" and call load_from_loader() instead.
            kl_drift_threshold: Divergence threshold for ground-truth drift labeling.
                Default: 0.1 for JS divergence, 10.0 for KL divergence.
            field_name: Distribution field to evaluate on.
            drift_divergence: "js" (symmetric, default) or "kl" (asymmetric, legacy).
        """
        self.data_dir = data_dir
        if kl_drift_threshold is None:
            self.kl_drift_threshold = 0.1 if drift_divergence == "js" else 10.0
        else:
            self.kl_drift_threshold = kl_drift_threshold
        self.field_name = field_name
        self.drift_divergence = drift_divergence
        self.eval_seed = int(eval_seed)
        self.metric_sample_prob = float(metric_sample_prob)
        self.shuffle_orchestrator_clients = bool(shuffle_orchestrator_clients)

        # Loaded data: client_id -> {timestamp -> pdf}
        self._client_data: Dict[int, Dict[int, List[float]]] = {}
        # Relative request volume per window: client_id -> {timestamp -> weight}
        self._window_weights: Dict[int, Dict[int, float]] = {}
        # Ground truth drifts: client_id -> set of drift timestamps
        self._drift_ground_truth: Dict[int, set] = {}
        self._timestamps: List[int] = []

    def include_window_in_metrics(self, client_id: int, timestamp: int) -> bool:
        """Return whether a window should contribute to aggregate metrics.

        The default includes every window.  When ``metric_sample_prob < 1``,
        inclusion is deterministic for a given seed/client/timestamp so all
        strategies see the same metric mask.
        """
        if self.metric_sample_prob >= 1.0:
            return True
        if self.metric_sample_prob <= 0.0:
            return False
        key = f"{self.eval_seed}:{client_id}:{timestamp}".encode("utf-8")
        h = zlib.adler32(key) & 0xFFFFFFFF
        return (h / 0xFFFFFFFF) < self.metric_sample_prob

    def load_from_loader(self, loader: Any, field: Optional[str] = None) -> None:
        """
        Load data from a pre-built data loader (ServeGenDataLoader or BurstGPTDataLoader).

        This is the preferred way to feed BurstGPT data into the Evaluator, since
        the Evaluator's built-in load_data() only handles the ServeGen on-disk format.

        Args:
            loader: A ServeGenDataLoader or BurstGPTDataLoader instance (must have
                    called .load() already).
            field: Distribution field to use.  Defaults to self.field_name.
        """
        use_field = field or self.field_name
        self._client_data = loader.get_client_distributions(field=use_field)
        self._window_weights = {}
        if hasattr(loader, "clients"):
            for cid, client in loader.clients.items():
                weights = {}
                for ts in client.active_timestamps:
                    if ts in client.trace:
                        weights[ts] = float(max(client.trace[ts].rate, 0.0))
                if weights:
                    self._window_weights[cid] = weights

        all_ts: set = set()
        for cd in self._client_data.values():
            all_ts.update(cd.keys())
        self._timestamps = sorted(all_ts)

        self._compute_drift_ground_truth()

        logger.info(
            f"Loaded {len(self._client_data)} clients, "
            f"{len(self._timestamps)} timestamps from loader ({type(loader).__name__})"
        )

    def load_data(self) -> None:
        """Load ServeGen trace and dataset files from self.data_dir."""
        if not self.data_dir:
            raise ValueError(
                "data_dir is empty. Use load_from_loader() to load BurstGPT data."
            )
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        dataset_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.endswith("-dataset.json") and not f.startswith("._")
        ])
        self._window_weights = {}

        for dataset_file in dataset_files:
            client_id = int(dataset_file.split("-")[1])
            filepath = os.path.join(self.data_dir, dataset_file)
            trace_path = os.path.join(
                self.data_dir, f"chunk-{client_id}-trace.csv"
            )

            with open(filepath, "r") as f:
                raw = json.load(f)

            client_data: Dict[int, List[float]] = {}
            for ts_str, fields in raw.items():
                ts = int(ts_str)
                if self.field_name not in fields:
                    continue
                pdf_str = fields[self.field_name]
                try:
                    pdf_dict = ast.literal_eval(pdf_str)
                except (ValueError, SyntaxError):
                    continue
                if not pdf_dict:
                    continue

                if isinstance(next(iter(pdf_dict.keys())), int):
                    max_bin = max(pdf_dict.keys())
                    pdf = [pdf_dict.get(b, 0.0) for b in range(max_bin + 1)]
                else:
                    max_bin = max(int(k) for k in pdf_dict.keys())
                    pdf = [pdf_dict.get(str(b), pdf_dict.get(b, 0.0))
                           for b in range(max_bin + 1)]

                client_data[ts] = pdf

            if client_data:
                self._client_data[client_id] = client_data
                self._window_weights[client_id] = self._load_trace_weights(trace_path)

        # Compute timestamps
        all_ts = set()
        for cd in self._client_data.values():
            all_ts.update(cd.keys())
        self._timestamps = sorted(all_ts)

        # Compute ground truth drifts via KL divergence
        self._compute_drift_ground_truth()

        logger.info(
            f"Loaded {len(self._client_data)} clients, "
            f"{len(self._timestamps)} timestamps from {self.data_dir}"
        )

    def _compute_drift_ground_truth(self) -> None:
        """Mark windows where divergence exceeds threshold as drift.

        Uses JS divergence (symmetric) by default. KL (asymmetric) available
        for backward compatibility. JS is preferred because it avoids the
        directional bias of KL(P||Q) != KL(Q||P).

        Note that this is an evaluation-time label on full distributions.
        The online detector itself may consume only scalar workload summaries
        such as mean or P95 token length.
        """
        self._drift_ground_truth = {}
        div_fn = js_divergence if self.drift_divergence == "js" else kl_divergence

        for cid, data in self._client_data.items():
            drift_timestamps = set()
            sorted_ts = sorted(data.keys())

            for i in range(1, len(sorted_ts)):
                prev_pdf = data[sorted_ts[i - 1]]
                curr_pdf = data[sorted_ts[i]]
                div = div_fn(curr_pdf, prev_pdf)
                if div > self.kl_drift_threshold:
                    drift_timestamps.add(sorted_ts[i])

            self._drift_ground_truth[cid] = drift_timestamps

    @staticmethod
    def _load_trace_weights(filepath: str) -> Dict[int, float]:
        """Load per-window request rates from a ServeGen trace CSV."""
        weights: Dict[int, float] = {}
        if not os.path.exists(filepath):
            return weights
        with open(filepath, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    continue
                try:
                    ts = int(parts[0])
                    rate = float(parts[1])
                except ValueError:
                    continue
                if rate > 0:
                    weights[ts] = rate
        return weights

    @staticmethod
    def _detector_value(
        pdf: List[float],
        prev_pdf: Optional[List[float]],
        mode: str,
    ) -> Optional[float]:
        if mode == "mean":
            return float(sum(i * p for i, p in enumerate(pdf)))
        if prev_pdf is None:
            return None
        if mode == "js_divergence":
            return js_divergence(pdf, prev_pdf)
        if mode == "kl_divergence":
            return kl_divergence(pdf, prev_pdf)
        raise ValueError(f"Unknown detector_input mode: {mode}")

    def _window_weight(self, client_id: int, timestamp: int) -> float:
        return float(self._window_weights.get(client_id, {}).get(timestamp, 1.0))

    def _build_window_result(
        self,
        cid: int,
        ts: int,
        pdf: List[float],
        budget: float,
        actual_p95: float,
        actual_p99: float,
        actual_p999: float,
        actual_mean: float,
        drift_detected: bool,
    ) -> Optional[WindowResult]:
        """Compute derived per-window metrics and assemble a WindowResult.

        Returns ``None`` when this window is excluded by the metric sampler;
        the caller treats that as "skip this window".  Centralising the
        construction keeps allocator-only and orchestrator paths perfectly
        consistent on what they record (P95/P99/P99.9 actuals, both APEs,
        token-level over/under allocation, request-level OOM mass).
        """
        if not self.include_window_in_metrics(cid, ts):
            return None

        oom = actual_p99 > budget
        if budget > 0:
            waste = max(0.0, budget - actual_p99) / budget
        else:
            waste = 0.0
        ape_p95 = abs(budget - actual_p95) / actual_p95 if actual_p95 > 0 else 0.0
        ape_p99 = abs(budget - actual_p99) / actual_p99 if actual_p99 > 0 else 0.0
        over_alloc = max(0.0, budget - actual_p99)
        under_alloc = max(0.0, actual_p99 - budget)
        per_req_oom = _compute_per_request_oom(pdf, budget)
        drift_gt = ts in self._drift_ground_truth.get(cid, set())

        return WindowResult(
            client_id=cid,
            timestamp=ts,
            budget=budget,
            actual_p99=actual_p99,
            actual_p95=actual_p95,
            actual_p999=actual_p999,
            actual_mean=actual_mean,
            waste=waste,
            oom=oom,
            ape=ape_p95,
            ape_p99=ape_p99,
            over_allocation=over_alloc,
            under_allocation=under_alloc,
            per_request_oom=per_req_oom,
            request_weight=self._window_weight(cid, ts),
            drift_detected=drift_detected,
            drift_ground_truth=drift_gt,
        )

    def temporal_split(self, train_ratio: float = 0.7) -> int:
        """Return the timestamp that splits data into train/test by ratio.

        The first train_ratio fraction of timestamps is the calibration period
        (for hyperparameter tuning); the remainder is the evaluation period.
        Returns the split timestamp.
        """
        if not self._timestamps:
            return 0
        idx = int(len(self._timestamps) * train_ratio)
        idx = min(max(idx, 1), len(self._timestamps) - 1)
        return self._timestamps[idx]

    @property
    def client_ids(self) -> List[int]:
        return sorted(self._client_data.keys())

    @property
    def n_clients(self) -> int:
        return len(self._client_data)

    @property
    def n_timestamps(self) -> int:
        return len(self._timestamps)

    def evaluate_allocator(
        self,
        allocator: BaseAllocator,
        detector: Optional[BaseDriftDetector] = None,
        max_windows: Optional[int] = None,
        eval_after_ts: Optional[int] = None,
        detector_input: str = "mean",
    ) -> Metrics:
        """
        Evaluate a single allocator on the loaded trace data.

        Args:
            allocator: Budget allocator to evaluate.
            detector: Optional drift detector (for DriftAdaptive allocators).
            max_windows: Limit evaluation to first N windows per client.
            eval_after_ts: Only record metrics for windows after this timestamp.
                The allocator still processes all windows (warm-up), but only
                windows with ts > eval_after_ts contribute to reported metrics.
                Use with temporal_split() for honest train-test evaluation.

        Returns:
            Metrics object with all results.
        """
        metrics = Metrics()

        for cid in self.client_ids:
            data = self._client_data[cid]
            sorted_ts = sorted(data.keys())
            prev_pdf: Optional[List[float]] = None
            prev_detector_stat: Optional[float] = None
            if max_windows:
                sorted_ts = sorted_ts[:max_windows]

            for ts in sorted_ts:
                pdf = data[ts]

                # Compute ground truth statistics
                actual_p99 = BaseAllocator.pdf_percentile(pdf, 0.99)
                actual_p95 = BaseAllocator.pdf_percentile(pdf, 0.95)
                actual_p999 = BaseAllocator.pdf_percentile(pdf, 0.999)
                actual_mean = BaseAllocator.pdf_mean(pdf)

                # Run drift detection if provided
                drift_detected = False
                if detector is not None and prev_detector_stat is not None:
                    signal = detector.update(cid, ts, prev_detector_stat)
                    drift_detected = signal.is_drift

                    # Notify allocator before allocation, using committed history only.
                    if isinstance(allocator, DriftAdaptiveAllocator):
                        allocator.notify_drift(cid, signal)

                # Get budget from allocator
                result = allocator.allocate(cid, ts)
                budget = result.budget

                # Update allocator with this window's distribution
                allocator.update_distribution(cid, ts, pdf)
                prev_detector_stat = self._detector_value(pdf, prev_pdf, detector_input)
                prev_pdf = pdf

                # Skip metric recording during warm-up (distribution already updated above)
                if eval_after_ts is not None and ts <= eval_after_ts:
                    continue

                window_result = self._build_window_result(
                    cid=cid,
                    ts=ts,
                    pdf=pdf,
                    budget=budget,
                    actual_p95=actual_p95,
                    actual_p99=actual_p99,
                    actual_p999=actual_p999,
                    actual_mean=actual_mean,
                    drift_detected=drift_detected,
                )
                if window_result is None:
                    continue
                metrics.add_result(window_result)

        return metrics

    def evaluate_orchestrator(
        self,
        orchestrator: DeCalOrchestrator,
        max_windows: Optional[int] = None,
        eval_after_ts: Optional[int] = None,
    ) -> Metrics:
        """
        Evaluate the full DeCal orchestrator.

        Args:
            orchestrator: DeCalOrchestrator instance.
            max_windows: Limit evaluation per client.
            eval_after_ts: Only record metrics for windows after this timestamp.
                The orchestrator still processes all windows to warm up its state,
                but only windows with ts > eval_after_ts contribute to reported
                metrics. This enables honest temporal-split evaluation.

        Returns:
            Metrics object.
        """
        metrics = Metrics()
        # Track per-client window count for max_windows enforcement
        client_window_count: Dict[int, int] = {}

        # Process windows in temporal order (all clients per timestamp)
        for ts in self._timestamps:
            for cid in self.client_ids:
                data = self._client_data[cid]
                if ts not in data:
                    continue

                # Enforce per-client max_windows limit
                if max_windows is not None:
                    count = client_window_count.get(cid, 0)
                    if count >= max_windows:
                        continue
                    client_window_count[cid] = count + 1

                pdf = data[ts]
                actual_p99 = BaseAllocator.pdf_percentile(pdf, 0.99)
                actual_p95 = BaseAllocator.pdf_percentile(pdf, 0.95)
                actual_p999 = BaseAllocator.pdf_percentile(pdf, 0.999)
                actual_mean = BaseAllocator.pdf_mean(pdf)

                # Process through orchestrator
                result = orchestrator.process_window(cid, ts, pdf, commit_transfer_update=False)
                budget = result.budget

                # Check if drift was detected by orchestrator's detector
                hist = orchestrator.detector.get_history(cid)
                drift_detected = (
                    len(hist) > 0
                    and hist[-1].timestamp == ts
                    and hist[-1].is_drift
                )

                if eval_after_ts is not None and ts <= eval_after_ts:
                    continue

                window_result = self._build_window_result(
                    cid=cid,
                    ts=ts,
                    pdf=pdf,
                    budget=budget,
                    actual_p95=actual_p95,
                    actual_p99=actual_p99,
                    actual_p999=actual_p999,
                    actual_mean=actual_mean,
                    drift_detected=drift_detected,
                )
                if window_result is None:
                    continue
                metrics.add_result(window_result)

            orchestrator.commit_transfer_updates()

        return metrics

    def compare_strategies(
        self,
        strategies: Dict[str, Any],
        max_windows: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple allocation strategies.

        Args:
            strategies: Dict mapping name -> allocator or orchestrator.
            max_windows: Limit per client.

        Returns:
            Dict mapping strategy name -> summary metrics dict.
        """
        results = {}
        for name, strategy in strategies.items():
            logger.info(f"Evaluating strategy: {name}")
            if isinstance(strategy, DeCalOrchestrator):
                metrics = self.evaluate_orchestrator(strategy, max_windows)
            elif isinstance(strategy, tuple):
                # (allocator, detector)
                alloc, det = strategy
                metrics = self.evaluate_allocator(alloc, det, max_windows)
            else:
                metrics = self.evaluate_allocator(strategy, max_windows=max_windows)
            results[name] = metrics.summary()
            logger.info(f"  {name}: {metrics.summary()}")

        return results

    def cold_start_evaluation(
        self,
        orchestrator: DeCalOrchestrator,
        n_cold_windows: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate cold-start performance: MAPE in first n_cold_windows.

        Returns:
            Dict with cold_start_mape and warm_mape.
        """
        cold_apes = []
        warm_apes = []

        orchestrator.reset()

        for cid in self.client_ids:
            data = self._client_data[cid]
            sorted_ts = sorted(data.keys())

            for i, ts in enumerate(sorted_ts):
                pdf = data[ts]
                actual_p95 = BaseAllocator.pdf_percentile(pdf, 0.95)
                result = orchestrator.process_window(cid, ts, pdf)

                if actual_p95 > 0:
                    ape = abs(result.budget - actual_p95) / actual_p95
                    if i < n_cold_windows:
                        cold_apes.append(ape)
                    else:
                        warm_apes.append(ape)

        return {
            "cold_start_mape": float(np.mean(cold_apes)) if cold_apes else 0.0,
            "warm_mape": float(np.mean(warm_apes)) if warm_apes else 0.0,
            "cold_n": len(cold_apes),
            "warm_n": len(warm_apes),
        }

    def leave_one_out_transfer_eval(
        self,
        top_k: int = 3,
        n_eval_windows: int = 5,
    ) -> Dict[str, float]:
        """
        Leave-one-out evaluation of cross-client transfer.

        For each client, exclude it, build transfer from remaining clients'
        *median-timestamp* distributions, then measure prediction MAPE on the
        target's subsequent windows.  The no-transfer baseline uses the global
        average distribution across all other clients.

        Returns:
            Dict with transfer_mape, no_transfer_mape, improvement.
        """
        from decal.transfer.wasserstein import WassersteinTransfer

        transfer_apes = []
        no_transfer_apes = []

        # Use the median timestamp as the reference "pool" snapshot
        mid_idx = len(self._timestamps) // 2
        ref_ts = self._timestamps[mid_idx] if self._timestamps else None
        if ref_ts is None:
            return {"transfer_mape": 0.0, "no_transfer_mape": 0.0, "improvement": 0.0}

        # Build pool: for each client, pick the closest available timestamp
        pool: Dict[int, List[float]] = {}
        for cid in self.client_ids:
            data = self._client_data[cid]
            # Find closest timestamp
            available = sorted(data.keys())
            if not available:
                continue
            best_ts = min(available, key=lambda t: abs(t - ref_ts))
            pool[cid] = data[best_ts]

        if len(pool) < 3:
            return {"transfer_mape": 0.0, "no_transfer_mape": 0.0, "improvement": 0.0}

        for target_cid in self.client_ids:
            if target_cid not in pool:
                continue

            # Build transfer without target
            wt = WassersteinTransfer(top_k=top_k)
            for cid, pdf in pool.items():
                if cid != target_cid:
                    wt.update_distribution(cid, pdf)

            # Wasserstein-based transfer prediction using the target's
            # reference-timestamp snapshot for leave-one-out matching.
            transferred_pdf = wt.transfer_from_reference(pool[target_cid])

            # Global average baseline (no-transfer)
            max_len = max(len(p) for c, p in pool.items() if c != target_cid)
            global_avg = np.zeros(max_len)
            n_donors = 0
            for cid, pdf in pool.items():
                if cid != target_cid:
                    arr = np.zeros(max_len)
                    arr[: len(pdf)] = pdf
                    global_avg += arr
                    n_donors += 1
            if n_donors > 0:
                global_avg /= n_donors
                total = global_avg.sum()
                if total > 0:
                    global_avg /= total
            global_avg_pdf = global_avg.tolist()

            # Evaluate on windows *after* the reference timestamp
            data = self._client_data[target_cid]
            sorted_ts = sorted(data.keys())
            future_ts = [t for t in sorted_ts if t > ref_ts][:n_eval_windows]

            for ts in future_ts:
                actual_pdf = data[ts]
                actual_p95 = BaseAllocator.pdf_percentile(actual_pdf, 0.95)
                if actual_p95 <= 0:
                    continue

                # No transfer: global average P95
                global_p95 = BaseAllocator.pdf_percentile(global_avg_pdf, 0.95)
                nt_ape = abs(global_p95 - actual_p95) / actual_p95
                no_transfer_apes.append(nt_ape)

                # Transfer: Wasserstein-weighted P95
                if transferred_pdf is not None:
                    pred_p95 = BaseAllocator.pdf_percentile(transferred_pdf, 0.95)
                    t_ape = abs(pred_p95 - actual_p95) / actual_p95
                    transfer_apes.append(t_ape)
                else:
                    transfer_apes.append(nt_ape)

        t_mape = float(np.mean(transfer_apes)) if transfer_apes else 0.0
        nt_mape = float(np.mean(no_transfer_apes)) if no_transfer_apes else 0.0

        return {
            "transfer_mape": t_mape,
            "no_transfer_mape": nt_mape,
            "improvement": (nt_mape - t_mape) / nt_mape if nt_mape > 0 else 0.0,
            "n_samples": len(transfer_apes),
        }
