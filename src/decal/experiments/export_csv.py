#!/usr/bin/env python3
"""
DeCal Results CSV Exporter.

Converts DeCal experiment JSON results to a unified, well-structured
set of CSV files suitable for statistical analysis, plotting, and LaTeX
table generation.

Output files (under ``output_dir``):
  allocation_summary.csv    — Per-strategy aggregate metrics across all models
  allocation_detail.csv     — Per-model per-strategy metrics
  all_strategies.csv     — Unified wide-format table for DeCal outputs

Usage:
    python -m decal.experiments.export_csv \\
        --results_dir results \\
        --output_dir results/csv

    # Or import and call directly:
    from decal.experiments.export_csv import export_all
    export_all("results", "results/csv")
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

from decal.experiments.datasets import infer_servegen_category


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRATEGY_ORDER = ["DeCal"]

# Columns kept in the summary CSVs (in this order)
ALLOC_COLS = [
    "n_windows",
    "waste_rate",
    "oom_rate",
    "per_request_oom",
    "p95_coverage",
    "p99_coverage",
    "p999_coverage",
    "mape",
    "mape_p95",
    "mape_p99",
    "mean_budget",
    "mean_actual_p95",
    "mean_actual_p99",
    "mean_actual_p999",
    "budget_efficiency",
    "mean_waste_tokens",
    "over_allocation_mean",
    "over_allocation_p95",
    "under_allocation_mean",
    "under_allocation_p95",
    "worst_window_oom",
    "consecutive_oom_max",
    # Paper metrics (capacity-pressure indicators)
    "throughput",
    "p95_latency",
    "p99_latency",
    "quality_proxy",
    "slo_goodput",
    # Drift metrics
    "drift_precision",
    "drift_recall",
    "drift_f1",
    "drift_fpr",
    "n_drifts_detected",
    "n_drifts_ground_truth",
    "detection_latency_mean",
    "recovery_time_mean",
]

DRIFT_COLS = ["precision", "recall", "f1", "fpr", "n_detections", "n_ground_truth"]

SENSITIVITY_COLS = [
    "param_name", "param_value",
    "waste_rate", "oom_rate", "p99_coverage", "p999_coverage",
    "mape_p95", "mape_p99",
    "under_allocation_mean", "drift_f1",
]

COLD_START_COLS = [
    "method", "transfer_mape", "no_transfer_mape", "improvement", "n_samples",
]

def _safe(d: Dict, key: str, default: Any = "") -> Any:
    """Safely retrieve a value from a dict."""
    return d.get(key, default)


def _fmt(v: Any, digits: int = 4) -> str:
    """Format a numeric value for CSV output."""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _category(model_name: str) -> str:
    return infer_servegen_category(model_name)


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    """Write rows to a CSV file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) if isinstance(v, float) else v
                             for k, v in row.items()})
    print(f"  Written: {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Allocation strategy comparison
# ---------------------------------------------------------------------------

def _section(model_data: Dict, primary: str) -> Dict:
    data = model_data.get(primary)
    return data if isinstance(data, dict) else {}


def export_allocation(
    all_results: Dict, output_dir: str
) -> None:
    """Export allocation comparison to CSV."""
    # Per-model per-strategy detail
    detail_rows = []
    # Aggregated across models (macro-average)
    agg: Dict[str, Dict[str, List[float]]] = {}

    for model_name, model_data in all_results.items():
        allocation = _section(model_data, "allocation")
        for strategy_name, metrics in allocation.items():
            if not isinstance(metrics, dict):
                continue
            row = {
                "category": _category(model_name),
                "model": model_name,
                "strategy": strategy_name,
            }
            for col in ALLOC_COLS:
                row[col] = metrics.get(col, "")
            detail_rows.append(row)

            # Accumulate for aggregate
            if strategy_name not in agg:
                agg[strategy_name] = {c: [] for c in ALLOC_COLS}
            for col in ALLOC_COLS:
                v = metrics.get(col)
                if isinstance(v, (int, float)):
                    agg[strategy_name][col].append(float(v))

    _write_csv(
        os.path.join(output_dir, "allocation_detail.csv"),
        ["category", "model", "strategy"] + ALLOC_COLS,
        detail_rows,
    )

    # Summary: macro-average across models
    import statistics
    summary_rows = []
    for strategy_name in STRATEGY_ORDER + [
        s for s in agg if s not in STRATEGY_ORDER
    ]:
        if strategy_name not in agg:
            continue
        row = {"strategy": strategy_name}
        for col in ALLOC_COLS:
            vals = agg[strategy_name][col]
            row[col + "_mean"] = statistics.mean(vals) if vals else ""
            row[col + "_std"] = statistics.stdev(vals) if len(vals) > 1 else ""
        summary_rows.append(row)

    summary_cols = ["strategy"]
    for col in ALLOC_COLS:
        summary_cols += [col + "_mean", col + "_std"]
    _write_csv(
        os.path.join(output_dir, "allocation_summary.csv"),
        summary_cols,
        summary_rows,
    )

    # Category-level macro-average, so ServeGen language/reason/multimodal
    # coverage can be checked without manually grouping models.
    cat_agg: Dict[tuple, Dict[str, List[float]]] = {}
    for row in detail_rows:
        key = (row["category"], row["strategy"])
        cat_agg.setdefault(key, {c: [] for c in ALLOC_COLS})
        for col in ALLOC_COLS:
            v = row.get(col)
            if isinstance(v, (int, float)):
                cat_agg[key][col].append(float(v))

    category_rows = []
    for (category, strategy_name), vals_by_col in sorted(cat_agg.items()):
        row = {"category": category, "strategy": strategy_name}
        for col in ALLOC_COLS:
            vals = vals_by_col[col]
            row[col + "_mean"] = statistics.mean(vals) if vals else ""
            row[col + "_std"] = statistics.stdev(vals) if len(vals) > 1 else ""
        category_rows.append(row)
    _write_csv(
        os.path.join(output_dir, "allocation_category_summary.csv"),
        ["category", "strategy"] + [c + suffix for c in ALLOC_COLS for suffix in ("_mean", "_std")],
        category_rows,
    )

    # Wide-format: models as rows, strategies as columns
    _export_wide_metric(all_results, "oom_rate", "allocation", output_dir, "allocation_oom_wide.csv")
    _export_wide_metric(all_results, "waste_rate", "allocation", output_dir, "allocation_waste_wide.csv")
    _export_wide_metric(all_results, "p95_coverage", "allocation", output_dir, "allocation_p95_coverage_wide.csv")
    _export_wide_metric(all_results, "p99_coverage", "allocation", output_dir, "allocation_p99_coverage_wide.csv")
    _export_wide_metric(all_results, "p999_coverage", "allocation", output_dir, "allocation_p999_coverage_wide.csv")
    _export_wide_metric(all_results, "mape", "allocation", output_dir, "allocation_mape_wide.csv")
    _export_wide_metric(all_results, "mape_p99", "allocation", output_dir, "allocation_mape_p99_wide.csv")
    _export_wide_metric(all_results, "under_allocation_mean", "allocation", output_dir, "allocation_under_alloc_wide.csv")
    _export_wide_metric(all_results, "over_allocation_mean", "allocation", output_dir, "allocation_over_alloc_wide.csv")


def _export_wide_metric(
    all_results: Dict,
    metric: str,
    section_key: str,
    output_dir: str,
    filename: str,
) -> None:
    """Export a single metric in wide format: rows=models, cols=strategies."""
    # Gather all strategy names
    strategies_seen = set()
    for model_data in all_results.values():
        section = _section(model_data, section_key)
        strategies_seen.update(section.keys())

    ordered = [s for s in STRATEGY_ORDER if s in strategies_seen]
    ordered += sorted(s for s in strategies_seen if s not in ordered)

    rows = []
    for model_name, model_data in all_results.items():
        section = _section(model_data, section_key)
        row = {"category": _category(model_name), "model": model_name}
        for s in ordered:
            v = section.get(s, {})
            row[s] = v.get(metric, "") if isinstance(v, dict) else ""
        rows.append(row)

    _write_csv(
        os.path.join(output_dir, filename),
        ["category", "model"] + ordered,
        rows,
    )


# ---------------------------------------------------------------------------
# Drift detection accuracy
# ---------------------------------------------------------------------------

def export_drift_detection(all_results: Dict, output_dir: str) -> None:
    rows = []
    for model_name, model_data in all_results.items():
        drift_detection = _section(model_data, "drift_detection")
        for det_name, metrics in drift_detection.items():
            if not isinstance(metrics, dict):
                continue
            row = {"category": _category(model_name), "model": model_name, "detector": det_name}
            for col in DRIFT_COLS:
                row[col] = metrics.get(col, "")
            rows.append(row)

    _write_csv(
        os.path.join(output_dir, "drift_detection.csv"),
        ["category", "model", "detector"] + DRIFT_COLS,
        rows,
    )


# ---------------------------------------------------------------------------
# Hyperparameter sensitivity
# ---------------------------------------------------------------------------

def export_sensitivity(all_results: Dict, output_dir: str) -> None:
    rows = []
    for model_name, model_data in all_results.items():
        sensitivity = _section(model_data, "sensitivity")
        for param_key, metrics in sensitivity.items():
            if not isinstance(metrics, dict):
                continue
            # param_key looks like "alpha=0.3" or "lambda=5.0"
            if "=" in param_key:
                p_name, p_val = param_key.split("=", 1)
            else:
                p_name, p_val = param_key, ""
            row = {
                "category": _category(model_name),
                "model": model_name,
                "param_name": p_name.strip(),
                "param_value": p_val.strip(),
            }
            for col in ["waste_rate", "oom_rate", "p99_coverage", "p999_coverage",
                        "p95_coverage", "mape", "mape_p95", "mape_p99",
                        "under_allocation_mean", "drift_f1"]:
                row[col] = metrics.get(col, "")
            rows.append(row)

    _write_csv(
        os.path.join(output_dir, "sensitivity.csv"),
        ["category", "model", "param_name", "param_value",
         "waste_rate", "oom_rate", "p99_coverage", "p999_coverage", "p95_coverage",
         "mape", "mape_p95", "mape_p99",
         "under_allocation_mean", "drift_f1"],
        rows,
    )


# ---------------------------------------------------------------------------
# Cold-start transfer
# ---------------------------------------------------------------------------

def export_cold_start(all_results: Dict, output_dir: str) -> None:
    rows = []
    for model_name, model_data in all_results.items():
        cold_start = _section(model_data, "cold_start")
        for method_name, result in cold_start.items():
            if not isinstance(result, dict):
                continue
            row = {"category": _category(model_name), "model": model_name, "method": method_name}
            row["transfer_mape"] = result.get("transfer_mape", result.get("cold_start_mape", ""))
            row["no_transfer_mape"] = result.get("no_transfer_mape", "")
            row["improvement"] = result.get("improvement", "")
            row["n_samples"] = result.get("n_samples", result.get("cold_n", ""))
            row["warm_mape"] = result.get("warm_mape", "")
            rows.append(row)

    _write_csv(
        os.path.join(output_dir, "cold_start.csv"),
        ["category", "model", "method", "transfer_mape", "no_transfer_mape",
         "improvement", "n_samples", "warm_mape"],
        rows,
    )


# ---------------------------------------------------------------------------
# Master summary: all strategies x all models
# ---------------------------------------------------------------------------

def export_master_table(all_results: Dict, output_dir: str) -> None:
    """
    Export a single comprehensive CSV with one row per (model, strategy)
    covering all allocation metrics.
    """
    rows = []
    for model_name, model_data in all_results.items():
        allocation = _section(model_data, "allocation")
        for strategy_name, metrics in allocation.items():
            if not isinstance(metrics, dict):
                continue
            row = {
                "category": _category(model_name),
                "model": model_name,
                "strategy": strategy_name,
            }
            for col in ALLOC_COLS:
                row[col] = metrics.get(col, "")
            rows.append(row)

    _write_csv(
        os.path.join(output_dir, "all_strategies.csv"),
        ["category", "model", "strategy"] + ALLOC_COLS,
        rows,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_all(
    results_dir: str,
    output_dir: str,
    results_file: str = "all_results.json",
) -> None:
    """
    Load all_results.json and export all CSVs.

    Args:
        results_dir:  Directory containing all_results.json and model sub-dirs.
        output_dir:   Directory to write CSV files into.
        results_file: Name of the combined results JSON (default all_results.json).
    """
    json_path = os.path.join(results_dir, results_file)
    if not os.path.exists(json_path):
        fallback = os.path.join(results_dir, "chain_summary.json")
        if os.path.exists(fallback):
            json_path = fallback
        else:
            print(
                f"[ERROR] Results file not found: {json_path} "
                f"(fallback also missing: {fallback})",
                file=sys.stderr,
            )
            sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting results from {json_path} → {output_dir}/")

    print("\n[Allocation] Strategy comparison")
    export_allocation(all_results, output_dir)

    print("\n[Master] All strategies x models table")
    export_master_table(all_results, output_dir)

    print(f"\nDone. All CSVs written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DeCal JSON results to structured CSVs"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory containing all_results.json (default: results)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/csv",
        help="Output directory for CSV files (default: results/csv)"
    )
    parser.add_argument(
        "--results_file", type=str, default="all_results.json",
        help="Name of combined results JSON (default: all_results.json)"
    )
    args = parser.parse_args()
    export_all(args.results_dir, args.output_dir, args.results_file)


if __name__ == "__main__":
    main()
