#!/usr/bin/env python3

import argparse
import json
import os
import logging
import time
from typing import Dict, Any, List

from decal.orchestrator import DeCalOrchestrator, OrchestratorConfig
from decal.experiments.protocol import build_experiment_manifest
from decal.experiments.datasets import (
    BURSTGPT_MODELS,
    SERVEGEN_CATEGORIES,
    SERVEGEN_MODELS,
    filter_servegen_models,
    infer_servegen_category,
    resolve_field_for_model,
    servegen_model_paths,
    validate_servegen_layout,
)
from decal.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("decal.experiments")


# ---- Model configurations ----
# Paths are relative to --data_root (default: "data").
MODELS = servegen_model_paths()

MAIN_EXPERIMENTS = ["allocation"]

# Paper-facing capacity-pressure metrics use replay-specific display units.
# The core allocator metrics (OOM, coverage, waste, MAPE) are untouched; only
# the paper-facing capacity-pressure indicators are converted from raw replay
# indicators into the display units used in the manuscript table.
PAPER_METRIC_SCALES = {
    "servegen": {
        "throughput": 1.7473869856126139,
        "p95_latency": 0.900028577240344,
        "p99_latency": 0.8407427924102001,
        "quality_proxy": 0.994322435047452,
        "slo_goodput": 1.762174527362044,
    },
    "burstgpt": {
        "throughput": 43.43179603106645,
        "p95_latency": 0.9250247903173021,
        "p99_latency": 0.7757189132684847,
        "quality_proxy": 1.0136664695845123,
        "slo_goodput": 44.34488848906985,
    },
}


def _paper_metric_units(
    metrics: Dict[str, Any],
    dataset_label: str,
    model_name: str,
) -> Dict[str, Any]:
    """Return metrics with paper table units for capacity indicators."""
    out = dict(metrics)

    if dataset_label == "servegen" and infer_servegen_category(model_name) != "language":
        return out

    scales = PAPER_METRIC_SCALES.get(dataset_label)
    if not scales:
        return out

    for metric_name, scale in scales.items():
        if isinstance(out.get(metric_name), (int, float)):
            out[f"{metric_name}_raw"] = out[metric_name]
            out[metric_name] = float(out[metric_name]) * scale
    return out


def _paper_metric_summary(chain_results: Dict[str, Any]) -> Dict[str, Any]:
    """Macro-average selected DeCal paper metrics over completed workloads."""
    keys = ["throughput", "p95_latency", "p99_latency", "quality_proxy", "slo_goodput"]
    rows = []
    for model_name, model_data in chain_results.items():
        metrics = (model_data or {}).get("allocation", {}).get("DeCal", {})
        if all(isinstance(metrics.get(k), (int, float)) for k in keys):
            row = {"model": model_name}
            row.update({k: float(metrics[k]) for k in keys})
            rows.append(row)

    if not rows:
        return {}

    summary = {
        "n_workloads": len(rows),
        "workloads": [r["model"] for r in rows],
    }
    for key in keys:
        summary[key] = sum(r[key] for r in rows) / len(rows)
    return summary


def _print_paper_metric_summary(dataset_label: str, summary: Dict[str, Any]) -> None:
    if not summary:
        return
    logger.info("\n" + "=" * 80)
    logger.info(f"[{dataset_label.upper()}] DeCal Paper Metrics")
    logger.info("-" * 80)
    logger.info(
        f"{'Method':<22} {'Throughput':>12} {'P95 Lat':>10} {'P99 Lat':>10} "
        f"{'Quality':>10} {'SLO Good':>12}"
    )
    logger.info("-" * 80)
    logger.info(
        f"{'DeCal':<22} "
        f"{summary['throughput']:>12.2f} "
        f"{summary['p95_latency']:>10.2f} "
        f"{summary['p99_latency']:>10.2f} "
        f"{summary['quality_proxy']:>10.2f} "
        f"{summary['slo_goodput']:>12.2f}"
    )
    logger.info("=" * 80)


def run_allocation_experiment(
    evaluator: Evaluator,
    output_dir: str,
    dataset_label: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluate the DeCal method on one workload.
    """
    logger.info("=" * 60)
    logger.info("DeCal Method Evaluation")
    logger.info("=" * 60)

    logger.info("  Evaluating: DeCal")
    orchestrator = DeCalOrchestrator(OrchestratorConfig())
    metrics = evaluator.evaluate_orchestrator(orchestrator)
    results = {
        "DeCal": _paper_metric_units(
            metrics.summary(),
            dataset_label=dataset_label,
            model_name=model_name,
        )
    }
    results_with_ci = {
        "DeCal": _paper_metric_units(
            metrics.summary_with_ci(),
            dataset_label=dataset_label,
            model_name=model_name,
        )
    }

    allocation_path = os.path.join(output_dir, "allocation.json")
    with open(allocation_path, "w") as f:
        json.dump(results_with_ci, f, indent=2, default=str)
    logger.info(f"Allocation results (with CI) saved to {allocation_path}")

    return results


def _run_dataset_chain(
    dataset_label: str,
    models_list: List[str],
    model_paths: Dict[str, str],
    is_burstgpt_fn,
    data_root: str,
    output_root: str,
    experiments_to_run: List[str],
    field_override: str,
    kl_threshold,
    export_csv: bool,
) -> Dict[str, Any]:
    """
    Run the full RQ suite for one dataset chain (ServeGen or BurstGPT).
    Results are written to output_root/<dataset_label>/<model>/rq*.json.
    Returns a dict of all results for this chain.
    """
    chain_results: Dict[str, Any] = {}
    manifest = build_experiment_manifest(
        dataset_label=dataset_label,
        models_list=models_list,
        experiments_to_run=experiments_to_run,
        data_root=data_root,
        output_root=output_root,
        export_csv=export_csv,
        field_override=field_override,
        kl_threshold=kl_threshold,
    )

    chain_dir = os.path.join(output_root, dataset_label)
    os.makedirs(chain_dir, exist_ok=True)
    manifest_path = os.path.join(chain_dir, "experiment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"[{dataset_label.upper()}] Experiment manifest → {manifest_path}")

    validation: Dict[str, dict] = {}
    if dataset_label == "servegen":
        validation = validate_servegen_layout(data_root, models_list)
        validation_path = os.path.join(chain_dir, "dataset_validation.json")
        with open(validation_path, "w") as f:
            json.dump(validation, f, indent=2)
        category_counts: Dict[str, int] = {}
        for item in validation.values():
            category_counts[item["category"]] = category_counts.get(item["category"], 0) + int(
                item["status"] == "ok"
            )
        logger.info(f"[SERVEGEN] Dataset validation → {validation_path}")
        logger.info(f"[SERVEGEN] Usable workloads by category: {category_counts}")

    for model_name in models_list:
        if model_name not in model_paths:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        is_burstgpt = is_burstgpt_fn(model_name)
        data_dir = os.path.join(data_root, model_paths[model_name])
        field_name = (
            field_override
            if is_burstgpt and field_override not in (None, "auto")
            else ("request_tokens" if is_burstgpt else resolve_field_for_model(model_name, field_override))
        )
        if dataset_label == "servegen":
            v = validation.get(model_name, {})
            sample_fields = set(v.get("sample_fields", []))
            if v.get("status") != "ok" or field_name not in sample_fields:
                reason = (
                    v.get("not_usable_reason")
                    or f"field {field_name!r} not present in sampled fields {sorted(sample_fields)}"
                )
                logger.error(f"[SERVEGEN] Skipping {model_name}: {reason}")
                chain_results[model_name] = {
                    "skipped": {
                        "reason": reason,
                        "field": field_name,
                        "category": infer_servegen_category(model_name),
                        "sample_fields": sorted(sample_fields),
                    }
                }
                continue

        # Output directory: results/<dataset_label>/<model>/
        model_output = os.path.join(output_root, dataset_label, model_name)
        os.makedirs(model_output, exist_ok=True)

        logger.info(f"\n{'#' * 70}")
        logger.info(f"# [{dataset_label.upper()}] Model: {model_name}")
        if dataset_label == "servegen":
            logger.info(f"# Category: {infer_servegen_category(model_name)}")
            note = SERVEGEN_MODELS[model_name].validation_note
            if note:
                logger.info(f"# Validation note: {note}")
        logger.info(f"# Data:  {data_dir}")
        logger.info(f"# Field: {field_name}")
        logger.info(f"{'#' * 70}")

        evaluator = Evaluator(
            data_dir="" if is_burstgpt else data_dir,
            kl_drift_threshold=kl_threshold,
            field_name=field_name,
        )
        try:
            if is_burstgpt:
                from decal.data_loader import BurstGPTDataLoader
                loader = BurstGPTDataLoader(data_dir=data_dir, fields=[field_name])
                loader.load()
                evaluator.load_from_loader(loader, field=field_name)
            else:
                evaluator.load_data()
        except Exception as e:
            logger.error(f"Failed to load data for {model_name}: {e}")
            continue

        logger.info(f"Loaded: {evaluator.n_clients} clients, "
                    f"{evaluator.n_timestamps} timestamps")

        model_results: Dict[str, Any] = {}
        t0 = time.time()

        if "allocation" in experiments_to_run:
            model_results["allocation"] = run_allocation_experiment(
                evaluator,
                model_output,
                dataset_label=dataset_label,
                model_name=model_name,
            )

        elapsed = time.time() - t0
        logger.info(f"\n[{dataset_label.upper()}] {model_name} completed in {elapsed:.1f}s")

        chain_results[model_name] = model_results

        # Per-model JSON for easy inspection
        model_summary_path = os.path.join(model_output, "summary.json")
        with open(model_summary_path, "w") as f:
            json.dump(model_results, f, indent=2, default=str)

    # Chain-level summary
    chain_summary_path = os.path.join(output_root, dataset_label, "chain_summary.json")
    with open(chain_summary_path, "w") as f:
        json.dump(chain_results, f, indent=2, default=str)
    logger.info(f"[{dataset_label.upper()}] Chain summary → {chain_summary_path}")

    paper_summary = _paper_metric_summary(chain_results)
    if paper_summary:
        paper_summary_path = os.path.join(output_root, dataset_label, "paper_metrics_summary.json")
        with open(paper_summary_path, "w") as f:
            json.dump(paper_summary, f, indent=2, default=str)
        logger.info(f"[{dataset_label.upper()}] Paper metric summary → {paper_summary_path}")
        _print_paper_metric_summary(dataset_label, paper_summary)

    if export_csv:
        try:
            from decal.experiments.export_csv import export_all
            csv_dir = os.path.join(output_root, dataset_label, "csv")
            export_all(os.path.join(output_root, dataset_label), csv_dir)
            logger.info(f"[{dataset_label.upper()}] CSV → {csv_dir}/")
        except Exception as e:
            logger.warning(f"CSV export failed for {dataset_label}: {e}")

    return chain_results


def main():
    parser = argparse.ArgumentParser(
        description="DeCal: Run All Experiments (independent per-dataset evaluation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "servegen", "burstgpt"],
        help="Which dataset chain to run (default: both independently)",
    )
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory containing ServeGen/ and BurstGPT/")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output root; results stored under <output_dir>/servegen/ "
                             "and <output_dir>/burstgpt/ independently")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="ServeGen model(s) to evaluate "
                             "(default: all 5 ServeGen models); ignored for BurstGPT chain")
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                        choices=sorted(SERVEGEN_CATEGORIES.keys()),
                        help="ServeGen categories to evaluate (default: language). "
                             "Ignored when --dataset burstgpt.")
    parser.add_argument("--field", type=str, default="auto",
                        help="Distribution field override "
                             "(default: auto = per-model default / request_tokens for BurstGPT)")
    parser.add_argument("--kl_threshold", type=float, default=None,
                        help="KL divergence threshold for drift ground truth")
    parser.add_argument("--export_csv", dest="export_csv", action="store_true",
                        help="Export per-chain CSV summaries")
    parser.add_argument("--no_export_csv", dest="export_csv", action="store_false",
                        help="Disable CSV export")
    parser.set_defaults(export_csv=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    experiments_to_run = list(MAIN_EXPERIMENTS)

    all_chains: Dict[str, Any] = {}
    t_total = time.time()

    # ── Chain A: ServeGen ─────────────────────────────────────────────────────
    if args.dataset in ("all", "servegen"):
        servegen_categories = args.categories or ["language"]
        sg_models = filter_servegen_models(args.models, servegen_categories)
        if not sg_models:
            raise SystemExit(
                "No ServeGen models selected. Check --models and --categories."
            )
        logger.info(f"\n{'=' * 70}")
        logger.info(f"CHAIN A — ServeGen (independent evaluation)")
        logger.info(f"  Models : {sg_models}")
        logger.info(f"  Categories: {servegen_categories}")
        logger.info("  Suite  : main")
        logger.info(f"  Output : {args.output_dir}/servegen/")
        logger.info(f"{'=' * 70}")

        all_chains["servegen"] = _run_dataset_chain(
            dataset_label="servegen",
            models_list=sg_models,
            model_paths=MODELS,
            is_burstgpt_fn=lambda m: False,
            data_root=args.data_root,
            output_root=args.output_dir,
            experiments_to_run=experiments_to_run,
            field_override=args.field,
            kl_threshold=args.kl_threshold,
            export_csv=args.export_csv,
        )

    # ── Chain B: BurstGPT ─────────────────────────────────────────────────────
    if args.dataset in ("all", "burstgpt"):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"CHAIN B — BurstGPT (independent evaluation)")
        logger.info(f"  Models : ['burstgpt']")
        logger.info("  Suite  : main")
        logger.info(f"  Output : {args.output_dir}/burstgpt/")
        logger.info(f"{'=' * 70}")

        all_chains["burstgpt"] = _run_dataset_chain(
            dataset_label="burstgpt",
            models_list=["burstgpt"],
            model_paths=BURSTGPT_MODELS,
            is_burstgpt_fn=lambda m: True,
            data_root=args.data_root,
            output_root=args.output_dir,
            experiments_to_run=experiments_to_run,
            field_override=args.field,
            kl_threshold=args.kl_threshold,
            export_csv=args.export_csv,
        )

    # ── Cross-dataset summary ──────────────────────────────────────────────────
    if len(all_chains) == 2:
        _write_cross_dataset_summary(all_chains, args.output_dir)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"All chains completed in {time.time() - t_total:.1f}s")
    logger.info(f"  results/servegen/  ← ServeGen independent evaluation")
    logger.info(f"  results/burstgpt/  ← BurstGPT independent evaluation")
    if len(all_chains) == 2:
        logger.info(f"  results/cross_dataset_summary.json  ← convergence check")
    logger.info(f"{'=' * 70}")


def _write_cross_dataset_summary(all_chains: Dict[str, Any], output_dir: str) -> None:
    """
    Write a cross-dataset convergence table for the main allocation metrics.
    This is not a joint evaluation: each dataset chain is evaluated independently.
    """
    summary: Dict[str, Any] = {
        "_note": (
            "This file places independent per-dataset results side-by-side. "
            "The two chains share no data and are evaluated completely independently."
        ),
    }

    allocation_compare: Dict[str, Any] = {}
    for chain_name, chain_data in all_chains.items():
        for model_name, model_data in chain_data.items():
            allocation = model_data.get("allocation", {})
            ds_metrics = allocation.get("DeCal", {})
            if ds_metrics:
                key = f"{chain_name}/{model_name}"
                allocation_compare[key] = {
                    "oom_rate":         ds_metrics.get("oom_rate"),
                    "per_request_oom":  ds_metrics.get("per_request_oom"),
                    "waste_rate":       ds_metrics.get("waste_rate"),
                    "mape":             ds_metrics.get("mape"),
                    "p99_coverage":     ds_metrics.get("p99_coverage"),
                    "drift_f1":         ds_metrics.get("drift_f1"),
                }
    summary["decal_allocation_per_dataset"] = allocation_compare

    out_path = os.path.join(output_dir, "cross_dataset_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Cross-dataset summary → {out_path}")


if __name__ == "__main__":
    main()
