"""Experiment protocol registry for DeCal.

The goal is to keep the evaluation contract explicit in code: datasets,
experiments, metrics, and artifact layout.
"""
from __future__ import annotations

from decal.experiments.datasets import SERVEGEN_MODELS, servegen_model_categories


PROJECT_NAME = "DeCal"
PROBLEM_STATEMENT = (
    "under allocate-before-observe, emit one client-level, window-level "
    "tail-length proxy budget from completed history only, so as to reduce "
    "post-drift proxy failure exposure while bounding long-run over-allocation"
)
BUDGET_DEFINITION = (
    "one scalar tail-length proxy budget per client per window, in token-length-bin units, "
    "used as a monotone proxy for KV-cache pressure rather than as an exact engine reservation"
)
METHOD_POSITIONING = (
    "Core method: a causal two-path controller that combines a drift-triggered "
    "safety path with a decoupled calibration path, centered on the Coupling Trap "
    "and its decoupling invariant. Trigger: an online detector over cheap scalar "
    "summaries (e.g., mean or P95). Extension: optional cross-client transfer for "
    "cold-start and severe-drift recovery."
)

DATASET_CHAINS = {
    "servegen": {
        "role": "multi-client production-inspired workloads spanning language, reasoning, and multimodal classes",
        "coverage": "5 workloads, 422 clients, 14 days, 10-minute windows",
        "categories": {
            "language": ["m-large", "m-mid", "m-small"],
            "reason": ["deepseek-r1"],
            "multimodal": ["mm-image"],
        },
        "experiments": ["allocation"],
    },
    "burstgpt": {
        "role": "real production traffic validating that workload drift is not an artifact of one benchmark family",
        "coverage": "4 client streams, 231 days, 1-hour windows",
        "experiments": ["allocation"],
    },
}

EXPERIMENT_SPECS = {
    "allocation": {
        "title": "allocation performance comparison",
        "question": "how should waste vs failure trade off under non-stationary workloads?",
        "outputs": ["allocation.json"],
    },
}

METRIC_SPECS = [
    {
        "name": "oom_rate",
        "category": "window failure",
        "interpretation": "fraction of windows where actual P99 exceeds the capacity budget",
    },
    {
        "name": "per_request_oom",
        "category": "request failure",
        "interpretation": "tail probability mass above the allocated budget",
    },
    {
        "name": "waste_rate",
        "category": "over-allocation",
        "interpretation": "fraction of reserved capacity not needed by realized P99 demand",
    },
    {
        "name": "p95_coverage",
        "category": "coverage",
        "interpretation": "share of windows whose P95 demand is covered by the budget",
    },
    {
        "name": "p99_coverage",
        "category": "coverage",
        "interpretation": "share of windows whose P99 demand is covered by the budget (target)",
    },
    {
        "name": "p999_coverage",
        "category": "coverage",
        "interpretation": "share of windows whose P99.9 tail is covered by the budget",
    },
    {
        "name": "mape_p95",
        "category": "estimation quality",
        "interpretation": "absolute percentage error of budget vs actual P95",
    },
    {
        "name": "mape_p99",
        "category": "estimation quality",
        "interpretation": "absolute percentage error of budget vs actual P99 (target)",
    },
    {
        "name": "mape",
        "category": "estimation quality",
        "interpretation": "legacy alias for mape_p95 (kept for backward compatibility)",
    },
    {
        "name": "over_allocation_mean",
        "category": "over-allocation severity",
        "interpretation": "mean tokens reserved above actual P99 per window",
    },
    {
        "name": "over_allocation_p95",
        "category": "over-allocation severity",
        "interpretation": "95th-percentile of per-window over-allocation in tokens",
    },
    {
        "name": "under_allocation_mean",
        "category": "OOM severity",
        "interpretation": "mean token shortfall on windows where budget < actual P99",
    },
    {
        "name": "under_allocation_p95",
        "category": "OOM severity",
        "interpretation": "95th-percentile of token shortfall over OOM windows",
    },
    {
        "name": "worst_window_oom",
        "category": "OOM severity",
        "interpretation": "largest per-request OOM probability mass observed in any window",
    },
    {
        "name": "consecutive_oom_max",
        "category": "OOM severity",
        "interpretation": "longest streak of consecutive OOM windows for any client",
    },
    {
        "name": "drift_precision",
        "category": "drift detection",
        "interpretation": "precision of predicted drift events against divergence-based labels",
    },
    {
        "name": "drift_recall",
        "category": "drift detection",
        "interpretation": "recall of predicted drift events against divergence-based labels",
    },
    {
        "name": "drift_f1",
        "category": "drift detection",
        "interpretation": "harmonic mean of drift precision and recall",
    },
    {
        "name": "drift_fpr",
        "category": "drift detection",
        "interpretation": "false positive rate of drift alarms",
    },
]

def build_experiment_manifest(
    dataset_label: str,
    models_list: list[str],
    experiments_to_run: list[str],
    data_root: str,
    output_root: str,
    export_csv: bool,
    field_override: str | None,
    kl_threshold: float | None,
) -> dict:
    """Return a chain-level evaluation manifest for DeCal."""
    chain = DATASET_CHAINS[dataset_label]
    effective_experiments = [
        exp for exp in experiments_to_run if exp in chain["experiments"]
    ]
    servegen_fields = {
        name: SERVEGEN_MODELS[name].default_field
        for name in models_list
        if dataset_label == "servegen" and name in SERVEGEN_MODELS
    }
    return {
        "project_name": PROJECT_NAME,
        "problem_statement": PROBLEM_STATEMENT,
        "budget_definition": BUDGET_DEFINITION,
        "method_positioning": METHOD_POSITIONING,
        "dataset_chain": {
            "name": dataset_label,
            **chain,
            "models": models_list,
            "model_categories": servegen_model_categories() if dataset_label == "servegen" else {},
            "default_validation_fields": servegen_fields,
        },
        "experiments": {
            exp: EXPERIMENT_SPECS[exp] for exp in effective_experiments
        },
        "metrics": METRIC_SPECS,
        "drift_monitoring_note": (
            "The online detector consumes scalar workload summaries such as mean "
            "or P95 token length, while evaluation labels drift using divergence "
            "between consecutive full distributions."
        ),
        "execution": {
            "data_root": data_root,
            "output_root": output_root,
            "field_override": field_override,
            "kl_threshold": kl_threshold,
            "export_csv": export_csv,
        },
        "artifacts": {
            "chain_summary": f"{dataset_label}/chain_summary.json",
            "cross_dataset_summary": "cross_dataset_summary.json",
            "csv_dir": f"{dataset_label}/csv/",
            "manifest_json": f"{dataset_label}/experiment_manifest.json",
        },
    }
