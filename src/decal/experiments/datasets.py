"""Dataset registry and validation helpers for DeCal experiments."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ServeGenModelSpec:
    """ServeGen workload metadata used by runners and exporters."""

    name: str
    category: str
    path: str
    default_field: str
    validation_note: str = ""


SERVEGEN_MODELS: Dict[str, ServeGenModelSpec] = {
    "m-large": ServeGenModelSpec(
        name="m-large",
        category="language",
        path="ServeGen/language/m-large",
        default_field="input_tokens",
    ),
    "m-mid": ServeGenModelSpec(
        name="m-mid",
        category="language",
        path="ServeGen/language/m-mid",
        default_field="input_tokens",
    ),
    "m-small": ServeGenModelSpec(
        name="m-small",
        category="language",
        path="ServeGen/language/m-small",
        default_field="input_tokens",
    ),
    "deepseek-r1": ServeGenModelSpec(
        name="deepseek-r1",
        category="reason",
        path="ServeGen/reason/deepseek-r1",
        default_field="input_tokens",
        validation_note=(
            "reason_ratio is a derived ratio rather than a token-length budget target; "
            "the default validation field remains input_tokens."
        ),
    ),
    "mm-image": ServeGenModelSpec(
        name="mm-image",
        category="multimodal",
        path="ServeGen/multimodal/mm-image",
        default_field="output_tokens",
        validation_note=(
            "mm-image does not provide an input_tokens field. Its modality-specific "
            "input marginals (text_tokens, image_tokens, audio_tokens, video_tokens) "
            "are not a joint distribution, so a total input-token PDF would require "
            "an independence assumption. The default validation field is output_tokens; "
            "input-side multimodal budgeting should be reported separately when joint "
            "token traces are available."
        ),
    ),
}

SERVEGEN_CATEGORIES: Dict[str, List[str]] = {
    "language": ["m-large", "m-mid", "m-small"],
    "reason": ["deepseek-r1"],
    "multimodal": ["mm-image"],
}

BURSTGPT_MODELS: Dict[str, str] = {
    "burstgpt": "BurstGPT",
}


def servegen_model_paths() -> Dict[str, str]:
    """Return model-name to path mapping relative to the experiment data root."""

    return {name: spec.path for name, spec in SERVEGEN_MODELS.items()}


def servegen_model_categories() -> Dict[str, str]:
    """Return model-name to ServeGen category mapping."""

    return {name: spec.category for name, spec in SERVEGEN_MODELS.items()}


def servegen_model_specs(models: Optional[Iterable[str]] = None) -> List[ServeGenModelSpec]:
    """Return specs in registry order, optionally restricted to model names."""

    if models is None:
        return list(SERVEGEN_MODELS.values())
    wanted = set(models)
    return [spec for name, spec in SERVEGEN_MODELS.items() if name in wanted]


def filter_servegen_models(
    models: Optional[Iterable[str]] = None,
    categories: Optional[Iterable[str]] = None,
) -> List[str]:
    """Resolve a ServeGen model list from optional model/category filters."""

    if models:
        base = [m for m in models if m in SERVEGEN_MODELS]
    else:
        base = list(SERVEGEN_MODELS.keys())

    if categories:
        allowed_categories = set(categories)
        base = [m for m in base if SERVEGEN_MODELS[m].category in allowed_categories]

    return base


def default_field_for_model(model_name: str) -> str:
    """Return the default validation field for a ServeGen model."""

    return SERVEGEN_MODELS[model_name].default_field


def resolve_field_for_model(model_name: str, field_override: Optional[str]) -> str:
    """Resolve ``auto``/unset field names to a model-appropriate default."""

    if field_override and field_override != "auto":
        return field_override
    return default_field_for_model(model_name)


def infer_servegen_category(model_name: str) -> str:
    """Infer ServeGen category from a model name."""

    spec = SERVEGEN_MODELS.get(model_name)
    return spec.category if spec else "unknown"


def _sample_fields(dataset_path: str) -> List[str]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for fields in data.values():
        if isinstance(fields, dict):
            return sorted(fields.keys())
    return []


def validate_servegen_layout(
    data_root: str,
    models: Optional[Iterable[str]] = None,
) -> Dict[str, dict]:
    """Validate local ServeGen files and return a serializable report.

    The check is intentionally lightweight: it verifies that each selected
    workload directory exists, counts paired trace/dataset chunks, samples
    available fields, and marks whether the model's default validation field is
    available. It does not parse every PDF, so it can run before long replays.
    """

    report: Dict[str, dict] = {}
    for spec in servegen_model_specs(models):
        data_dir = os.path.join(data_root, spec.path)
        exists = os.path.isdir(data_dir)
        dataset_files = []
        trace_files = []
        fields: List[str] = []
        missing_reason = ""

        if exists:
            dataset_files = sorted(f for f in os.listdir(data_dir) if f.endswith("-dataset.json"))
            trace_files = sorted(f for f in os.listdir(data_dir) if f.endswith("-trace.csv"))
            if dataset_files:
                fields = _sample_fields(os.path.join(data_dir, dataset_files[0]))
            else:
                missing_reason = "no dataset JSON files found"
        else:
            missing_reason = "data directory does not exist"

        report[spec.name] = {
            **asdict(spec),
            "data_dir": data_dir,
            "exists": exists,
            "n_dataset_files": len(dataset_files),
            "n_trace_files": len(trace_files),
            "paired_chunks": min(len(dataset_files), len(trace_files)),
            "sample_fields": fields,
            "default_field_available": spec.default_field in fields,
            "status": "ok" if exists and dataset_files and spec.default_field in fields else "not_usable",
            "not_usable_reason": missing_reason,
        }

        if exists and dataset_files and spec.default_field not in fields:
            report[spec.name]["not_usable_reason"] = (
                f"default field {spec.default_field!r} not present in sampled fields"
            )

    return report

