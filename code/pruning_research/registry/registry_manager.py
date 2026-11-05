from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pruning_research.utils.io_utils import ensure_dir, load_csv_rows, save_csv


MODEL_REGISTRY_HEADERS = [
    "model_id",
    "parent_model_id",
    "stage",
    "weights_path",
    "checkpoint_type",
    "model_source",
    "fused",
    "nc",
    "stride",
    "status",
    "notes",
    "updated_utc",
]

EXPERIMENT_REGISTRY_HEADERS = [
    "experiment_id",
    "experiment_type",
    "model_id",
    "config_hash",
    "timestamp_utc",
    "status",
    "output_dir",
    "summary_path",
    "notes",
]


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_registry_files(registry_dir: str | Path) -> Dict[str, Path]:
    target = ensure_dir(registry_dir)
    model_registry = target / "model_registry.csv"
    experiment_registry = target / "experiment_registry.csv"

    if not model_registry.exists():
        save_csv([], model_registry, fieldnames=MODEL_REGISTRY_HEADERS)
    if not experiment_registry.exists():
        save_csv([], experiment_registry, fieldnames=EXPERIMENT_REGISTRY_HEADERS)

    return {
        "registry_dir": target,
        "model_registry_csv": model_registry,
        "experiment_registry_csv": experiment_registry,
    }


def _upsert_rows(rows: List[Dict[str, Any]], key_field: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    key_value = str(payload.get(key_field, "")).strip()
    if not key_value:
        raise ValueError(f"Missing required key field: {key_field}")

    updated = False
    normalized_payload = {k: ("" if v is None else str(v)) for k, v in payload.items()}
    result: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get(key_field, "")).strip() == key_value:
            merged = dict(row)
            merged.update(normalized_payload)
            result.append(merged)
            updated = True
        else:
            result.append(row)

    if not updated:
        result.append(normalized_payload)
    return result


def register_model(registry_dir: str | Path, payload: Dict[str, Any]) -> Path:
    paths = ensure_registry_files(registry_dir)
    target = paths["model_registry_csv"]
    rows = load_csv_rows(target)
    payload = dict(payload)
    payload["updated_utc"] = _now_iso_utc()
    updated_rows = _upsert_rows(rows, "model_id", payload)
    save_csv(updated_rows, target, fieldnames=MODEL_REGISTRY_HEADERS)
    return target


def register_experiment(registry_dir: str | Path, payload: Dict[str, Any]) -> Path:
    paths = ensure_registry_files(registry_dir)
    target = paths["experiment_registry_csv"]
    rows = load_csv_rows(target)
    payload = dict(payload)
    if not payload.get("timestamp_utc"):
        payload["timestamp_utc"] = _now_iso_utc()
    updated_rows = _upsert_rows(rows, "experiment_id", payload)
    save_csv(updated_rows, target, fieldnames=EXPERIMENT_REGISTRY_HEADERS)
    return target
