from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CODE_ROOT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE15_ROOT = Path(_release_paths.expand('${EXTERNAL_GENERALIZATION_ROOT}'))
YOLO_ROOT = Path(_release_paths.expand('${YOLOV5_ROOT}'))
PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))
CARPK_ROOT = Path(_release_paths.expand('${CODE_ROOT}\\metaheuristic\\CARPK'))
UAVDT_ROOT = Path(_release_paths.expand('${CODE_ROOT}\\metaheuristic\\UAVDT'))

MODEL_ORDER = [
    "Meta-AP+APS-B6",
    "Meta-AP-only-B6",
    "RandomSearch-B6",
    "MatchedManual-B6",
    "SourceLocked-Stage6-B6",
]
SEED_ORDER = [42, 43, 44]
VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(records[0].keys()) if records else []
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def model_slug(model: str) -> str:
    return {
        "Meta-AP+APS-B6": "meta_ap_aps_b6",
        "Meta-AP-only-B6": "meta_ap_only_b6",
        "RandomSearch-B6": "random_search_b6",
        "MatchedManual-B6": "matched_manual_b6",
        "SourceLocked-Stage6-B6": "source_locked_stage6_b6",
    }[model]


def load_final_runs() -> list[dict[str, Any]]:
    manifest = read_json(CODE_ROOT / "10_final_fulltrain_300e_matched" / "final_training_manifest.json")
    runs = manifest["runs"]
    require(len(runs) == 15, f"Expected 15 final runs, found {len(runs)}")
    by_key = {(item["model"], int(item["seed"])): item for item in runs}
    expected = [(model, seed) for model in MODEL_ORDER for seed in SEED_ORDER]
    require(set(by_key) == set(expected), f"Final run population mismatch: {sorted(by_key)}")
    return [by_key[key] for key in expected]


def source_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def ensure_output_structure() -> None:
    for relative in (
        "00_PRECHECK",
        "01_CARPK_PREP",
        "02_UAVDT_PREP",
        "03_INFERENCE_CARPK",
        "04_INFERENCE_UAVDT",
        "05_AGGREGATES",
        "06_AUDIT",
        "scripts",
    ):
        (STAGE15_ROOT / relative).mkdir(parents=True, exist_ok=True)


def file_tree(root: Path) -> list[str]:
    lines = [str(root)]
    for current, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        base = Path(current)
        relative = base.relative_to(root)
        depth = len(relative.parts)
        indent = "  " * depth
        if depth:
            lines.append(f"{indent}{base.name}/")
        for name in files:
            path = base / name
            lines.append(f"{indent}  {name}\t{path.stat().st_size}")
    return lines


def scientific_settings() -> dict[str, Any]:
    metric_lock = CODE_ROOT / "11_official_val_statistics" / "00_final_evaluation_role_lock" / "evaluation_metric_lock.json"
    execution_lock = CODE_ROOT / "11_official_val_statistics" / "01_val548_evaluation" / "stage11_execution_lock.json"
    metric = read_json(metric_lock)
    execution = read_json(execution_lock)
    settings = dict(execution["evaluation_settings"])
    settings.update(
        {
            "backend": "PyTorch",
            "multi_scale": False,
            "checkpoint_policy": "terminal EMA after epoch 300",
            "preprocessing": metric["locked_settings"]["preprocessing"],
            "letterbox": metric["locked_settings"]["letterbox"],
            "class_order": metric["class_order"],
        }
    )
    return settings


def normalize_model_name(name: str) -> str:
    if name == "SourceLocked-Stage6-B6":
        return "SourceLocked-B6"
    return name

