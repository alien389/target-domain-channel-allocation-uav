#!/usr/bin/env python3
"""Prepare the evidence-locked Stage02 Lite-v3 SearchParent run."""

from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
YOLO_ROOT = Path(_release_paths.expand('${YOLOV5_ROOT}'))
PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))
TRAIN_SCRIPT = YOLO_ROOT / "train_drone.py"
MODEL_YAML = YOLO_ROOT / "models" / "yolov5s_lite_v3.yaml"
SOURCE_WEIGHTS = Path(_release_paths.expand('${SOURCE_REFERENCE_ROOT}\\checkpoints\\lite_v3_full.pt'))
HYP_YAML = Path(_release_paths.expand('${SOURCE_PARENT_HYP}'))
DATA_YAML = PROJECT / "01_data_split" / "data_yamls" / "visdrone_meta_train5177_inner1294.yaml"
TRAIN_LIST = PROJECT / "01_data_split" / "meta_train5177_seed42.txt"
INNER_VAL_LIST = PROJECT / "01_data_split" / "meta_inner_val1294_seed42.txt"
OFFICIAL_LOCKED_LIST = PROJECT / "01_data_split" / "official_val548_locked.txt"
AMENDMENT_STATE = PROJECT / "01A_protocol_amendment" / "amendment_state.json"
AMENDMENT_VALIDATION = PROJECT / "01A_protocol_amendment" / "amendment_validation.json"
APPROVAL_TOKEN = PROJECT / "manifests" / "approvals" / "stage01a.token"
STAGE_ROOT = PROJECT / "02_search_parent_teacher"
RUN_DIR = STAGE_ROOT / "lite_v3_search_parent"

EXPECTED_SHA256 = {
    SOURCE_WEIGHTS: "01F7185F50FF96E61277916856A56EA4CF0EE22755D5E9812E6C4A1DAF15FDBD",
    MODEL_YAML: "724080FB67EED6D413BA39CCA5084E3B84ADE4296BAAE3FB97A98A24E70D5CAB",
    HYP_YAML: "E6A6D755AD525C4DAC609334C528EC21C219D4E355096B9B9599206C12115F4B",
    DATA_YAML: "79ABEBA90BB791F75809502E7F116A5AD51F7B00D4003E11CB69250CAA41B5CF",
    TRAIN_SCRIPT: "52549EDD8C0FA2890BBD6843460D34C0238E33082F900987EDC636BACBD7243A",
    TRAIN_LIST: "A890318CB684513A4A4F27FC5E0FEDA7207A54CC554D556940907B2FBD1EA936",
    INNER_VAL_LIST: "1814CF80594A014FAB3AD8D6F80D956441FD06EAA43DDDA8970D148F82827278",
    OFFICIAL_LOCKED_LIST: "8A30707BCC6FB6E119C929BAD1A4D531299E497A9762666AD70A9C6A31B2DE4A",
}

EXPECTED_CLASSES = [
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" not in lowered:
        raise ValueError(f"Image path does not contain an images component: {image_path}")
    index = lowered.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def forbidden_dataset_path(path: str) -> bool:
    normalized = path.replace("/", "\\").lower()
    return any(
        token in normalized
        for token in (
            "\\images\\val\\",
            "\\labels\\val\\",
            "visdrone2019-det-val",
            "test-dev",
            "testdev",
        )
    )


def validate_split_list(path: Path, expected_count: int) -> dict[str, Any]:
    rows = read_nonempty_lines(path)
    if len(rows) != expected_count:
        raise RuntimeError(f"{path.name}: expected {expected_count} rows, found {len(rows)}")
    if len(set(rows)) != len(rows):
        raise RuntimeError(f"{path.name}: duplicate image paths found")
    missing_images: list[str] = []
    missing_labels: list[str] = []
    forbidden_paths: list[str] = []
    for row in rows:
        image = Path(row)
        if not image.is_file():
            missing_images.append(row)
        label = label_path_for_image(image)
        if not label.is_file():
            missing_labels.append(str(label))
        if forbidden_dataset_path(row):
            forbidden_paths.append(row)
    if missing_images or missing_labels or forbidden_paths:
        raise RuntimeError(
            f"{path.name}: missing_images={len(missing_images)}, "
            f"missing_labels={len(missing_labels)}, forbidden_paths={len(forbidden_paths)}"
        )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "count": len(rows),
        "unique_count": len(set(rows)),
        "missing_images": 0,
        "missing_labels": 0,
        "forbidden_paths": 0,
    }


def active_prohibited_processes() -> list[dict[str, Any]]:
    current_pid = os.getpid()
    found: list[dict[str, Any]] = []
    try:
        import psutil

        current_process = psutil.Process(current_pid)
        excluded_pids = {current_pid, *(parent.pid for parent in current_process.parents())}
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            if process.info["pid"] in excluded_pids:
                continue
            arguments = process.info.get("cmdline") or []
            executable_tokens = {Path(argument).name.lower() for argument in arguments}
            prohibited_script = bool(
                executable_tokens.intersection(
                    {
                        "train.py",
                        "train_drone.py",
                        "val.py",
                        "taylor_scoring.py",
                        "prune.py",
                        "nsga2.py",
                        "random_search.py",
                        "kd1.py",
                        "kd2.py",
                    }
                )
            )
            command = " ".join(arguments)
            if prohibited_script:
                found.append(
                    {
                        "pid": process.info["pid"],
                        "name": process.info.get("name"),
                        "command": command,
                    }
                )
    except ImportError:
        pass
    return found


def nvidia_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def recursive_shapes(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [recursive_shapes(item) for item in value]
    if isinstance(value, dict):
        return {str(key): recursive_shapes(item) for key, item in value.items()}
    return type(value).__name__


def measure_complexity(model: Any, imgsz: int) -> dict[str, Any]:
    import torch
    from thop import profile

    def stats(instance: Any) -> dict[str, Any]:
        parameter_count = sum(parameter.numel() for parameter in instance.parameters())
        stride = max(int(instance.stride.max()), 32) if hasattr(instance, "stride") else 32
        first_parameter = next(instance.parameters())
        sample = torch.zeros((1, 3, stride, stride), dtype=first_parameter.dtype)
        macs = profile(copy.deepcopy(instance).cpu(), inputs=(sample,), verbose=False)[0]
        scale = (imgsz / stride) ** 2
        macs_g = float(macs * scale / 1e9)
        return {
            "params": int(parameter_count),
            "params_m": float(parameter_count / 1e6),
            "macs_g": macs_g,
            "flops_g_thop_macs_x2": float(2.0 * macs_g),
            "input_shape": [1, 3, imgsz, imgsz],
        }

    unfused = stats(model)
    fused_model = copy.deepcopy(model).cpu().eval().fuse()
    fused = stats(fused_model)
    return {
        "protocol": "THOP MACs x 2 at batch=1, nc=10, 640x640",
        "unfused_diagnostic": unfused,
        "fused_inference_graph": fused,
    }


def build_weight_transfer_report() -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if str(YOLO_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLO_ROOT))
    from models.yolo import Model
    from utils.general import intersect_dicts

    checkpoint = torch.load(SOURCE_WEIGHTS, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError("COCO source checkpoint has no model object")
    source_model = checkpoint["model"].float()
    source_state = source_model.state_dict()
    hyp = yaml.safe_load(HYP_YAML.read_text(encoding="utf-8"))
    target_model = Model(str(MODEL_YAML), ch=3, nc=10, anchors=hyp.get("anchors")).float()
    target_state = target_model.state_dict()
    excluded_fragments = ["anchor"]
    transferred = intersect_dicts(source_state, target_state, exclude=excluded_fragments)
    load_result = target_model.load_state_dict(transferred, strict=False)

    detect_index = len(target_model.model) - 1
    detect_prefix = f"model.{detect_index}."
    shape_mismatch = {
        key: {
            "source_shape": list(value.shape),
            "target_shape": list(target_state[key].shape),
        }
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    }
    unexpected = sorted(key for key in source_state if key not in target_state)
    excluded = sorted(
        key for key in source_state if any(fragment in key for fragment in excluded_fragments)
    )
    loaded_keys = sorted(transferred)
    missing_keys = sorted(load_result.missing_keys)
    rebuilt_detect_keys = sorted(key for key in missing_keys if key.startswith(detect_prefix))

    target_model.eval()
    with torch.no_grad():
        output = target_model(torch.zeros((1, 3, 640, 640), dtype=torch.float32))
    output_shapes = recursive_shapes(output)
    detect_module = target_model.model[-1]
    detect_scales = len(getattr(detect_module, "m", []))
    nc = int(getattr(detect_module, "nc", -1))
    if nc != 10 or detect_scales != 3:
        raise RuntimeError(f"Detect sanity failed: nc={nc}, scales={detect_scales}")

    report = {
        "created_utc": utc_now(),
        "transfer_method": "YOLOv5 intersect_dicts with shape-compatible keys and anchor exclusion",
        "source_checkpoint": str(SOURCE_WEIGHTS),
        "source_checkpoint_sha256": sha256(SOURCE_WEIGHTS),
        "target_model_yaml": str(MODEL_YAML),
        "target_model_yaml_sha256": sha256(MODEL_YAML),
        "source_checkpoint_nc": int(getattr(source_model.model[-1], "nc", -1)),
        "target_nc": nc,
        "source_key_count": len(source_state),
        "target_key_count": len(target_state),
        "loaded_key_count": len(loaded_keys),
        "loaded_parameter_count": int(sum(transferred[key].numel() for key in transferred)),
        "skipped_key_count": len(set(source_state) - set(transferred)),
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatch),
        "detect_head_rebuilt": True,
        "detect_head_index": detect_index,
        "detect_head_rebuilt_key_count": len(rebuilt_detect_keys),
        "detect_scale_count": detect_scales,
        "forward_sanity_input_shape": [1, 3, 640, 640],
        "forward_sanity_output_shapes": output_shapes,
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_source_keys": unexpected,
        "excluded_anchor_keys": excluded,
        "shape_mismatches": shape_mismatch,
        "detect_head_rebuilt_keys": rebuilt_detect_keys,
        "status": "PASS",
    }
    return report, measure_complexity(target_model, 640)


def collect_source_snapshot() -> dict[str, Any]:
    roots = [
        PROJECT / "00_preflight_freeze",
        PROJECT / "01_data_split",
        PROJECT / "01A_protocol_amendment",
    ]
    files: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files[str(path)] = {"size": path.stat().st_size, "sha256": sha256(path)}
    for path in (TRAIN_SCRIPT, MODEL_YAML, SOURCE_WEIGHTS, HYP_YAML):
        files[str(path)] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return {"created_utc": utc_now(), "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Create the Stage02 audited run directory")
    args = parser.parse_args()
    if not args.prepare:
        parser.error("--prepare is required")

    if APPROVAL_TOKEN.read_text(encoding="utf-8").strip() != "APPROVE_AMENDMENT_A1_START_STAGE02":
        raise RuntimeError("Stage02 approval token is absent or incorrect")
    amendment = json.loads(AMENDMENT_STATE.read_text(encoding="utf-8"))
    amendment_validation = json.loads(AMENDMENT_VALIDATION.read_text(encoding="utf-8"))
    if amendment.get("stage01a_protocol_amendment") != "PASS":
        raise RuntimeError("Stage01A protocol amendment is not PASS")
    amendment_validation_status = amendment_validation.get("overall_status") or amendment_validation.get("status")
    if amendment_validation_status not in {"PASS", "STAGE01A_VALIDATION_PASS"}:
        raise RuntimeError(f"Stage01A validation is not PASS: {amendment_validation_status}")
    if not amendment.get("official_val_locked"):
        raise RuntimeError("Official val548 lock is not active")

    for path, expected in EXPECTED_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {path} expected={expected} actual={actual}")

    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    names = data.get("names", {})
    ordered_names = [names[index] if index in names else names[str(index)] for index in range(10)]
    if data.get("nc") != 10 or ordered_names != EXPECTED_CLASSES:
        raise RuntimeError(f"Dataset class contract mismatch: nc={data.get('nc')} names={ordered_names}")
    if data.get("official_val_used") is not False:
        raise RuntimeError("Internal data YAML does not explicitly deny official-val use")

    split_audit = {
        "meta_train": validate_split_list(TRAIN_LIST, 5177),
        "meta_inner_val": validate_split_list(INNER_VAL_LIST, 1294),
    }
    train_set = set(read_nonempty_lines(TRAIN_LIST))
    inner_set = set(read_nonempty_lines(INNER_VAL_LIST))
    official_set = set(read_nonempty_lines(OFFICIAL_LOCKED_LIST))
    if train_set & inner_set or train_set & official_set or inner_set & official_set:
        raise RuntimeError("Split overlap detected during Stage02 preflight")

    prohibited = active_prohibited_processes()
    if prohibited:
        raise RuntimeError(f"Active prohibited worker(s) detected: {prohibited}")
    if (RUN_DIR / ".complete").exists():
        raise RuntimeError(f"Stage02 is already complete: {RUN_DIR}")
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise RuntimeError(f"Stage02 run directory is not empty and cannot be silently overwritten: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    transfer_report, initial_complexity = build_weight_transfer_report()
    shutil.copy2(MODEL_YAML, RUN_DIR / "model.yaml")
    shutil.copy2(HYP_YAML, RUN_DIR / "hyp.yaml")
    write_json(RUN_DIR / "weight_transfer_report.json", transfer_report)
    write_json(RUN_DIR / "model_complexity_initialization.json", initial_complexity)

    source_snapshot = collect_source_snapshot()
    write_json(RUN_DIR / "source_snapshot_before_stage02.json", source_snapshot)

    command = [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--img",
        "640",
        "--batch-size",
        "16",
        "--epochs",
        "300",
        "--data",
        str(DATA_YAML),
        "--weights",
        str(SOURCE_WEIGHTS),
        "--cfg",
        str(MODEL_YAML),
        "--hyp",
        str(HYP_YAML),
        "--optimizer",
        "SGD",
        "--seed",
        "42",
        "--max-det",
        "500",
        "--val-iou-thres",
        "0.60",
        "--device",
        "0",
        "--workers",
        "4",
        "--patience",
        "300",
        "--save-period",
        "25",
        "--noplots",
        "--project",
        str(STAGE_ROOT),
        "--name",
        RUN_DIR.name,
        "--exist-ok",
    ]
    command_text = subprocess.list2cmdline(command)
    (RUN_DIR / "train_command.txt").write_text(command_text + "\n", encoding="utf-8")
    (RUN_DIR / "stdout.log").touch()
    (RUN_DIR / "stderr.log").touch()

    disk = shutil.disk_usage(PROJECT)
    preflight = {
        "created_utc": utc_now(),
        "approval_token": "APPROVE_AMENDMENT_A1_START_STAGE02",
        "stage00_preserved": True,
        "stage01_preserved": True,
        "stage01a_protocol_amendment": "PASS",
        "stage01a_validation": "PASS",
        "official_val_locked": True,
        "official_val_first_evaluation_stage": "Stage11",
        "split_audit": split_audit,
        "split_intersections": {
            "train_inner_val": 0,
            "train_official_val": 0,
            "inner_val_official_val": 0,
        },
        "nc": 10,
        "class_order": EXPECTED_CLASSES,
        "active_prohibited_workers": [],
        "disk": {
            "path": str(PROJECT.anchor),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_gib": disk.free / (1024**3),
            "action": "RECORDED_ONLY_USER_WILL_RESOLVE_DISK_SPACE",
        },
        "gpu": nvidia_snapshot(),
        "python": {
            "executable": str(PYTHON),
            "version": platform.python_version(),
        },
        "training_protocol": {
            "model": "Lite-v3-SearchParent",
            "train_count": 5177,
            "validation_count": 1294,
            "epochs": 300,
            "batch_size": 16,
            "imgsz": 640,
            "seed": 42,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "amp": True,
            "ema": True,
            "autoanchor": True,
            "multi_scale": False,
            "tta": False,
            "early_stopping_effectively_disabled": "patience=300 for a 300-epoch run",
            "best_pt_selection": "meta-inner-val1294 YOLOv5 fitness",
            "validation_max_det": 500,
            "validation_iou_threshold": 0.60,
        },
        "prohibited_work": {
            "yolov5m_training": False,
            "kd": False,
            "taylor": False,
            "pruning": False,
            "nsga2": False,
            "ap_only_search": False,
            "random_search": False,
            "official_val548_access": False,
        },
        "command": command,
        "command_text": command_text,
        "status": "PASS_READY_TO_START_TRAINING",
    }
    write_json(RUN_DIR / "stage02_preflight.json", preflight)
    write_json(
        RUN_DIR / "run_manifest.json",
        {
            "stage": "Stage02 SearchParent Training Only",
            "model": "Lite-v3-SearchParent",
            "created_utc": utc_now(),
            "run_dir": str(RUN_DIR),
            "command": command,
            "source_checkpoint": str(SOURCE_WEIGHTS),
            "source_checkpoint_sha256": sha256(SOURCE_WEIGHTS),
            "model_yaml": str(MODEL_YAML),
            "model_yaml_sha256": sha256(MODEL_YAML),
            "hyp_yaml": str(HYP_YAML),
            "hyp_yaml_sha256": sha256(HYP_YAML),
            "data_yaml": str(DATA_YAML),
            "data_yaml_sha256": sha256(DATA_YAML),
            "official_val_locked": True,
            "status": "PREPARED_NOT_STARTED",
        },
    )
    write_json(
        RUN_DIR / "stage02_state.json",
        {
            "stage": "stage02",
            "status": "prepared_waiting_for_training_process_start",
            "timestamp": utc_now(),
            "stage02_started": False,
            "training_executed": False,
            "evaluation_executed": False,
            "official_val_accessed": False,
            "taylor_executed": False,
            "pruning_executed": False,
            "search_executed": False,
        },
    )
    (PROJECT / "manifests" / "current_stage.txt").write_text("stage02_prepared\n", encoding="utf-8")
    (PROJECT / "manifests" / "current_run.txt").write_text(str(RUN_DIR) + "\n", encoding="utf-8")

    print("STAGE02_PREFLIGHT=PASS")
    print(f"STAGE02_RUN_DIR={RUN_DIR}")
    print(f"STAGE02_TRAIN_COMMAND={command_text}")
    print(f"STAGE02_DISK_FREE_GIB={preflight['disk']['free_gib']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
