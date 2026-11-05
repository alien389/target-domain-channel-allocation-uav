#!/usr/bin/env python3
"""Run the single permitted Stage02 FP32 evaluation on inner-val1294."""

from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
YOLO_ROOT = Path(_release_paths.expand('${YOLOV5_ROOT}'))
RUN_DIR = PROJECT / "02_search_parent_teacher" / "lite_v3_search_parent"
DATA_YAML = PROJECT / "01_data_split" / "data_yamls" / "visdrone_meta_train5177_inner1294.yaml"
INNER_VAL_LIST = PROJECT / "01_data_split" / "meta_inner_val1294_seed42.txt"
RAW_TRAIN_ANNOTATIONS = Path(_release_paths.expand('${DATASETS_ROOT}\\VisDrone2019-YOLO\\VisDrone2019-DET-train\\annotations'))
RAW_TRAIN_IMAGES = Path(_release_paths.expand('${DATASETS_ROOT}\\VisDrone2019-YOLO\\VisDrone2019-DET-train\\images'))
TOOLKIT = Path(_release_paths.expand('${PAPER1_ROOT}\\deployment_drone_laptop_raspberry\\00_external_official_drone\\VisDrone2018-DET-toolkit'))
TOOLKIT_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
MATLAB = Path(_release_paths.expand('${DRIVE_C_ROOT}Program Files\\MATLAB\\R2025a\\bin\\matlab.exe'))
MATLAB_WRAPPER = PROJECT / "scripts" / "run_visdrone_internal_eval_stage02.m"
EXPECTED_IMAGE_COUNT = 1294
MAX_DET = 500


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


def read_image_list() -> list[Path]:
    paths = [
        Path(line.strip())
        for line in INNER_VAL_LIST.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(paths) != EXPECTED_IMAGE_COUNT or len(set(paths)) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(f"Inner-val list contract failed: rows={len(paths)}, unique={len(set(paths))}")
    for path in paths:
        normalized = str(path).replace("/", "\\").lower()
        if (
            not path.is_file()
            or "\\images\\train\\" not in normalized
            or "\\images\\val\\" in normalized
            or "visdrone2019-det-val" in normalized
            or "test-dev" in normalized
        ):
            raise RuntimeError(f"Forbidden or missing Stage02 image path: {path}")
    return paths


def yolo_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    index = lowered.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def recursive_shapes(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [recursive_shapes(item) for item in value]
    if isinstance(value, dict):
        return {str(key): recursive_shapes(item) for key, item in value.items()}
    return type(value).__name__


def load_checkpoint_model(path: Path) -> Any:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise RuntimeError(f"Checkpoint contains neither ema nor model: {path}")
    return model.float().eval()


def checkpoint_sanity(path: Path) -> dict[str, Any]:
    model = load_checkpoint_model(path)
    nonfinite_tensors: list[str] = []
    for key, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            nonfinite_tensors.append(key)
    if nonfinite_tensors:
        raise RuntimeError(f"Non-finite tensors in {path.name}: {nonfinite_tensors[:10]}")
    detect = model.model[-1]
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 640, 640), dtype=torch.float32))
    result = {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "nc": int(getattr(detect, "nc", -1)),
        "detect_scale_count": len(getattr(detect, "m", [])),
        "input_shape": [1, 3, 640, 640],
        "output_shapes": recursive_shapes(output),
        "nonfinite_tensor_count": 0,
        "status": "PASS",
    }
    if result["nc"] != 10 or result["detect_scale_count"] != 3:
        raise RuntimeError(f"Detect contract failed for {path}: {result}")
    return result


def measure_model_complexity(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    from thop import profile

    def graph_stats(instance: Any) -> dict[str, Any]:
        params = sum(parameter.numel() for parameter in instance.parameters())
        stride = max(int(instance.stride.max()), 32)
        sample = torch.zeros((1, 3, stride, stride), dtype=torch.float32)
        macs = profile(copy.deepcopy(instance).cpu(), inputs=(sample,), verbose=False)[0]
        scale = (640 / stride) ** 2
        macs_g = float(macs * scale / 1e9)
        return {
            "params": int(params),
            "params_m": float(params / 1e6),
            "macs_g": macs_g,
            "flops_g_thop_macs_x2": float(2.0 * macs_g),
        }

    native = graph_stats(model)
    fused = graph_stats(copy.deepcopy(model).cpu().eval().fuse())
    return {
        "created_utc": utc_now(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "weights_size_bytes": checkpoint_path.stat().st_size,
        "weights_size_mib": checkpoint_path.stat().st_size / (1024**2),
        "input_shape": [1, 3, 640, 640],
        "nc": 10,
        "protocol": "THOP MACs x 2 at batch=1 and 640x640",
        "unfused_diagnostic": native,
        "fused_inference_graph": fused,
    }


def sanitize_prediction(
    raw_line: str,
    image_width: int,
    image_height: int,
    source: Path,
    line_number: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parts = raw_line.strip().split()
    audit = {
        "source_file": str(source),
        "source_line": line_number,
        "raw": raw_line,
        "reasons": [],
    }
    if len(parts) != 6:
        audit["reasons"].append("malformed_field_count")
        return None, audit
    try:
        class_value, xc, yc, width, height, score = map(float, parts)
    except ValueError:
        audit["reasons"].append("parse_error")
        return None, audit
    values = (class_value, xc, yc, width, height, score)
    if not all(np.isfinite(value) for value in values):
        audit["reasons"].append("nonfinite")
    class_id = int(class_value) if np.isfinite(class_value) else -1
    if class_id < 0 or class_id > 9:
        audit["reasons"].append("invalid_class")
    if width <= 0 or height <= 0:
        audit["reasons"].append("nonpositive_geometry")
    x1 = max(0.0, min(float(image_width), (xc - width / 2.0) * image_width))
    y1 = max(0.0, min(float(image_height), (yc - height / 2.0) * image_height))
    x2 = max(0.0, min(float(image_width), (xc + width / 2.0) * image_width))
    y2 = max(0.0, min(float(image_height), (yc + height / 2.0) * image_height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        audit["reasons"].append("nonpositive_clipped_geometry")
    if audit["reasons"]:
        return None, audit
    return {
        "bbox": [x1, y1, clipped_width, clipped_height],
        "score": score,
        "category_id": class_id + 1,
    }, None


def export_official_predictions(
    image_paths: list[Path],
    val_labels_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dropped: list[dict[str, Any]] = []
    predictions_by_stem: dict[str, list[dict[str, Any]]] = {}
    total = 0
    max_rows = 0
    for image_path in sorted(image_paths, key=lambda item: item.name):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read inner-val image: {image_path}")
        height, width = image.shape[:2]
        source = val_labels_dir / f"{image_path.stem}.txt"
        parsed: list[dict[str, Any]] = []
        if source.is_file():
            for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                row, drop = sanitize_prediction(raw, width, height, source, line_number)
                if row is not None:
                    parsed.append(row)
                if drop is not None:
                    dropped.append(drop)
        parsed.sort(key=lambda item: item["score"], reverse=True)
        parsed = parsed[:MAX_DET]
        predictions_by_stem[image_path.stem] = parsed
        lines = [
            (
                f"{row['bbox'][0]:.10f},{row['bbox'][1]:.10f},"
                f"{row['bbox'][2]:.10f},{row['bbox'][3]:.10f},"
                f"{row['score']:.10f},{row['category_id']},-1,-1"
            )
            for row in parsed
        ]
        (output_dir / f"{image_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        total += len(parsed)
        max_rows = max(max_rows, len(parsed))
    generated = list(output_dir.glob("*.txt"))
    if len(generated) != EXPECTED_IMAGE_COUNT or max_rows > MAX_DET:
        raise RuntimeError(
            f"Official export failed: files={len(generated)}, expected={EXPECTED_IMAGE_COUNT}, max_rows={max_rows}"
        )
    audit = {
        "created_utc": utc_now(),
        "image_count": len(generated),
        "prediction_count": total,
        "max_rows_per_image": max_rows,
        "max_det": MAX_DET,
        "dropped_prediction_count": len(dropped),
        "dropped_predictions": dropped,
        "official_val548_used": False,
        "testdev_used": False,
        "status": "PASS",
    }
    return audit, predictions_by_stem


def prepare_raw_annotation_subset(image_paths: list[Path], destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"Annotation subset destination must be empty: {destination}")
    copied: list[dict[str, Any]] = []
    for image_path in sorted(image_paths, key=lambda item: item.name):
        source_annotation = RAW_TRAIN_ANNOTATIONS / f"{image_path.stem}.txt"
        raw_image = RAW_TRAIN_IMAGES / image_path.name
        if not source_annotation.is_file() or not raw_image.is_file():
            raise RuntimeError(f"Original train evidence missing for {image_path.name}")
        yolo_image = cv2.imread(str(image_path))
        source_image = cv2.imread(str(raw_image))
        if yolo_image is None or source_image is None or yolo_image.shape != source_image.shape:
            raise RuntimeError(f"Image identity/dimension mismatch for {image_path.name}")
        destination_file = destination / source_annotation.name
        shutil.copy2(source_annotation, destination_file)
        copied.append(
            {
                "image": image_path.name,
                "source_annotation": str(source_annotation),
                "source_annotation_sha256": sha256(source_annotation),
                "copied_annotation": str(destination_file),
                "copied_annotation_sha256": sha256(destination_file),
                "image_shape": list(yolo_image.shape),
            }
        )
    if len(copied) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(f"Raw annotation subset count mismatch: {len(copied)}")
    return {
        "created_utc": utc_now(),
        "source_annotation_root": str(RAW_TRAIN_ANNOTATIONS),
        "source_image_root": str(RAW_TRAIN_IMAGES),
        "copied_count": len(copied),
        "source_files_modified": 0,
        "records": copied,
        "status": "PASS",
    }


def matlab_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def run_matlab(
    annotation_subset: Path,
    prediction_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path in (MATLAB, MATLAB_WRAPPER, TOOLKIT / "evalDET.m"):
        if not path.exists():
            raise FileNotFoundError(f"Required evaluator component is missing: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command_expression = (
        f"addpath('{matlab_quote(MATLAB_WRAPPER.parent)}'); "
        "run_visdrone_internal_eval_stage02("
        f"'{matlab_quote(TOOLKIT)}',"
        f"'{matlab_quote(annotation_subset)}',"
        f"'{matlab_quote(RAW_TRAIN_IMAGES)}',"
        f"'{matlab_quote(prediction_dir)}',"
        f"'{matlab_quote(output_dir)}',"
        f"{EXPECTED_IMAGE_COUNT},"
        f"'{TOOLKIT_COMMIT}');"
    )
    completed = subprocess.run(
        [str(MATLAB), "-batch", command_expression],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path = output_dir / "matlab_console_stage02.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MATLAB internal evaluation failed ({completed.returncode}); see {log_path}")
    metrics_path = output_dir / "internal_visdrone_metrics_stage02.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"MATLAB did not generate metrics JSON: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["matlab_console_log"] = str(log_path)
    payload["matlab_wrapper_sha256"] = sha256(MATLAB_WRAPPER)
    return payload


def build_supplemental_coco_inputs(
    image_paths: list[Path],
    predictions_by_stem: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> tuple[Path, Path, list[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    image_ids: list[int] = []
    annotation_id = 1
    class_names = [
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
    for image_id, image_path in enumerate(sorted(image_paths, key=lambda item: item.name), 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image for supplemental COCO data: {image_path}")
        height, width = image.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(width),
                "height": int(height),
            }
        )
        image_ids.append(image_id)
        label_path = yolo_label_path(image_path)
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) != 5:
                raise RuntimeError(f"Malformed YOLO ground-truth row: {label_path}: {raw}")
            class_id = int(float(parts[0]))
            xc, yc, box_width, box_height = map(float, parts[1:])
            x = (xc - box_width / 2.0) * width
            y = (yc - box_height / 2.0) * height
            box_width_pixels = box_width * width
            box_height_pixels = box_height * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, box_width_pixels, box_height_pixels],
                    "area": box_width_pixels * box_height_pixels,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        for row in predictions_by_stem[image_path.stem]:
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": row["category_id"],
                    "bbox": row["bbox"],
                    "score": row["score"],
                }
            )
    ground_truth = {
        "info": {"description": "Stage02 inner-val1294 supplemental area-metric annotations"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(class_names)],
    }
    annotation_path = output_dir / "inner_val1294_coco_annotations.json"
    prediction_path = output_dir / "inner_val1294_coco_predictions.json"
    write_json(annotation_path, ground_truth)
    write_json(prediction_path, predictions)
    return annotation_path, prediction_path, image_ids


def supplemental_coco_metrics(
    annotation_path: Path,
    prediction_path: Path,
    image_ids: list[int],
) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(str(prediction_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = list(range(1, 11))
    evaluator.params.maxDets = [1, 10, 500]
    evaluator.params.useCats = 1
    evaluator.evaluate()
    evaluator.accumulate()
    max_det_index = evaluator.params.maxDets.index(500)

    def precision_average(area_label: str, iou: float | None = None) -> float:
        area_index = evaluator.params.areaRngLbl.index(area_label)
        values = evaluator.eval["precision"][:, :, :, area_index, max_det_index]
        if iou is not None:
            threshold_index = int(np.where(np.isclose(evaluator.params.iouThrs, iou))[0][0])
            values = values[threshold_index : threshold_index + 1]
        valid = values[values > -1]
        if valid.size == 0:
            raise RuntimeError(f"No valid COCO precision values for area={area_label}, iou={iou}")
        return float(valid.mean())

    return {
        "AP_COCO_style": precision_average("all"),
        "AP50_COCO_style": precision_average("all", 0.50),
        "AP75_COCO_style": precision_average("all", 0.75),
        "APS_COCO_style_supplemental": precision_average("small"),
        "APM_COCO_style_supplemental": precision_average("medium"),
        "APL_COCO_style_supplemental": precision_average("large"),
    }


def training_best_epoch(results_csv: Path) -> dict[str, Any]:
    frame = pd.read_csv(results_csv)
    frame.columns = [column.strip() for column in frame.columns]
    if len(frame) != 300:
        raise RuntimeError(f"Expected 300 training rows, found {len(frame)}")
    required = [
        "epoch",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"results.csv missing columns: {missing}")
    for column in required:
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise RuntimeError(f"results.csv has non-finite values in {column}")
    fitness = 0.1 * frame["metrics/mAP_0.5"] + 0.9 * frame["metrics/mAP_0.5:0.95"]
    index = int(fitness.idxmax())
    row = frame.loc[index]
    return {
        "best_epoch_zero_based": int(row["epoch"]),
        "best_epoch_one_based": int(row["epoch"]) + 1,
        "best_fitness": float(fitness.loc[index]),
        "precision": float(row["metrics/precision"]),
        "recall": float(row["metrics/recall"]),
        "map50": float(row["metrics/mAP_0.5"]),
        "map50_95": float(row["metrics/mAP_0.5:0.95"]),
        "training_rows": len(frame),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir != RUN_DIR.resolve():
        raise RuntimeError(f"Stage02 evaluation is locked to {RUN_DIR}")

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    for path in (best, last, results_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Required completed-training artifact is missing: {path}")

    image_paths = read_image_list()
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    if data.get("nc") != 10 or data.get("official_val_used") is not False:
        raise RuntimeError("Stage02 data YAML contract failed")

    validation_root = run_dir / "explicit_inner_val_predictions"
    if validation_root.exists():
        raise RuntimeError(f"Explicit evaluation directory already exists; refusing silent overwrite: {validation_root}")
    validation_root.mkdir(parents=True)
    validation_run = validation_root / "yolov5_fp32"

    if str(YOLO_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLO_ROOT))
    import val

    validation_parameters = {
        "data": str(DATA_YAML),
        "weights": str(best),
        "batch_size": 16,
        "imgsz": 640,
        "conf_thres": 0.001,
        "iou_thres": 0.60,
        "max_det": MAX_DET,
        "task": "val",
        "device": "0",
        "workers": 4,
        "single_cls": False,
        "augment": False,
        "verbose": True,
        "save_txt": True,
        "save_hybrid": False,
        "save_conf": True,
        "save_json": False,
        "project": validation_root,
        "name": validation_run.name,
        "exist_ok": True,
        "half": False,
        "dnn": False,
        "plots": False,
    }
    (run_dir / "explicit_validation_command.txt").write_text(
        "val.run(" + json.dumps(validation_parameters, ensure_ascii=False, default=str) + ")\n",
        encoding="utf-8",
    )
    results, class_maps, timing = val.run(**validation_parameters)
    precision, recall, map50, map50_95 = [float(value) for value in results[:4]]
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if not all(np.isfinite(value) for value in (precision, recall, map50, map50_95, f1)):
        raise RuntimeError("YOLOv5 explicit evaluation returned non-finite metrics")

    labels_dir = validation_run / "labels"
    if not labels_dir.is_dir():
        raise RuntimeError(f"Validation prediction labels were not created: {labels_dir}")
    official_prediction_dir = validation_root / "visdrone_predictions"
    prediction_audit, predictions_by_stem = export_official_predictions(
        image_paths, labels_dir, official_prediction_dir
    )
    write_json(validation_root / "prediction_export_audit.json", prediction_audit)

    annotation_subset = validation_root / "visdrone_train_annotation_subset"
    annotation_audit = prepare_raw_annotation_subset(image_paths, annotation_subset)
    write_json(validation_root / "annotation_subset_audit.json", annotation_audit)

    matlab_payload = run_matlab(
        annotation_subset,
        official_prediction_dir,
        validation_root / "visdrone_toolkit_metrics",
    )
    official_percent = matlab_payload["visdrone_metrics"]
    if matlab_payload["metadata"].get("evaluated_images") != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(f"MATLAB evaluated-image mismatch: {matlab_payload['metadata']}")

    annotation_json, prediction_json, image_ids = build_supplemental_coco_inputs(
        image_paths,
        predictions_by_stem,
        validation_root / "supplemental_coco_area_metrics",
    )
    area_metrics = supplemental_coco_metrics(annotation_json, prediction_json, image_ids)
    write_json(
        validation_root / "supplemental_coco_area_metrics" / "metrics.json",
        {
            "created_utc": utc_now(),
            "namespace": "COCO_STYLE_SUPPLEMENTAL_AREA_METRICS",
            "image_count": len(image_ids),
            "max_det": MAX_DET,
            "metrics": area_metrics,
            "official_val548_used": False,
        },
    )

    best_epoch = training_best_epoch(results_csv)
    comparison = {
        "training_results_at_best_epoch": best_epoch,
        "explicit_yolov5_metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
        },
        "differences": {
            "precision": precision - best_epoch["precision"],
            "recall": recall - best_epoch["recall"],
            "map50": map50 - best_epoch["map50"],
            "map50_95": map50_95 - best_epoch["map50_95"],
        },
        "explanation": (
            "The YOLOv5 namespace is scientifically comparable but not expected to be bitwise identical: "
            "in-epoch validation used the EMA model with half=AMP and batch_size=32, whereas the required "
            "explicit audit used FP32 and batch_size=16. Official VisDrone AP additionally suppresses "
            "detections in ignored regions and is retained separately."
        ),
        "training_validation_precision_mode": "FP16 through half=AMP",
        "training_validation_batch_size": 32,
        "explicit_validation_precision_mode": "FP32",
        "explicit_validation_batch_size": 16,
        "documented_protocol_delta_tolerance": 0.005,
    }
    comparison["direct_yolov5_match_within_1e-6"] = all(
        abs(value) <= 1e-6 for value in comparison["differences"].values()
    )
    comparison["direct_yolov5_difference_explained_within_protocol_tolerance"] = all(
        abs(value) <= comparison["documented_protocol_delta_tolerance"]
        for value in comparison["differences"].values()
    )
    write_json(run_dir / "protocol_equivalence_audit.json", comparison)

    best_sanity = checkpoint_sanity(best)
    last_sanity = checkpoint_sanity(last)
    model = load_checkpoint_model(best)
    complexity = measure_model_complexity(model, best)
    complexity["best_checkpoint_sanity"] = best_sanity
    complexity["last_checkpoint_sanity"] = last_sanity
    write_json(run_dir / "model_complexity.json", complexity)

    official_fraction = {key: float(value) / 100.0 for key, value in official_percent.items()}
    final_payload = {
        "created_utc": utc_now(),
        "stage": "Stage02 SearchParent explicit inner-val evaluation",
        "model": "Lite-v3-SearchParent",
        "checkpoint": str(best),
        "checkpoint_sha256": sha256(best),
        "dataset": "VisDrone2019-DET meta-inner-val1294",
        "data_yaml": str(DATA_YAML),
        "data_yaml_sha256": sha256(DATA_YAML),
        "image_count": EXPECTED_IMAGE_COUNT,
        "input_size": 640,
        "batch_size": 16,
        "confidence_threshold": 0.001,
        "nms_iou_threshold": 0.60,
        "max_det": MAX_DET,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "AP_0.5_0.95": official_fraction["AP"],
        "AP50": official_fraction["AP50"],
        "AP75": official_fraction["AP75"],
        "APS": area_metrics["APS_COCO_style_supplemental"],
        "APM": area_metrics["APM_COCO_style_supplemental"],
        "APL": area_metrics["APL_COCO_style_supplemental"],
        "primary_ap_namespace": (
            "Official VisDrone DET toolkit on inner-val1294 with ignored-region suppression"
        ),
        "area_metric_namespace": "COCO-style supplemental area metrics on the same 1294 images",
        "yolov5_metrics_for_results_csv_equivalence": {
            "precision": precision,
            "recall": recall,
            "AP50": map50,
            "AP_0.5_0.95": map50_95,
            "class_maps": [float(value) for value in np.asarray(class_maps).tolist()],
        },
        "official_visdrone_metrics_fraction": official_fraction,
        "official_visdrone_metrics_percentage": official_percent,
        "supplemental_coco_area_metrics": area_metrics,
        "timing_seconds": [float(value) for value in timing],
        "best_epoch": best_epoch,
        "params": complexity["fused_inference_graph"]["params"],
        "flops_g": complexity["fused_inference_graph"]["flops_g_thop_macs_x2"],
        "weights_size_mib": complexity["weights_size_mib"],
        "official_val548_accessed": False,
        "testdev_accessed": False,
        "augment": False,
        "tta": False,
        "single_scale": True,
        "precision_mode": "FP32",
        "status": "PASS",
    }
    write_json(run_dir / "explicit_inner_val_metrics.json", final_payload)
    print("STAGE02_EXPLICIT_EVALUATION=PASS")
    print(f"STAGE02_EXPLICIT_IMAGES={EXPECTED_IMAGE_COUNT}")
    print(f"STAGE02_AP={final_payload['AP_0.5_0.95']:.12f}")
    print(f"STAGE02_APS={final_payload['APS']:.12f}")
    print(f"STAGE02_CHECKPOINT_SHA256={final_payload['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
