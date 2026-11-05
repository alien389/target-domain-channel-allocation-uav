from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(_release_paths.expand('${STUDY_ROOT}'))
OUT = ROOT / "11_official_val_statistics" / "01_val548_evaluation"
YOLO_ROOT = Path(_release_paths.expand('${YOLOV5_ROOT}'))
DATA_YAML = ROOT / "01_data_split" / "data_yamls" / "visdrone_full_train6471_official_val548.yaml"
VAL_LIST = ROOT / "01_data_split" / "official_val548_locked.txt"
RAW_VAL_IMAGES = Path(_release_paths.expand('${DATASETS_ROOT}\\VisDrone2019-YOLO\\VisDrone2019-DET-val\\images'))
RAW_VAL_ANNOTATIONS = Path(_release_paths.expand('${DATASETS_ROOT}\\VisDrone2019-YOLO\\VisDrone2019-DET-val\\annotations'))
SOURCE = ROOT / "scripts" / "stage02_explicit_inner_val_eval.py"
PORT = ROOT / "scripts" / "visdrone_toolkit_python_port.py"
EXPECTED = 548


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen evaluator source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slug(model: str) -> str:
    return {
        "Meta-AP+APS-B6": "meta_ap_aps_b6",
        "SourceLocked-Stage6-B6": "source_locked_stage6_b6",
        "MatchedManual-B6": "matched_manual_b6",
        "Meta-AP-only-B6": "meta_ap_only_b6",
        "RandomSearch-B6": "random_search_b6",
    }[model]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    campaign = json.loads((OUT / "checkpoint_evaluation_manifest.json").read_text(encoding="utf-8"))
    records = [item for item in campaign["records"] if item["run_id"] == args.run_id]
    if len(records) != 1:
        raise RuntimeError(f"Unknown or duplicate run ID: {args.run_id}")
    record = records[0]
    model_slug = slug(record["model"])
    prediction_root = OUT / "predictions" / model_slug / f"seed{record['seed']}"
    raw_root = OUT / "raw_evaluator_outputs" / model_slug / f"seed{record['seed']}"
    for path in (prediction_root, raw_root):
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Refusing nonempty incomplete evaluation output for {args.run_id}: {path}")
        path.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(record["checkpoint_path"])
    if sha256(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch immediately before inference: {args.run_id}")
    image_paths = [Path(line.strip()) for line in VAL_LIST.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(image_paths) != EXPECTED or len(set(image_paths)) != EXPECTED:
        raise RuntimeError("val548 list contract failed")

    source = load_module("stage11_frozen_explicit_source", SOURCE)
    source.EXPECTED_IMAGE_COUNT = EXPECTED
    source.MAX_DET = 500
    source.RAW_TRAIN_IMAGES = RAW_VAL_IMAGES
    source.RAW_TRAIN_ANNOTATIONS = RAW_VAL_ANNOTATIONS
    port = load_module("stage11_frozen_visdrone_port", PORT)

    if str(YOLO_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLO_ROOT))
    import val

    validation_parameters = {
        "data": str(DATA_YAML), "weights": str(checkpoint), "batch_size": 16, "imgsz": 640,
        "conf_thres": 0.001, "iou_thres": 0.60, "max_det": 500, "task": "val",
        "device": "0", "workers": 4, "single_cls": False, "augment": False, "verbose": True,
        "save_txt": True, "save_hybrid": False, "save_conf": True, "save_json": False,
        "project": prediction_root, "name": "yolov5_fp32", "exist_ok": True,
        "half": False, "dnn": False, "plots": False,
    }
    write_json(raw_root / "evaluation_command.json", validation_parameters)
    start = utc_now()
    results, class_maps, timing = val.run(**validation_parameters)
    precision, recall, map50_yolo, map50_95_yolo = [float(value) for value in results[:4]]
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if not all(math.isfinite(value) for value in (precision, recall, f1, map50_yolo, map50_95_yolo)):
        raise RuntimeError("Non-finite base evaluator metrics")

    labels_dir = prediction_root / "yolov5_fp32" / "labels"
    official_dir = prediction_root / "visdrone_txt"
    prediction_audit, predictions_by_stem = source.export_official_predictions(image_paths, labels_dir, official_dir)
    write_json(raw_root / "prediction_export_audit.json", prediction_audit)
    official_payload = port.evaluate(RAW_VAL_ANNOTATIONS, official_dir, RAW_VAL_IMAGES, raw_root / "visdrone_metrics.json", EXPECTED)
    official_percentage = official_payload["visdrone_metrics"]
    official_fraction = {key: float(value) / 100.0 for key, value in official_percentage.items()}

    coco_dir = prediction_root / "coco_records"
    annotation_json, prediction_json, image_ids = source.build_supplemental_coco_inputs(image_paths, predictions_by_stem, coco_dir)
    area = source.supplemental_coco_metrics(annotation_json, prediction_json, image_ids)
    write_json(coco_dir / "metrics.json", area)

    val_manifest = json.loads((OUT / "val548_manifest.json").read_text(encoding="utf-8"))
    val_by_name = {item["image_name"]: item for item in val_manifest["records"]}
    prediction_records: list[dict[str, Any]] = []
    for image_path in image_paths:
        yolo_path = labels_dir / f"{image_path.stem}.txt"
        official_path = official_dir / f"{image_path.stem}.txt"
        rows = predictions_by_stem[image_path.stem]
        prediction_records.append({
            "image_name": image_path.name,
            "image_sha256": val_by_name[image_path.name]["image_sha256"],
            "ground_truth_annotation_path": val_by_name[image_path.name]["raw_annotation_path"],
            "ground_truth_annotation_sha256": val_by_name[image_path.name]["raw_annotation_sha256"],
            "yolov5_prediction_path": str(yolo_path) if yolo_path.is_file() else None,
            "yolov5_prediction_sha256": sha256(yolo_path) if yolo_path.is_file() else None,
            "visdrone_prediction_path": str(official_path),
            "visdrone_prediction_sha256": sha256(official_path),
            "prediction_count": len(rows),
            "preprocessing": {"imgsz": 640, "single_scale": True, "letterbox_stride": 32, "augment": False, "tta": False},
        })
    prediction_manifest = {
        "schema": "stage11_image_prediction_manifest_v1", "created_utc": utc_now(),
        "run_id": record["run_id"], "model": record["model"], "seed": record["seed"],
        "checkpoint_sha256": record["checkpoint_sha256"], "image_count": EXPECTED,
        "prediction_score_stage": "post_NMS_pre_metric_aggregation", "records": prediction_records,
        "coco_prediction_json": str(prediction_json), "coco_prediction_json_sha256": sha256(prediction_json),
        "status": "PASS", "heldout1610_accessed": False,
    }
    write_json(prediction_root / "prediction_manifest.json", prediction_manifest)

    metrics = {
        "schema": "stage11_per_checkpoint_metrics_v1", "created_utc": utc_now(), "status": "PASS",
        "run_id": record["run_id"], "model": record["model"], "seed": record["seed"],
        "architecture_signature": record["architecture_signature"],
        "terminal_checkpoint_path": str(checkpoint), "terminal_checkpoint_sha256": record["checkpoint_sha256"],
        "terminal_checkpoint_size_bytes": checkpoint.stat().st_size,
        "AP": official_fraction["AP"], "AP50": official_fraction["AP50"], "AP75": official_fraction["AP75"],
        "APS": area["APS_COCO_style_supplemental"], "APM": area["APM_COCO_style_supplemental"],
        "APL": area["APL_COCO_style_supplemental"], "Precision": precision, "Recall": recall, "F1": f1,
        "Params": int(record["params_exact"]), "FLOPs_raw": int(record["flops_raw_exact"]),
        "FLOPs_G": float(record["flops_raw_exact"]) / 1e9,
        "official_visdrone_metrics_fraction": official_fraction,
        "official_visdrone_metrics_percentage": official_percentage,
        "yolov5_base_metrics": {"AP50": map50_yolo, "AP": map50_95_yolo, "class_maps": [float(value) for value in np.asarray(class_maps).tolist()]},
        "timing_ms_per_image": [float(value) for value in timing], "evaluation_start_utc": start,
        "evaluation_end_utc": utc_now(), "precision_mode": "FP32", "imgsz": 640, "batch_size": 16,
        "conf_thres": 0.001, "nms_iou_thres": 0.60, "max_det": 500, "tta": False,
        "training_executed": False, "kd_executed": False, "teacher_loaded": False,
        "heldout1610_accessed": False,
    }
    write_json(raw_root / "metrics.json", metrics)
    write_json(raw_root / "evaluation_record.json", {
        "run_id": record["run_id"], "model": record["model"], "seed": record["seed"],
        "metrics_path": str(raw_root / "metrics.json"), "metrics_sha256": sha256(raw_root / "metrics.json"),
        "prediction_manifest_path": str(prediction_root / "prediction_manifest.json"),
        "prediction_manifest_sha256": sha256(prediction_root / "prediction_manifest.json"),
        "status": "COMPLETE", "heldout1610_accessed": False,
    })
    (raw_root / ".complete").write_text("STAGE11_CHECKPOINT_EVALUATION=PASS\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "run_id": record["run_id"], "model": record["model"], "seed": record["seed"], "AP": metrics["AP"], "APS": metrics["APS"], "Recall": metrics["Recall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
