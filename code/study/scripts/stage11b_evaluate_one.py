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

import numpy as np


ROOT = Path(_release_paths.expand('${STUDY_ROOT}'))
OUT = ROOT / "11b_heldout1610_confirmation"
YOLO_ROOT = Path(_release_paths.expand('${YOLOV5_ROOT}'))
DATA_YAML = OUT / "heldout1610_eval_data.yaml"
IMAGE_LIST = OUT / "heldout1610_ordered_images.txt"
RAW_IMAGES = Path(_release_paths.expand('${DATASETS_ROOT}\\VisDrone2019-YOLO\\VisDrone2019-DET-test-dev\\images'))
RAW_ANNOTATIONS = OUT / "evaluator_annotation_view"
SOURCE = ROOT / "scripts" / "stage02_explicit_inner_val_eval.py"
PORT = ROOT / "scripts" / "visdrone_toolkit_python_port.py"
EXPECTED = 1610


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
        "Meta-AP+APS-B6": "meta_ap_aps_b6", "SourceLocked-Stage6-B6": "source_locked_stage6_b6",
        "MatchedManual-B6": "matched_manual_b6", "Meta-AP-only-B6": "meta_ap_only_b6",
        "RandomSearch-B6": "random_search_b6",
    }[model]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    campaign = json.loads((OUT / "checkpoint_evaluation_manifest.json").read_text(encoding="utf-8-sig"))
    matches = [item for item in campaign["records"] if item["run_id"] == args.run_id]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown or duplicate run ID: {args.run_id}")
    record = matches[0]
    model_slug = slug(record["model"])
    prediction_root = OUT / "predictions" / model_slug / f"seed{record['seed']}"
    raw_root = OUT / "raw_evaluator_outputs" / model_slug / f"seed{record['seed']}"
    resume_postprocessing = (raw_root / "resume_postprocessing.json").is_file()
    for path in (prediction_root, raw_root):
        if path.exists() and any(path.iterdir()) and not resume_postprocessing:
            raise RuntimeError(f"Refusing nonempty incomplete Stage11B output: {path}")
        path.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(record["checkpoint_path"])
    if sha256(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch immediately before inference: {args.run_id}")
    image_paths = [Path(line.strip()) for line in IMAGE_LIST.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(image_paths) != EXPECTED or len(set(image_paths)) != EXPECTED:
        raise RuntimeError("heldout1610 image-list contract failed")

    source = load_module("stage11b_frozen_explicit_source", SOURCE)
    source.EXPECTED_IMAGE_COUNT = EXPECTED
    source.MAX_DET = 500
    source.RAW_TRAIN_IMAGES = RAW_IMAGES
    source.RAW_TRAIN_ANNOTATIONS = RAW_ANNOTATIONS
    port = load_module("stage11b_frozen_visdrone_port", PORT)
    if str(YOLO_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLO_ROOT))
    import val
    from utils.callbacks import Callbacks

    capture_dir = prediction_root / "full_precision_post_nms"
    parameters = {
        "data": str(DATA_YAML), "weights": str(checkpoint), "batch_size": 16, "imgsz": 640,
        "conf_thres": 0.001, "iou_thres": 0.60, "max_det": 500, "task": "val", "device": "0", "workers": 4,
        "single_cls": False, "augment": False, "verbose": True, "save_txt": True, "save_hybrid": False,
        "save_conf": True, "save_json": False, "project": prediction_root, "name": "yolov5_fp32", "exist_ok": True,
        "half": False, "dnn": False, "plots": False,
    }
    write_json(raw_root / "evaluation_command.json", parameters)
    start = utc_now()
    if not resume_postprocessing:
        capture_dir.mkdir()
        original_process_batch = val.process_batch
        pending: dict[str, np.ndarray] = {}

        def capture_process_batch(detections, labels, iouv):
            correct = original_process_batch(detections, labels, iouv)
            pending["detections"] = detections.detach().cpu().numpy()
            pending["labels"] = labels.detach().cpu().numpy()
            pending["correct"] = correct.detach().cpu().numpy()
            return correct

        def capture_image(pred, predn, path, names, im):
            if "correct" not in pending:
                raise RuntimeError(f"Missing full-precision process_batch capture for {path}")
            np.savez_compressed(
                capture_dir / f"{Path(path).stem}.npz", detections=pending["detections"], labels=pending["labels"],
                correct=pending["correct"], image_name=np.asarray(Path(path).name), checkpoint_sha256=np.asarray(record["checkpoint_sha256"]),
            )
            pending.clear()

        val.process_batch = capture_process_batch
        callbacks = Callbacks()
        callbacks.register_action("on_val_image_end", "stage11b_full_precision_capture", capture_image)
        results, class_maps, timing = val.run(**parameters, callbacks=callbacks)
        precision, recall, map50_yolo, map50_95_yolo = [float(value) for value in results[:4]]
    else:
        correct_parts, confidence_parts, class_parts, target_parts = [], [], [], []
        for image_path in sorted(image_paths, key=lambda item: item.name):
            captured = np.load(capture_dir / f"{image_path.stem}.npz")
            detections = np.asarray(captured["detections"])
            labels = np.asarray(captured["labels"])
            correct_parts.append(np.asarray(captured["correct"], dtype=bool))
            confidence_parts.append(detections[:, 4] if len(detections) else np.empty(0))
            class_parts.append(detections[:, 5] if len(detections) else np.empty(0))
            target_parts.append(labels[:, 0] if len(labels) else np.empty(0))
        tp, fp, p, r, _, ap, ap_class = val.ap_per_class(
            np.concatenate(correct_parts), np.concatenate(confidence_parts), np.concatenate(class_parts),
            np.concatenate(target_parts), plot=False, save_dir=raw_root,
            names={index: name for index, name in enumerate(["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"])},
        )
        ap50, ap_mean = ap[:, 0], ap.mean(1)
        precision, recall, map50_yolo, map50_95_yolo = float(p.mean()), float(r.mean()), float(ap50.mean()), float(ap_mean.mean())
        class_maps = np.full(10, map50_95_yolo)
        for index, class_id in enumerate(ap_class):
            class_maps[int(class_id)] = ap_mean[index]
        timing = [float("nan"), float("nan"), float("nan")]
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if not all(math.isfinite(value) for value in (precision, recall, f1, map50_yolo, map50_95_yolo)):
        raise RuntimeError("Non-finite base evaluator metrics")
    capture_files = sorted(capture_dir.glob("*.npz"))
    if len(capture_files) != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} full-precision files, found {len(capture_files)}")
    write_json(capture_dir / "capture_validation.json", {
        "schema": "stage11b_full_precision_capture_validation_v1", "status": "PASS", "run_id": record["run_id"],
        "capture_files": EXPECTED, "checkpoint_sha256": record["checkpoint_sha256"], "captured_during_primary_inference": True,
        "second_technical_inference_pass": False,
    })
    (capture_dir / ".complete").write_text("FULL_PRECISION_CAPTURE=PASS\n", encoding="ascii")

    labels_dir = prediction_root / "yolov5_fp32" / "labels"
    official_dir = prediction_root / "visdrone_txt"
    prediction_audit, predictions_by_stem = source.export_official_predictions(image_paths, labels_dir, official_dir)
    write_json(raw_root / "prediction_export_audit.json", prediction_audit)
    official_payload = port.evaluate(RAW_ANNOTATIONS, official_dir, RAW_IMAGES, raw_root / "visdrone_metrics.json", EXPECTED)
    official_percentage = official_payload["visdrone_metrics"]
    official_fraction = {key: float(value) / 100.0 for key, value in official_percentage.items()}

    coco_dir = prediction_root / "coco_records"
    annotation_json, prediction_json, image_ids = source.build_supplemental_coco_inputs(image_paths, predictions_by_stem, coco_dir)
    area = source.supplemental_coco_metrics(annotation_json, prediction_json, image_ids)
    write_json(coco_dir / "metrics.json", area)

    heldout_manifest = json.loads((OUT / "heldout1610_manifest.json").read_text(encoding="utf-8-sig"))
    heldout_by_name = {item["image_name"]: item for item in heldout_manifest["records"]}
    prediction_records: list[dict[str, Any]] = []
    for image_path in image_paths:
        yolo_path = labels_dir / f"{image_path.stem}.txt"
        official_path = official_dir / f"{image_path.stem}.txt"
        item = heldout_by_name[image_path.name]
        prediction_records.append({
            "image_name": image_path.name, "image_sha256": item["image_sha256"],
            "ground_truth_annotation_path": item["raw_annotation_path"], "ground_truth_annotation_sha256": item["raw_annotation_sha256"],
            "yolov5_prediction_path": str(yolo_path) if yolo_path.is_file() else None,
            "yolov5_prediction_sha256": sha256(yolo_path) if yolo_path.is_file() else None,
            "visdrone_prediction_path": str(official_path), "visdrone_prediction_sha256": sha256(official_path),
            "full_precision_prediction_path": str(capture_dir / f"{image_path.stem}.npz"),
            "full_precision_prediction_sha256": sha256(capture_dir / f"{image_path.stem}.npz"),
            "prediction_count": len(predictions_by_stem[image_path.stem]),
            "preprocessing": {"imgsz": 640, "single_scale": True, "letterbox_stride": 32, "augment": False, "tta": False},
        })
    prediction_manifest = {
        "schema": "stage11b_image_prediction_manifest_v1", "created_utc": utc_now(), "run_id": record["run_id"],
        "model": record["model"], "seed": record["seed"], "checkpoint_sha256": record["checkpoint_sha256"],
        "image_count": EXPECTED, "prediction_score_stage": "post_NMS_pre_metric_aggregation", "records": prediction_records,
        "coco_prediction_json": str(prediction_json), "coco_prediction_json_sha256": sha256(prediction_json), "status": "PASS",
    }
    write_json(prediction_root / "prediction_manifest.json", prediction_manifest)

    metrics = {
        "schema": "stage11b_per_checkpoint_metrics_v1", "created_utc": utc_now(), "status": "PASS", "run_id": record["run_id"],
        "model": record["model"], "seed": record["seed"], "architecture_signature": record["architecture_signature"],
        "terminal_checkpoint_path": str(checkpoint), "terminal_checkpoint_sha256": record["checkpoint_sha256"],
        "terminal_checkpoint_size_bytes": checkpoint.stat().st_size,
        "AP": official_fraction["AP"], "AP50": official_fraction["AP50"], "AP75": official_fraction["AP75"],
        "APS": area["APS_COCO_style_supplemental"], "APM": area["APM_COCO_style_supplemental"],
        "APL": area["APL_COCO_style_supplemental"], "Precision": precision, "Recall": recall, "F1": f1,
        "Params": int(record["params_exact"]), "FLOPs_raw": int(record["flops_raw_exact"]),
        "FLOPs_G": float(record["flops_raw_exact"]) / 1e9, "official_visdrone_metrics_fraction": official_fraction,
        "official_visdrone_metrics_percentage": official_percentage,
        "yolov5_base_metrics": {"AP50": map50_yolo, "AP": map50_95_yolo, "class_maps": [float(value) for value in np.asarray(class_maps).tolist()]},
        "timing_ms_per_image": [float(value) if math.isfinite(float(value)) else None for value in timing],
        "inference_reused_after_uncommitted_packaging_failure": resume_postprocessing,
        "evaluation_start_utc": start, "evaluation_end_utc": utc_now(),
        "precision_mode": "FP32", "imgsz": 640, "batch_size": 16, "conf_thres": 0.001, "nms_iou_thres": 0.60,
        "max_det": 500, "tta": False, "training_executed": False, "kd_executed": False, "teacher_loaded": False,
        "val548_reevaluated": False, "secondary_heldout_confirmation": True,
    }
    write_json(raw_root / "metrics.json", metrics)
    write_json(raw_root / "evaluation_record.json", {
        "run_id": record["run_id"], "model": record["model"], "seed": record["seed"],
        "metrics_path": str(raw_root / "metrics.json"), "metrics_sha256": sha256(raw_root / "metrics.json"),
        "prediction_manifest_path": str(prediction_root / "prediction_manifest.json"),
        "prediction_manifest_sha256": sha256(prediction_root / "prediction_manifest.json"),
        "status": "COMPLETE", "val548_reevaluated": False,
    })
    (raw_root / ".complete").write_text("STAGE11B_CHECKPOINT_EVALUATION=PASS\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "run_id": record["run_id"], "model": record["model"], "seed": record["seed"], "AP": metrics["AP"], "APS": metrics["APS"], "Recall": metrics["Recall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
