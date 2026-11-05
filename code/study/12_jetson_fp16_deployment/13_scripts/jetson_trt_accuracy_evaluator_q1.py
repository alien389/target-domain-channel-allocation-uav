#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_NAMES = [
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
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10, dtype=np.float64)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen evaluator source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ordered_images(path: Path, image_root: Path, expected: int) -> list[Path]:
    images = [Path(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(images) != expected or len(set(images)) != expected:
        raise RuntimeError(f"Image-list contract failed: rows={len(images)}, unique={len(set(images))}")
    resolved_root = image_root.resolve()
    for image in images:
        if not image.is_file() or image.parent.resolve() != resolved_root:
            raise RuntimeError(f"Missing or out-of-root image: {image}")
    return images


def parse_annotation(path: Path, width: int, height: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels: list[list[float]] = []
    coco_rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip().rstrip(",")
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            raise RuntimeError(f"Malformed VisDrone annotation {path}:{line_number}")
        x, y, box_width, box_height, score = map(float, parts[:5])
        category = int(float(parts[5]))
        if not all(math.isfinite(value) for value in (x, y, box_width, box_height, score)):
            raise RuntimeError(f"Non-finite VisDrone annotation {path}:{line_number}")
        if not 1 <= category <= 10 or box_width <= 0.0 or box_height <= 0.0:
            continue
        coco_rows.append(
            {
                "category_id": category,
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
            }
        )
        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + box_width))
        y2 = max(0.0, min(float(height), y + box_height))
        if x2 > x1 and y2 > y1:
            labels.append([float(category - 1), x1, y1, x2, y2])
    matrix = np.asarray(labels, dtype=np.float64).reshape(-1, 5) if labels else np.empty((0, 5), dtype=np.float64)
    return matrix, coco_rows


def pairwise_iou(labels: np.ndarray, detections: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    if not len(labels) or not len(detections):
        return np.zeros((len(labels), len(detections)), dtype=np.float64)
    left_top = np.maximum(labels[:, None, :2], detections[None, :, :2])
    right_bottom = np.minimum(labels[:, None, 2:4], detections[None, :, 2:4])
    intersection = np.maximum(right_bottom - left_top, 0.0).prod(2)
    label_area = np.maximum(labels[:, 2] - labels[:, 0], 0.0) * np.maximum(labels[:, 3] - labels[:, 1], 0.0)
    detection_area = np.maximum(detections[:, 2] - detections[:, 0], 0.0) * np.maximum(detections[:, 3] - detections[:, 1], 0.0)
    return intersection / (label_area[:, None] + detection_area[None, :] - intersection + eps)


def process_batch(detections: np.ndarray, labels: np.ndarray) -> np.ndarray:
    correct = np.zeros((len(detections), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(detections) or not len(labels):
        return correct
    iou = pairwise_iou(labels[:, 1:5], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5][None, :]
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        label_indices, detection_indices = np.where((iou >= threshold) & correct_class)
        if not len(label_indices):
            continue
        matches = np.column_stack((label_indices, detection_indices, iou[label_indices, detection_indices]))
        if len(matches) > 1:
            matches = matches[np.argsort(-matches[:, 2], kind="stable")]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def smooth(values: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    count = round(len(values) * fraction * 2) // 2 + 1
    padding = np.ones(count // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]))
    return np.convolve(padded, np.ones(count) / count, mode="valid")


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    metric_recall = np.concatenate(([0.0], recall, [1.0]))
    metric_precision = np.concatenate(([1.0], precision, [0.0]))
    metric_precision = np.flip(np.maximum.accumulate(np.flip(metric_precision)))
    x_axis = np.linspace(0, 1, 101)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(np.interp(x_axis, metric_recall, metric_precision), x_axis))


def yolo_base_metrics(
    correct_parts: list[np.ndarray],
    confidence_parts: list[np.ndarray],
    class_parts: list[np.ndarray],
    target_parts: list[np.ndarray],
) -> dict[str, Any]:
    true_positive = np.concatenate(correct_parts, axis=0)
    confidence = np.concatenate(confidence_parts)
    predicted_class = np.concatenate(class_parts)
    target_class = np.concatenate(target_parts)
    if not len(target_class):
        raise RuntimeError("No target labels were available for base metrics")
    order = np.argsort(-confidence)
    true_positive = true_positive[order]
    confidence = confidence[order]
    predicted_class = predicted_class[order]
    unique_classes, target_counts = np.unique(target_class, return_counts=True)
    class_count = len(unique_classes)
    px = np.linspace(0, 1, 1000)
    ap = np.zeros((class_count, true_positive.shape[1]))
    precision_curve = np.zeros((class_count, 1000))
    recall_curve = np.zeros((class_count, 1000))
    for class_index, class_id in enumerate(unique_classes):
        selected = predicted_class == class_id
        prediction_count = int(selected.sum())
        label_count = int(target_counts[class_index])
        if prediction_count == 0 or label_count == 0:
            continue
        false_positive_cumulative = (1 - true_positive[selected]).cumsum(0)
        true_positive_cumulative = true_positive[selected].cumsum(0)
        recall = true_positive_cumulative / (label_count + 1e-16)
        recall_curve[class_index] = np.interp(-px, -confidence[selected], recall[:, 0], left=0)
        precision = true_positive_cumulative / (true_positive_cumulative + false_positive_cumulative)
        precision_curve[class_index] = np.interp(-px, -confidence[selected], precision[:, 0], left=1)
        for threshold_index in range(true_positive.shape[1]):
            ap[class_index, threshold_index] = compute_ap(recall[:, threshold_index], precision[:, threshold_index])
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-16)
    best_index = int(smooth(f1_curve.mean(0), 0.1).argmax())
    precision_by_class = precision_curve[:, best_index]
    recall_by_class = recall_curve[:, best_index]
    mean_precision = float(precision_by_class.mean())
    mean_recall = float(recall_by_class.mean())
    f1 = 2.0 * mean_precision * mean_recall / (mean_precision + mean_recall) if mean_precision + mean_recall else 0.0
    return {
        "Precision": mean_precision,
        "Recall": mean_recall,
        "F1": f1,
        "AP50_YOLO_style_diagnostic": float(ap[:, 0].mean()),
        "AP_YOLO_style_diagnostic": float(ap.mean(1).mean()),
        "best_F1_confidence_grid_index": best_index,
        "best_F1_confidence_grid_value": float(px[best_index]),
        "evaluated_classes": [int(value) for value in unique_classes],
        "target_counts": [int(value) for value in target_counts],
    }


def supplemental_area_metrics(ground_truth: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": ground_truth["images"],
            "annotations": [],
            "categories": ground_truth["categories"],
        }
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [int(row["id"]) for row in ground_truth["images"]]
    evaluator.params.catIds = list(range(1, 11))
    evaluator.params.maxDets = [1, 10, 500]
    evaluator.params.useCats = 1
    evaluator.evaluate()
    evaluator.accumulate()
    maximum_index = evaluator.params.maxDets.index(500)

    def average(area_label: str, iou_threshold: float | None = None) -> float:
        area_index = evaluator.params.areaRngLbl.index(area_label)
        values = evaluator.eval["precision"][:, :, :, area_index, maximum_index]
        if iou_threshold is not None:
            threshold_index = int(np.where(np.isclose(evaluator.params.iouThrs, iou_threshold))[0][0])
            values = values[threshold_index : threshold_index + 1]
        valid = values[values > -1]
        if not valid.size:
            raise RuntimeError(f"No valid COCO precision for area={area_label}")
        return float(valid.mean())

    return {
        "AP_COCO_style_diagnostic": average("all"),
        "AP50_COCO_style_diagnostic": average("all", 0.50),
        "AP75_COCO_style_diagnostic": average("all", 0.75),
        "APS": average("small"),
        "APM": average("medium"),
        "APL": average("large"),
    }


def write_gzip_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-evaluator", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--val-py-source", type=Path, required=True)
    parser.add_argument("--metrics-py-source", type=Path, required=True)
    parser.add_argument("--supplemental-source", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--paper-name", required=True)
    parser.add_argument("--precision-policy", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--architecture-signature", required=True)
    parser.add_argument("--params", type=int, required=True)
    parser.add_argument("--flops-raw", type=int, required=True)
    parser.add_argument("--split", choices=("val548", "heldout1610"), required=True)
    parser.add_argument("--split-role", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--ordered-images", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=500)
    args = parser.parse_args()

    started_utc = utc_now()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    official_directory = output / "official_prediction_txt"
    official_directory.mkdir()

    required = [
        args.base_evaluator,
        args.official_evaluator,
        args.val_py_source,
        args.metrics_py_source,
        args.supplemental_source,
        args.engine,
        args.image_root,
        args.annotation_root,
        args.ordered_images,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    images = ordered_images(args.ordered_images, args.image_root, args.expected_images)
    if len(list(args.annotation_root.glob("*.txt"))) != args.expected_images:
        raise RuntimeError("Annotation-directory count does not match the locked split")

    base = load_module("meta_accuracy_trt_base", args.base_evaluator.resolve())
    official = load_module("meta_accuracy_visdrone_official_port", args.official_evaluator.resolve())
    runner = base.TrtEngineRunner(str(args.engine.resolve()))
    if len([row for row in runner.binding_info if row["is_input"]]) != 1:
        raise RuntimeError("TensorRT engine does not have exactly one input")
    if np.dtype(runner.input_dtype) not in (np.dtype(np.float32), np.dtype(np.float16)):
        raise RuntimeError(f"Unsupported TensorRT input dtype: {runner.input_dtype}")

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    coco_detections: list[dict[str, Any]] = []
    correct_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    all_detections: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    detection_offsets = [0]
    label_offsets = [0]
    prediction_records: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not decode image: {image_path}")
        height, width = image.shape[:2]
        annotation_path = args.annotation_root / f"{image_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        labels, area_labels = parse_annotation(annotation_path, width, height)
        coco_images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        for area_label in area_labels:
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": area_label["category_id"],
                    "bbox": area_label["bbox"],
                    "area": area_label["area"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        tensor, ratio, pad = base.preprocess_bgr(image, imgsz=args.imgsz, stride=args.stride, dtype=np.dtype(runner.input_dtype))
        raw = runner.infer(tensor)
        if list(raw.shape) != [1, 25200, 15] or not np.isfinite(raw).all():
            raise RuntimeError(f"Raw output contract failed for {image_path.name}: {raw.shape}")
        detections = base.nms_yolov5(
            raw,
            conf_thres=args.conf,
            iou_thres=args.iou,
            max_det=args.max_det,
            multi_label=True,
            agnostic=False,
        )[0]
        if len(detections):
            detections = detections.astype(np.float64, copy=True)
            detections[:, :4] = base.scale_boxes(
                (args.imgsz, args.imgsz),
                detections[:, :4],
                (height, width),
                ratio_pad=(ratio, pad),
            )
            valid = (
                np.isfinite(detections).all(axis=1)
                & (detections[:, 2] > detections[:, 0])
                & (detections[:, 3] > detections[:, 1])
                & (detections[:, 4] >= args.conf)
                & (detections[:, 5] >= 0)
                & (detections[:, 5] < 10)
            )
            detections = detections[valid]
        else:
            detections = np.empty((0, 6), dtype=np.float64)
        if len(detections) > args.max_det:
            raise RuntimeError("NMS exceeded max_det")

        correct = process_batch(detections, labels)
        correct_parts.append(correct)
        confidence_parts.append(detections[:, 4])
        class_parts.append(detections[:, 5])
        target_parts.append(labels[:, 0])
        all_detections.append(detections.astype(np.float32))
        all_labels.append(labels.astype(np.float32))
        detection_offsets.append(detection_offsets[-1] + len(detections))
        label_offsets.append(label_offsets[-1] + len(labels))

        official_lines: list[str] = []
        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection.tolist()
            box_width = x2 - x1
            box_height = y2 - y1
            category_id = int(class_id) + 1
            official_lines.append(
                f"{x1:.10f},{y1:.10f},{box_width:.10f},{box_height:.10f},{score:.10f},{category_id},-1,-1"
            )
            coco_detections.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(box_width), float(box_height)],
                    "score": float(score),
                }
            )
        prediction_path = official_directory / f"{image_path.stem}.txt"
        atomic_text(prediction_path, "\n".join(official_lines) + ("\n" if official_lines else ""))
        prediction_records.append(
            {
                "index": image_id - 1,
                "image_name": image_path.name,
                "image_sha256": sha256(image_path),
                "annotation_sha256": sha256(annotation_path),
                "prediction_file": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "prediction_count": len(detections),
                "target_count": len(labels),
            }
        )
        if image_id == 1 or image_id % 100 == 0 or image_id == len(images):
            print(f"ACCURACY_PROGRESS={image_id}/{len(images)} engine={args.engine_id} split={args.split}", flush=True)

    prediction_npz = output / "full_precision_post_nms.npz"
    npz_temporary = output / "full_precision_post_nms.tmp.npz"
    np.savez_compressed(
        npz_temporary,
        image_names=np.asarray([path.name for path in images]),
        detections=np.concatenate(all_detections, axis=0) if all_detections else np.empty((0, 6), dtype=np.float32),
        detection_offsets=np.asarray(detection_offsets, dtype=np.int64),
        labels=np.concatenate(all_labels, axis=0) if all_labels else np.empty((0, 5), dtype=np.float32),
        label_offsets=np.asarray(label_offsets, dtype=np.int64),
        correct=np.concatenate(correct_parts, axis=0) if correct_parts else np.empty((0, 10), dtype=bool),
        checkpoint_sha256=np.asarray(args.checkpoint_sha256.upper()),
        engine_sha256=np.asarray(sha256(args.engine)),
    )
    npz_temporary.replace(prediction_npz)

    official_payload = official.evaluate(
        args.annotation_root.resolve(),
        official_directory,
        args.image_root.resolve(),
        output / "visdrone_official_metrics.json",
        args.expected_images,
    )
    official_percentage = official_payload["visdrone_metrics"]
    official_fraction = {key: float(value) / 100.0 for key, value in official_percentage.items()}
    ground_truth = {
        "info": {
            "description": f"{args.split} supplemental area-metric annotations",
            "ignored_region_note": "Categories 0 and 11 are excluded only from supplemental area metrics.",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)],
    }
    area = supplemental_area_metrics(ground_truth, coco_detections)
    base_metrics = yolo_base_metrics(correct_parts, confidence_parts, class_parts, target_parts)
    write_gzip_json(output / "supplemental_coco_predictions.json.gz", coco_detections)
    atomic_json(output / "supplemental_area_metrics.json", area)
    atomic_json(output / "yolov5_base_metrics.json", base_metrics)

    metrics = {
        "schema": "meta_jetson_actual_engine_accuracy_metrics_v1",
        "created_utc": utc_now(),
        "status": "PASS",
        "engine_id": args.engine_id,
        "model_id": args.model_id,
        "paper_name": args.paper_name,
        "precision_policy": args.precision_policy,
        "split": args.split,
        "split_role": args.split_role,
        "image_count": args.expected_images,
        "AP": official_fraction["AP"],
        "AP50": official_fraction["AP50"],
        "AP75": official_fraction["AP75"],
        "APS": area["APS"],
        "APM": area["APM"],
        "APL": area["APL"],
        "Precision": base_metrics["Precision"],
        "Recall": base_metrics["Recall"],
        "F1": base_metrics["F1"],
        "Params": args.params,
        "FLOPs_raw": args.flops_raw,
        "FLOPs_G": args.flops_raw / 1e9,
        "engine_sha256": sha256(args.engine),
        "checkpoint_sha256": args.checkpoint_sha256.upper(),
        "architecture_signature": args.architecture_signature.upper(),
        "official_visdrone_metrics_percentage": official_percentage,
        "official_visdrone_metrics_fraction": official_fraction,
        "yolov5_base_metrics": base_metrics,
        "supplemental_area_metrics": area,
        "settings": {
            "precision": "actual_engine_binding_precision",
            "single_scale": True,
            "imgsz": args.imgsz,
            "batch_size": 1,
            "conf_thres": args.conf,
            "nms_iou_thres": args.iou,
            "max_det": args.max_det,
            "multi_label": True,
            "augment": False,
            "tta": False,
            "letterbox_auto": False,
            "stride": args.stride,
        },
        "duration_seconds": time.monotonic() - started,
    }
    if not all(math.isfinite(float(metrics[key])) for key in ("AP", "AP50", "AP75", "APS", "APM", "APL", "Precision", "Recall", "F1")):
        raise RuntimeError("Accuracy metrics contain NaN or Infinity")
    atomic_json(output / "metrics.json", metrics)
    atomic_csv(
        output / "metrics.csv",
        [metrics],
        [
            "engine_id", "model_id", "paper_name", "precision_policy", "split", "image_count",
            "AP", "AP50", "AP75", "APS", "APM", "APL", "Precision", "Recall", "F1",
            "Params", "FLOPs_raw", "FLOPs_G", "engine_sha256", "checkpoint_sha256", "architecture_signature",
        ],
    )
    atomic_csv(
        output / "prediction_file_manifest.csv",
        prediction_records,
        [
            "index", "image_name", "image_sha256", "annotation_sha256", "prediction_file",
            "prediction_sha256", "prediction_count", "target_count",
        ],
    )
    source_hashes = {
        "base_tensor_runtime_evaluator": {"path": str(args.base_evaluator.resolve()), "sha256": sha256(args.base_evaluator)},
        "official_visdrone_python_port": {"path": str(args.official_evaluator.resolve()), "sha256": sha256(args.official_evaluator)},
        "frozen_yolov5_val_source": {"path": str(args.val_py_source.resolve()), "sha256": sha256(args.val_py_source)},
        "frozen_yolov5_metrics_source": {"path": str(args.metrics_py_source.resolve()), "sha256": sha256(args.metrics_py_source)},
        "frozen_supplemental_area_source": {"path": str(args.supplemental_source.resolve()), "sha256": sha256(args.supplemental_source)},
        "adapter": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    atomic_json(output / "evaluator_source_hashes.json", source_hashes)
    artifacts = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and not item.name.endswith(".tmp")):
        artifacts.append({"relative_path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": "meta_jetson_actual_engine_accuracy_unit_manifest_v1",
        "created_utc": utc_now(),
        "status": "PASS",
        "engine_id": args.engine_id,
        "engine_sha256": sha256(args.engine),
        "split": args.split,
        "expected_images": args.expected_images,
        "evaluated_images": len(images),
        "dataset_manifest_sha256": args.dataset_manifest_sha256.upper(),
        "ordered_image_list_sha256": sha256(args.ordered_images),
        "full_precision_predictions_sha256": sha256(prediction_npz),
        "metric_source_of_truth": "metrics.json",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "artifacts": artifacts,
    }
    atomic_json(output / "accuracy_manifest.json", manifest)
    atomic_text(output / ".unit_complete", "ACTUAL_ENGINE_ACCURACY_UNIT=PASS\n")
    print(json.dumps({"status": "PASS", "engine_id": args.engine_id, "split": args.split, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
