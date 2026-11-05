from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from stage15_common import VISDRONE_CLASSES, YOLO_ROOT, read_json, require, write_json


if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))
import val  # noqa: E402


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image_id = item["image_id"]
            require(image_id not in records, f"Duplicate prediction image ID: {image_id}")
            records[image_id] = item
    return records


def filter_uavdt_ignore(detections: list[list[float]], regions: list[list[int]]) -> tuple[list[list[float]], int]:
    kept: list[list[float]] = []
    removed = 0
    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        inside = any(
            x1 > left and y1 > top and x2 < left + width and y2 < top + height
            for left, top, width, height in regions
        )
        if inside:
            removed += 1
        else:
            kept.append(detection)
    return kept, removed


def coco_precision(evaluator: COCOeval, area_label: str = "all", iou: float | None = None, category_index: int | None = None) -> float:
    max_index = evaluator.params.maxDets.index(500)
    area_index = evaluator.params.areaRngLbl.index(area_label)
    values = evaluator.eval["precision"][:, :, :, area_index, max_index]
    if iou is not None:
        threshold_index = int(np.where(np.isclose(evaluator.params.iouThrs, iou))[0][0])
        values = values[threshold_index : threshold_index + 1]
    if category_index is not None:
        values = values[:, :, category_index : category_index + 1]
    valid = values[values > -1]
    require(valid.size > 0, f"No valid COCO precision values: area={area_label} iou={iou} category={category_index}")
    return float(valid.mean())


def evaluate_external(
    dataset: str,
    predictions_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = dataset.lower()
    if dataset == "carpk":
        prep_root = predictions_path.parents[3] / "01_CARPK_PREP"
        prep = read_json(prep_root / "CARPK_PREP_CONFIG.json")
        manifest_path = prep_root / "CARPK_TEST_MANIFEST.csv"
        annotation_path = Path(prep["coco_annotation_path"])
        target_to_source = {1: 3}
        allowed_sources = {3}
        target_names = {1: "car"}
        ignore_regions: dict[str, list[list[int]]] = {}
    elif dataset == "uavdt":
        prep_root = predictions_path.parents[3] / "02_UAVDT_PREP"
        prep = read_json(prep_root / "UAVDT_PREP_CONFIG.json")
        manifest_path = prep_root / "UAVDT_TEST_MANIFEST.csv"
        annotation_path = Path(prep["coco_annotation_path"])
        target_to_source = {1: 3, 2: 5, 3: 8}
        allowed_sources = {3, 5, 8}
        target_names = {1: "car", 2: "truck", 3: "bus"}
        ignore_regions = read_json(Path(prep["ignore_regions_path"]))["regions_by_image"]
    else:
        raise RuntimeError(f"Unsupported dataset: {dataset}")
    source_to_target = {source: target for target, source in target_to_source.items()}

    import csv

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    captured = load_jsonl(predictions_path)
    require(len(captured) == len(manifest), f"Prediction image count mismatch: {len(captured)} != {len(manifest)}")
    gt_payload = read_json(annotation_path)
    gt_images = {item["id"]: item for item in gt_payload["images"]}
    gt_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in gt_payload["annotations"]:
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    coco_predictions: list[dict[str, Any]] = []
    correctness: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    predicted_classes: list[np.ndarray] = []
    target_classes: list[np.ndarray] = []
    final_records: list[dict[str, Any]] = []
    ignored_detections = 0
    raw_selected_predictions = 0
    iouv = torch.linspace(0.5, 0.95, 10)
    for row in manifest:
        image_key = row["image_id"]
        item = captured[image_key]
        detections = [
            [float(value) for value in detection]
            for detection in item["detections"]
            if int(round(float(detection[5]))) in allowed_sources
        ]
        raw_selected_predictions += len(detections)
        if dataset == "uavdt":
            detections, removed = filter_uavdt_ignore(detections, ignore_regions.get(image_key, []))
            ignored_detections += removed
        image_id = int(row["order"])
        gt_rows = gt_by_image.get(image_id, [])
        det_tensor = torch.tensor(detections, dtype=torch.float32) if detections else torch.empty((0, 6), dtype=torch.float32)
        labels = []
        for annotation in gt_rows:
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            labels.append([target_to_source[int(annotation["category_id"])], x, y, x + width, y + height])
        label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0, 5), dtype=torch.float32)
        correct = val.process_batch(det_tensor, label_tensor, iouv) if len(detections) and len(labels) else torch.zeros((len(detections), 10), dtype=torch.bool)
        correctness.append(correct.cpu().numpy())
        confidences.append(det_tensor[:, 4].cpu().numpy() if len(detections) else np.empty(0, dtype=np.float32))
        predicted_classes.append(det_tensor[:, 5].cpu().numpy() if len(detections) else np.empty(0, dtype=np.float32))
        target_classes.append(label_tensor[:, 0].cpu().numpy() if len(labels) else np.empty(0, dtype=np.float32))
        mapped_detections = []
        for x1, y1, x2, y2, confidence, source_class in detections:
            source_id = int(round(source_class))
            target_id = source_to_target[source_id]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            coco_predictions.append(
                {"image_id": image_id, "category_id": target_id, "bbox": [x1, y1, width, height], "score": confidence}
            )
            mapped_detections.append(
                {
                    "box_xyxy": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "source_class_id": source_id,
                    "source_class_name": VISDRONE_CLASSES[source_id],
                    "mapped_class_id": target_id,
                    "mapped_class_name": target_names[target_id],
                    "post_nms": True,
                }
            )
        final_records.append(
            {
                "image_id": image_key,
                "original_width": int(row["width"]),
                "original_height": int(row["height"]),
                "input_shape": item["input_shape"],
                "letterbox_ratio": item["letterbox_ratio"],
                "letterbox_pad": item["letterbox_pad"],
                "detections": mapped_detections,
            }
        )

    require(coco_predictions, f"No mapped {dataset} predictions were produced")
    coco_predictions_path = output_dir / "coco_predictions.json"
    write_json(coco_predictions_path, coco_predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation_path))
        coco_dt = coco_gt.loadRes(str(coco_predictions_path))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(gt_images)
        evaluator.params.catIds = sorted(target_to_source)
        evaluator.params.maxDets = [1, 10, 500]
        evaluator.params.useCats = 1
        evaluator.evaluate()
        evaluator.accumulate()

    correct_array = np.concatenate(correctness, axis=0)
    confidence_array = np.concatenate(confidences)
    predicted_class_array = np.concatenate(predicted_classes)
    target_class_array = np.concatenate(target_classes)
    tp, fp, precision_by_class, recall_by_class, f1_by_class, ap, ap_class = val.ap_per_class(
        correct_array,
        confidence_array,
        predicted_class_array,
        target_class_array,
        plot=False,
        save_dir=output_dir,
        names={index: name for index, name in enumerate(VISDRONE_CLASSES)},
    )
    precision = float(np.mean(precision_by_class))
    recall = float(np.mean(recall_by_class))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics: dict[str, Any] = {
        "AP": coco_precision(evaluator),
        "AP50": coco_precision(evaluator, iou=0.50),
        "AP75": coco_precision(evaluator, iou=0.75),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "number_images": len(manifest),
        "number_gt_objects": len(gt_payload["annotations"]),
        "number_predictions_before_ignore": raw_selected_predictions,
        "number_predictions": len(coco_predictions),
        "ignored_region_predictions_removed": ignored_detections,
        "recall_definition": "YOLOv5 ap_per_class mean recall at the confidence index maximizing smoothed mean F1, after the frozen class/ignore adapter",
        "ap_definition": "pycocotools COCOeval IoU=0.50:0.05:0.95, maxDets=500, shared target categories only",
    }
    if dataset == "uavdt":
        for category_index, category_id in enumerate(evaluator.params.catIds):
            metrics[f"AP_{target_names[category_id]}"] = coco_precision(evaluator, category_index=category_index)

    exact_predictions_path = output_dir / "predictions.jsonl"
    temp = exact_predictions_path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for item in final_records:
            handle.write(json.dumps(item, separators=(",", ":"), ensure_ascii=True) + "\n")
    temp.replace(exact_predictions_path)
    return metrics

