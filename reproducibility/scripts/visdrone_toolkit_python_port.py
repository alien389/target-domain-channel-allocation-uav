from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TOOLKIT_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
THRESHOLDS = tuple(float(value) for value in np.arange(0.5, 0.951, 0.05))
MAX_DETECTIONS = (1, 10, 100, 500)
CLASS_IDS = tuple(range(1, 11))


def load_matrix(path: Path) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.empty((0, 8), dtype=np.float64)
    matrix = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if matrix.shape[1] < 6:
        raise RuntimeError(f"VisDrone row has fewer than six columns: {path}")
    return matrix


def matlab_round_positive(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5)


def filter_by_ignored_regions(
    ground_truth: np.ndarray,
    detections: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_gt = ground_truth[ground_truth[:, 5] != 0].copy()
    ignored = np.maximum(1.0, ground_truth[ground_truth[:, 5] == 0, :4])
    if ignored.size:
        mask = np.zeros((image_height, image_width), dtype=np.float64)
        for x_value, y_value, width_value, height_value in ignored:
            x = int(x_value)
            y = int(y_value)
            x_end = min(image_width, int(x_value + width_value))
            y_end = min(image_height, int(y_value + height_value))
            mask[y - 1 : y_end, x - 1 : x_end] = 1.0
        integral = np.cumsum(np.cumsum(mask, axis=0), axis=1)

        def retain(rows: np.ndarray) -> np.ndarray:
            kept: list[int] = []
            for index, row in enumerate(rows):
                position = np.maximum(1.0, matlab_round_positive(row[:4]))
                x = max(1, min(image_width, int(position[0])))
                y = max(1, min(image_height, int(position[1])))
                width = int(position[2])
                height = int(position[3])
                x2 = min(image_width, x + width)
                y2 = max(1, min(image_height, y + height))
                top_left = integral[y - 1, x - 1]
                top_right = integral[y - 1, x2 - 1]
                bottom_left = integral[y2 - 1, x - 1]
                bottom_right = integral[y2 - 1, x2 - 1]
                ignored_value = top_left + bottom_right - top_right - bottom_left
                if ignored_value / (height * width) < 0.5:
                    kept.append(index)
            return rows[np.asarray(kept, dtype=np.int64)] if kept else rows[:0]

        current_gt = retain(current_gt)
        detections = retain(detections)
    output_gt = current_gt.copy()
    original_flags = current_gt[:, 4].copy()
    output_gt[original_flags == 0, 4] = 1
    output_gt[original_flags == 1, 4] = 0
    return output_gt, detections


def overlap_matrix(
    detections: np.ndarray,
    ground_truth: np.ndarray,
    ignored: np.ndarray,
) -> np.ndarray:
    count_det = len(detections)
    count_gt = len(ground_truth)
    if not count_det or not count_gt:
        return np.zeros((count_det, count_gt), dtype=np.float64)
    det_end = detections[:, :2] + detections[:, 2:4]
    gt_end = ground_truth[:, :2] + ground_truth[:, 2:4]
    det_area = detections[:, 2] * detections[:, 3]
    gt_area = ground_truth[:, 2] * ground_truth[:, 3]
    width = np.minimum(det_end[:, None, 0], gt_end[None, :, 0]) - np.maximum(
        detections[:, None, 0], ground_truth[None, :, 0]
    )
    height = np.minimum(det_end[:, None, 1], gt_end[None, :, 1]) - np.maximum(
        detections[:, None, 1], ground_truth[None, :, 1]
    )
    intersection = np.maximum(width, 0.0) * np.maximum(height, 0.0)
    union = (
        det_area[:, None]
        + gt_area[None, :]
        - intersection
    )
    union[:, ignored] = det_area[:, None]
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def eval_result(
    gt_input: np.ndarray,
    det_input: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    gt0 = (
        np.asarray(gt_input, dtype=np.float64).reshape(-1, 5)
        if gt_input.size
        else np.empty((0, 5), dtype=np.float64)
    )
    dt0 = (
        np.asarray(det_input, dtype=np.float64).reshape(-1, 5)
        if det_input.size
        else np.empty((0, 5), dtype=np.float64)
    )
    if len(dt0):
        dt0 = dt0[np.argsort(-dt0[:, 4], kind="stable")]
    if len(gt0):
        gt0 = gt0[np.argsort(gt0[:, 4], kind="stable")]
    gt = gt0.copy()
    if len(gt):
        gt[:, 4] = -gt[:, 4]
    det = np.column_stack((dt0, np.zeros(len(dt0), dtype=np.float64)))
    overlaps = overlap_matrix(det[:, :4], gt[:, :4], gt[:, 4] == -1)
    for det_index in range(len(det)):
        best_overlap = threshold
        best_gt = -1
        best_match = 0
        for gt_index in range(len(gt)):
            match = gt[gt_index, 4]
            if match == 1:
                continue
            if best_match != 0 and match == -1:
                break
            if overlaps[det_index, gt_index] < best_overlap:
                continue
            best_overlap = overlaps[det_index, gt_index]
            best_gt = gt_index
            best_match = 1 if match == 0 else -1
        if best_match == -1:
            det[det_index, 5] = -1
        elif best_match == 1:
            gt[best_gt, 4] = 1
            det[det_index, 5] = 1
    return gt, det


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    indices = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[indices] - mrec[indices - 1]) * mpre[indices]))


def prepare_records(
    annotation_dir: Path,
    prediction_dir: Path,
    image_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    names = sorted(path.name for path in annotation_dir.iterdir() if path.is_file())
    all_gt: list[np.ndarray] = []
    all_det: list[np.ndarray] = []
    for name in names:
        gt = load_matrix(annotation_dir / name)
        prediction_path = prediction_dir / name
        det = (
            load_matrix(prediction_path)
            if prediction_path.is_file()
            else np.empty((0, 8), dtype=np.float64)
        )
        image = cv2.imread(str(image_dir / f"{Path(name).stem}.jpg"))
        if image is None:
            raise FileNotFoundError(f"Unable to read VisDrone image for evaluator: {name}")
        filtered_gt, filtered_det = filter_by_ignored_regions(
            gt,
            det,
            int(image.shape[0]),
            int(image.shape[1]),
        )
        all_gt.append(filtered_gt)
        all_det.append(filtered_det)
    return all_gt, all_det, names


def calculate_accuracy(
    all_gt: list[np.ndarray],
    all_det: list[np.ndarray],
) -> dict[str, float]:
    ap = np.zeros((10, 10), dtype=np.float64)
    ar = np.zeros((10, 10, 4), dtype=np.float64)
    evaluated_classes: list[int] = []
    for class_index, class_id in enumerate(CLASS_IDS):
        for gt in all_gt:
            if np.count_nonzero(gt[:, 5] == class_id):
                evaluated_classes.append(class_index)
        for threshold_index, threshold in enumerate(THRESHOLDS):
            final_recall = np.empty(0, dtype=np.float64)
            final_precision = np.empty(0, dtype=np.float64)
            for max_index, max_detections in enumerate(MAX_DETECTIONS):
                gt_matches: list[np.ndarray] = []
                det_matches: list[np.ndarray] = []
                for gt, det in zip(all_gt, all_det):
                    class_gt = gt[gt[:, 5] == class_id, :5]
                    limited = det[: min(len(det), max_detections)]
                    class_det = limited[limited[:, 5] == class_id, :5]
                    matched_gt, matched_det = eval_result(
                        class_gt,
                        class_det,
                        threshold,
                    )
                    gt_matches.append(matched_gt[:, 4])
                    det_matches.append(matched_det[:, 4:6])
                gt_match = (
                    np.concatenate(gt_matches)
                    if gt_matches
                    else np.empty(0, dtype=np.float64)
                )
                det_match = (
                    np.concatenate(det_matches)
                    if det_matches
                    else np.empty((0, 2), dtype=np.float64)
                )
                if len(det_match):
                    rank = np.argsort(-det_match[:, 0], kind="stable")
                    true_positive = np.cumsum(det_match[rank, 1] == 1)
                    false_positive = np.cumsum(det_match[rank, 1] == 0)
                    recall = true_positive / max(1, len(gt_match))
                    precision = true_positive / np.maximum(
                        1,
                        false_positive + true_positive,
                    )
                else:
                    recall = np.empty(0, dtype=np.float64)
                    precision = np.empty(0, dtype=np.float64)
                if len(recall):
                    ar[class_index, threshold_index, max_index] = (
                        float(np.max(recall)) * 100.0
                    )
                final_recall = recall
                final_precision = precision
            ap[class_index, threshold_index] = (
                voc_ap(final_recall, final_precision) * 100.0
            )
    if not evaluated_classes:
        raise RuntimeError("No evaluable VisDrone classes were found.")
    indices = np.asarray(evaluated_classes, dtype=np.int64)
    return {
        "AP": float(np.mean(ap[indices, :])),
        "AP50": float(np.mean(ap[indices, 0])),
        "AP75": float(np.mean(ap[indices, 5])),
        "AR1": float(np.mean(ar[indices, :, 0])),
        "AR10": float(np.mean(ar[indices, :, 1])),
        "AR100": float(np.mean(ar[indices, :, 2])),
        "AR500": float(np.mean(ar[indices, :, 3])),
    }


def evaluate(
    annotation_dir: Path,
    prediction_dir: Path,
    image_dir: Path,
    output_path: Path,
    expected_count: int = 1294,
) -> dict[str, Any]:
    all_gt, all_det, names = prepare_records(
        annotation_dir,
        prediction_dir,
        image_dir,
    )
    if len(names) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} inner-validation annotations, found {len(names)}."
        )
    metrics = calculate_accuracy(all_gt, all_det)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Python-port VisDrone metrics contain non-finite values.")
    payload = {
        "visdrone_metrics": metrics,
        "metadata": {
            "evaluator": (
                "Line-for-line numerical Python port of the official VisDrone "
                "DET MATLAB toolkit, parity audited on Stage04 controls"
            ),
            "toolkit_commit": TOOLKIT_COMMIT,
            "evaluated_images": len(names),
            "max_detections_per_image": 500,
            "metric_scale": "percentage_0_to_100",
            "original_train_annotations_copied_read_only": True,
            "official_val548_used": False,
            "testdev_used": False,
            "implementation": str(Path(__file__).resolve()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1294)
    args = parser.parse_args()
    payload = evaluate(
        args.annotations.resolve(),
        args.predictions.resolve(),
        args.images.resolve(),
        args.output.resolve(),
        args.expected_count,
    )
    print(json.dumps(payload["visdrone_metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
