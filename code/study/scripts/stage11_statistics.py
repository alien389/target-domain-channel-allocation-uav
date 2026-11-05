from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import csv
import hashlib
import importlib.util
import json
import math
import pickle
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(_release_paths.expand('${STUDY_ROOT}'))
EVAL = ROOT / "11_official_val_statistics" / "01_val548_evaluation"
OUT = ROOT / "11_official_val_statistics" / "02_statistics"
PORT_PATH = ROOT / "scripts" / "visdrone_toolkit_python_port.py"
YOLO_ROOT = Path(_release_paths.expand('${YOLOV5_ROOT}'))
N_IMAGES = 548
ITERATIONS = 10000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
BATCH = 64 if DEVICE.type == "cuda" else 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_slug(model: str) -> str:
    return {
        "Meta-AP+APS-B6": "meta_ap_aps_b6", "SourceLocked-Stage6-B6": "source_locked_stage6_b6",
        "MatchedManual-B6": "matched_manual_b6", "Meta-AP-only-B6": "meta_ap_only_b6",
        "RandomSearch-B6": "random_search_b6",
    }[model]


def prediction_roots(metric: dict[str, Any]) -> tuple[Path, Path]:
    base = EVAL / "predictions" / model_slug(metric["model"]) / f"seed{metric['seed']}"
    return base / "visdrone_txt", base / "coco_records"


def load_yolo_rows(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.empty((0, 6), dtype=np.float64)
    return np.loadtxt(path, dtype=np.float64, ndmin=2)


def xywhn_to_xyxy(rows: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.empty((len(rows), 4), dtype=np.float64)
    out[:, 0] = (rows[:, 1] - rows[:, 3] / 2.0) * width
    out[:, 1] = (rows[:, 2] - rows[:, 4] / 2.0) * height
    out[:, 2] = (rows[:, 1] + rows[:, 3] / 2.0) * width
    out[:, 3] = (rows[:, 2] + rows[:, 4] / 2.0) * height
    return out


def build_cache(metric: dict[str, Any], val_manifest: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    if cache_path.is_file():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    port = load_module("stage11_stats_port", PORT_PATH)
    if str(YOLO_ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(YOLO_ROOT))
    import val

    official_dir, coco_dir = prediction_roots(metric)
    raw_annotations = Path(val_manifest["raw_annotation_root"])
    raw_images = Path(val_manifest["raw_image_root"])
    all_gt, all_det, names = port.prepare_records(raw_annotations, official_dir, raw_images)
    expected_names = sorted(item["image_name"].replace(".jpg", ".txt") for item in val_manifest["records"])
    if names != expected_names:
        raise RuntimeError("VisDrone evaluator image ordering mismatch")

    cache: dict[str, Any] = {
        "schema": "stage11_metric_resampling_cache_v1", "run_id": metric["run_id"],
        "model": metric["model"], "seed": metric["seed"], "official": [], "coco_small": [], "yolo": [],
    }
    thresholds = np.asarray(port.THRESHOLDS, dtype=np.float64)
    for class_id in port.CLASS_IDS:
        gt_counts = np.zeros(N_IMAGES, dtype=np.int32)
        scores_parts: list[np.ndarray] = []
        image_parts: list[np.ndarray] = []
        outcome_parts: list[list[np.ndarray]] = [[] for _ in thresholds]
        for image_index, (gt, det) in enumerate(zip(all_gt, all_det)):
            class_gt = gt[gt[:, 5] == class_id, :5]
            limited = det[: min(len(det), 500)]
            class_det = limited[limited[:, 5] == class_id, :5]
            gt_counts[image_index] = len(class_gt)
            scores_parts.append(class_det[:, 4].copy())
            image_parts.append(np.full(len(class_det), image_index, dtype=np.int16))
            for threshold_index, threshold in enumerate(thresholds):
                _, matched = port.eval_result(class_gt, class_det, float(threshold))
                outcome_parts[threshold_index].append(matched[:, 5].astype(np.int8))
        scores = np.concatenate(scores_parts) if scores_parts else np.empty(0, dtype=np.float64)
        images = np.concatenate(image_parts) if image_parts else np.empty(0, dtype=np.int16)
        outcomes = np.stack([np.concatenate(parts) for parts in outcome_parts], axis=0)
        order = np.argsort(-scores, kind="stable")
        cache["official"].append({
            "scores": scores[order], "images": images[order], "outcomes": outcomes[:, order],
            "tie_order": np.arange(len(scores), dtype=np.int32)[order],
            "gt_counts": gt_counts, "class_presence": (gt_counts > 0).astype(np.int8),
        })

    annotation_path = coco_dir / "inner_val1294_coco_annotations.json"
    prediction_path = coco_dir / "inner_val1294_coco_predictions.json"
    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(str(prediction_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = list(range(1, N_IMAGES + 1))
    evaluator.params.catIds = list(range(1, 11))
    evaluator.params.maxDets = [1, 10, 500]
    evaluator.params.useCats = 1
    evaluator.evaluate()
    area_index = evaluator.params.areaRngLbl.index("small")
    area_count = len(evaluator.params.areaRng)
    for category_index in range(10):
        entries = [evaluator.evalImgs[category_index * area_count * N_IMAGES + area_index * N_IMAGES + image_index] for image_index in range(N_IMAGES)]
        gt_counts = np.asarray([
            int(np.count_nonzero(~np.asarray(entry["gtIgnore"], dtype=bool))) if entry is not None else 0
            for entry in entries
        ], dtype=np.int32)
        score_parts: list[np.ndarray] = []
        image_parts: list[np.ndarray] = []
        tp_parts: list[np.ndarray] = []
        fp_parts: list[np.ndarray] = []
        for image_index, entry in enumerate(entries):
            if entry is None:
                score_parts.append(np.empty(0, dtype=np.float64))
                image_parts.append(np.empty(0, dtype=np.int16))
                tp_parts.append(np.empty((len(evaluator.params.iouThrs), 0), dtype=np.int8))
                fp_parts.append(np.empty((len(evaluator.params.iouThrs), 0), dtype=np.int8))
                continue
            scores = np.asarray(entry["dtScores"], dtype=np.float64)
            matches = np.asarray(entry["dtMatches"], dtype=np.float64)
            ignored = np.asarray(entry["dtIgnore"], dtype=bool)
            score_parts.append(scores)
            image_parts.append(np.full(len(scores), image_index, dtype=np.int16))
            tp_parts.append(((matches > 0) & ~ignored).astype(np.int8))
            fp_parts.append(((matches == 0) & ~ignored).astype(np.int8))
        scores = np.concatenate(score_parts)
        images = np.concatenate(image_parts)
        tp = np.concatenate(tp_parts, axis=1)
        fp = np.concatenate(fp_parts, axis=1)
        order = np.argsort(-scores, kind="stable")
        cache["coco_small"].append({
            "scores": scores[order], "images": images[order], "tp": tp[:, order], "fp": fp[:, order],
            "tie_order": np.arange(len(scores), dtype=np.int32)[order], "gt_counts": gt_counts,
        })

    records_by_name = {item["image_name"]: item for item in val_manifest["records"]}
    full_precision_dir = EVAL / "predictions" / model_slug(metric["model"]) / f"seed{metric['seed']}" / "full_precision_post_nms"
    if not (full_precision_dir / ".complete").is_file():
        raise RuntimeError(f"Full-precision prediction capture is missing: {full_precision_dir}")
    target_counts = np.zeros((10, N_IMAGES), dtype=np.int32)
    yolo_parts = [{"scores": [], "images": [], "tp": []} for _ in range(10)]
    iouv = torch.linspace(0.5, 0.95, 10)
    for image_index, image_name in enumerate(sorted(records_by_name)):
        record = records_by_name[image_name]
        width, height = int(record["width"]), int(record["height"])
        captured = np.load(full_precision_dir / f"{Path(image_name).stem}.npz")
        detections = np.asarray(captured["detections"])
        labels = np.asarray(captured["labels"])
        correct = np.asarray(captured["correct"], dtype=bool)
        for class_id in range(10):
            target_counts[class_id, image_index] = int(np.count_nonzero(labels[:, 0] == class_id))
            mask = detections[:, 5] == class_id if len(detections) else np.zeros(0, dtype=bool)
            yolo_parts[class_id]["scores"].append(detections[mask, 4] if len(detections) else np.empty(0))
            yolo_parts[class_id]["images"].append(np.full(int(np.count_nonzero(mask)), image_index, dtype=np.int16))
            yolo_parts[class_id]["tp"].append(correct[mask, 0].astype(np.int8) if len(detections) else np.empty(0, dtype=np.int8))
    for class_id in range(10):
        scores = np.concatenate(yolo_parts[class_id]["scores"])
        images = np.concatenate(yolo_parts[class_id]["images"])
        tp = np.concatenate(yolo_parts[class_id]["tp"])
        original_order = np.arange(len(scores), dtype=np.int32)
        order = np.argsort(-scores, kind="stable")
        scores, images, tp = scores[order], images[order], tp[order]
        group_ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True]).astype(np.int32)
        cache["yolo"].append({
            "scores": scores, "images": images, "tp": tp, "target_counts": target_counts[class_id],
            "tie_order": original_order[order],
            "group_ends": group_ends, "unique_scores": scores[group_ends],
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=5)
    return cache


def _row_weights(matrix: torch.Tensor, images: np.ndarray, sources: np.ndarray | None, orientation: int) -> torch.Tensor:
    indexes = torch.as_tensor(images.astype(np.int64), device=DEVICE)
    selected = matrix[:, indexes]
    if sources is None:
        return selected
    source = torch.as_tensor(sources.astype(np.int64), device=DEVICE)[None, :]
    if orientation == 0:
        return torch.where(source == 0, 1.0 - selected, selected)
    return torch.where(source == 0, selected, 1.0 - selected)


def official_distribution(cache: dict[str, Any], matrix_np: np.ndarray, orientation: int = 0) -> np.ndarray:
    outputs = np.zeros(len(matrix_np), dtype=np.float64)
    sources = cache.get("hybrid_sources_official")
    for start in range(0, len(matrix_np), BATCH):
        matrix = torch.as_tensor(matrix_np[start : start + BATCH], dtype=DTYPE, device=DEVICE)
        class_sum = torch.zeros(len(matrix), dtype=DTYPE, device=DEVICE)
        class_weight_sum = torch.zeros(len(matrix), dtype=DTYPE, device=DEVICE)
        for class_index, section in enumerate(cache["official"]):
            src = sources[class_index] if sources is not None else None
            row_weights = _row_weights(matrix, section["images"], src, orientation)
            if sources is None:
                gt = torch.as_tensor(section["gt_counts"], dtype=DTYPE, device=DEVICE)
                denominator = matrix @ gt
                class_weight = matrix @ torch.as_tensor(section["class_presence"], dtype=DTYPE, device=DEVICE)
            else:
                denominator = torch.full((len(matrix),), float(np.sum(section["gt_counts"])), dtype=DTYPE, device=DEVICE)
                class_weight = torch.full((len(matrix),), float(np.sum(section["class_presence"])), dtype=DTYPE, device=DEVICE)
            outcomes = torch.as_tensor(section["outcomes"], device=DEVICE)
            threshold_sum = torch.zeros(len(matrix), dtype=DTYPE, device=DEVICE)
            for threshold_index in range(outcomes.shape[0]):
                current = outcomes[threshold_index]
                tp = torch.cumsum(row_weights * (current[None, :] == 1), dim=1)
                fp = torch.cumsum(row_weights * (current[None, :] == 0), dim=1)
                recall = tp / denominator[:, None].clamp_min(1.0)
                precision = tp / (tp + fp).clamp_min(torch.finfo(DTYPE).eps)
                envelope = torch.flip(torch.cummax(torch.flip(precision, dims=[1]), dim=1).values, dims=[1])
                increments = torch.cat((recall[:, :1], recall[:, 1:] - recall[:, :-1]), dim=1)
                threshold_sum += torch.sum(increments * envelope, dim=1)
            class_sum += (threshold_sum / outcomes.shape[0]) * class_weight
            class_weight_sum += class_weight
        outputs[start : start + len(matrix)] = (class_sum / class_weight_sum.clamp_min(1)).detach().cpu().numpy()
    return outputs


def coco_small_distribution(cache: dict[str, Any], matrix_np: np.ndarray, orientation: int = 0) -> np.ndarray:
    outputs = np.zeros(len(matrix_np), dtype=np.float64)
    sources = cache.get("hybrid_sources_coco")
    recall_thresholds = torch.linspace(0.0, 1.0, 101, dtype=DTYPE, device=DEVICE)
    for start in range(0, len(matrix_np), BATCH):
        matrix = torch.as_tensor(matrix_np[start : start + BATCH], dtype=DTYPE, device=DEVICE)
        total = torch.zeros(len(matrix), dtype=DTYPE, device=DEVICE)
        valid_count = torch.zeros(len(matrix), dtype=DTYPE, device=DEVICE)
        for class_index, section in enumerate(cache["coco_small"]):
            src = sources[class_index] if sources is not None else None
            row_weights = _row_weights(matrix, section["images"], src, orientation)
            if sources is None:
                gt = torch.as_tensor(section["gt_counts"], dtype=DTYPE, device=DEVICE)
                denominator = matrix @ gt
            else:
                denominator = torch.full((len(matrix),), float(np.sum(section["gt_counts"])), dtype=DTYPE, device=DEVICE)
            tp_mask = torch.as_tensor(section["tp"], device=DEVICE)
            fp_mask = torch.as_tensor(section["fp"], device=DEVICE)
            class_values = torch.zeros(len(matrix), dtype=DTYPE, device=DEVICE)
            for threshold_index in range(tp_mask.shape[0]):
                tp = torch.cumsum(row_weights * tp_mask[threshold_index][None, :], dim=1)
                fp = torch.cumsum(row_weights * fp_mask[threshold_index][None, :], dim=1)
                recall = tp / denominator[:, None].clamp_min(1.0)
                precision = tp / (tp + fp + np.spacing(1))
                envelope = torch.flip(torch.cummax(torch.flip(precision, dims=[1]), dim=1).values, dims=[1])
                query = recall_thresholds[None, :].expand(len(matrix), -1).contiguous()
                indices = torch.searchsorted(recall.contiguous(), query, right=False)
                safe = indices.clamp(max=max(0, recall.shape[1] - 1))
                sampled = torch.gather(envelope, 1, safe)
                sampled = torch.where(indices < recall.shape[1], sampled, torch.zeros_like(sampled))
                class_values += sampled.mean(dim=1)
            valid = denominator > 0
            total += torch.where(valid, class_values / tp_mask.shape[0], torch.zeros_like(class_values))
            valid_count += valid
        outputs[start : start + len(matrix)] = (total / valid_count.clamp_min(1)).detach().cpu().numpy()
    return outputs


def _interp_curves(values: torch.Tensor, scores: np.ndarray, left_value: float) -> torch.Tensor:
    # values are cumulative curves at unique descending confidence scores.
    xp = -scores
    queries = -np.linspace(0.0, 1.0, 1000)
    right = np.searchsorted(xp, queries, side="left")
    result = torch.empty((values.shape[0], len(queries)), dtype=DTYPE, device=DEVICE)
    low = right == 0
    high = right >= len(xp)
    middle = ~(low | high)
    if np.any(low):
        exact_first = low & np.isclose(queries, xp[0], rtol=0, atol=0)
        result[:, low] = left_value
        if np.any(exact_first):
            result[:, exact_first] = values[:, 0:1]
    if np.any(high):
        result[:, high] = values[:, -1:]
    if np.any(middle):
        r = right[middle]
        l = r - 1
        alpha = (queries[middle] - xp[l]) / (xp[r] - xp[l])
        alpha_t = torch.as_tensor(alpha, dtype=DTYPE, device=DEVICE)[None, :]
        result[:, middle] = values[:, torch.as_tensor(l, device=DEVICE)] * (1 - alpha_t) + values[:, torch.as_tensor(r, device=DEVICE)] * alpha_t
    return result


def yolo_recall_distribution(cache: dict[str, Any], matrix_np: np.ndarray, orientation: int = 0) -> np.ndarray:
    outputs = np.zeros(len(matrix_np), dtype=np.float64)
    sources = cache.get("hybrid_sources_yolo")
    for start in range(0, len(matrix_np), BATCH):
        matrix = torch.as_tensor(matrix_np[start : start + BATCH], dtype=DTYPE, device=DEVICE)
        p_curves: list[torch.Tensor] = []
        r_curves: list[torch.Tensor] = []
        for class_index, section in enumerate(cache["yolo"]):
            src = sources[class_index] if sources is not None else None
            row_weights = _row_weights(matrix, section["images"], src, orientation)
            tp_mask = torch.as_tensor(section["tp"], dtype=DTYPE, device=DEVICE)[None, :]
            tp = torch.cumsum(row_weights * tp_mask, dim=1)
            fp = torch.cumsum(row_weights * (1.0 - tp_mask), dim=1)
            ends = torch.as_tensor(section["group_ends"].astype(np.int64), device=DEVICE)
            tp, fp = tp[:, ends], fp[:, ends]
            if sources is None:
                target = matrix @ torch.as_tensor(section["target_counts"], dtype=DTYPE, device=DEVICE)
            else:
                target = torch.full((len(matrix),), float(np.sum(section["target_counts"])), dtype=DTYPE, device=DEVICE)
            recall = tp / target[:, None].clamp_min(torch.finfo(DTYPE).eps)
            precision = tp / (tp + fp).clamp_min(torch.finfo(DTYPE).eps)
            r_curves.append(_interp_curves(recall, section["unique_scores"], 0.0))
            p_curves.append(_interp_curves(precision, section["unique_scores"], 1.0))
        p = torch.stack(p_curves, dim=1)
        r = torch.stack(r_curves, dim=1)
        f1 = 2.0 * p * r / (p + r + torch.finfo(DTYPE).eps)
        mean_f1 = f1.mean(dim=1)
        smooth = F.avg_pool1d(F.pad(mean_f1[:, None, :], (50, 50), mode="replicate"), kernel_size=101, stride=1)[:, 0, :]
        index = torch.argmax(smooth, dim=1)
        mean_recall = r.mean(dim=1)
        outputs[start : start + len(matrix)] = torch.gather(mean_recall, 1, index[:, None])[:, 0].detach().cpu().numpy()
    return outputs


def evaluate_distribution(cache: dict[str, Any], matrix: np.ndarray, orientation: int = 0) -> dict[str, np.ndarray]:
    return {
        "AP": official_distribution(cache, matrix, orientation),
        "APS": coco_small_distribution(cache, matrix, orientation),
        "Recall": yolo_recall_distribution(cache, matrix, orientation),
    }


def combine_cache(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"official": [], "coco_small": [], "yolo": [], "hybrid_sources_official": [], "hybrid_sources_coco": [], "hybrid_sources_yolo": []}
    for key, source_key in (("official", "hybrid_sources_official"), ("coco_small", "hybrid_sources_coco"), ("yolo", "hybrid_sources_yolo")):
        for left, right in zip(a[key], b[key]):
            scores = np.concatenate((left["scores"], right["scores"]))
            tie_order = np.concatenate((left["tie_order"], right["tie_order"]))
            order = np.lexsort((tie_order, -scores))
            section: dict[str, Any] = {
                "scores": scores[order], "images": np.concatenate((left["images"], right["images"]))[order],
                "tie_order": tie_order[order],
            }
            if key == "official":
                section["outcomes"] = np.concatenate((left["outcomes"], right["outcomes"]), axis=1)[:, order]
                section["gt_counts"] = left["gt_counts"]
                section["class_presence"] = left["class_presence"]
            elif key == "coco_small":
                section["tp"] = np.concatenate((left["tp"], right["tp"]), axis=1)[:, order]
                section["fp"] = np.concatenate((left["fp"], right["fp"]), axis=1)[:, order]
                section["gt_counts"] = left["gt_counts"]
            else:
                section["tp"] = np.concatenate((left["tp"], right["tp"]))[order]
                section["target_counts"] = left["target_counts"]
                scores_ordered = section["scores"]
                section["group_ends"] = np.flatnonzero(np.r_[scores_ordered[1:] != scores_ordered[:-1], True]).astype(np.int32)
                section["unique_scores"] = scores_ordered[section["group_ends"]]
            sources = np.concatenate((np.zeros(len(left["scores"]), dtype=np.int8), np.ones(len(right["scores"]), dtype=np.int8)))[order]
            result[key].append(section)
            result[source_key].append(sources)
    return result


def bootstrap_counts(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, N_IMAGES, size=(ITERATIONS, N_IMAGES), dtype=np.int16)
    counts = np.zeros((ITERATIONS, N_IMAGES), dtype=np.float64)
    for index in range(ITERATIONS):
        counts[index] = np.bincount(draws[index], minlength=N_IMAGES)
    return counts


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def holm(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in sorted({record["family"] for record in records}):
        group = sorted([dict(record) for record in records if record["family"] == family], key=lambda item: (item["raw_p_value"], item["hypothesis_id"]))
        running = 0.0
        size = len(group)
        for rank, item in enumerate(group, 1):
            adjusted = min(1.0, (size - rank + 1) * item["raw_p_value"])
            running = max(running, adjusted)
            item["holm_rank"] = rank
            item["holm_adjusted_p_value"] = running
            item["reject_alpha_0_05"] = running < 0.05
            output.append(item)
    return output


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError(f"Refusing empty CSV: {path}")
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in record.items()})


def main() -> int:
    if not (EVAL / ".complete").is_file():
        raise RuntimeError("Stage11 val548 evaluation is not complete")
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT / "resampling_caches"
    distribution_dir = OUT / "distributions"
    cache_dir.mkdir(exist_ok=True)
    distribution_dir.mkdir(exist_ok=True)
    lock = read_json(EVAL / "stage11_execution_lock.json")
    all_metrics = read_json(EVAL / "per_checkpoint_metrics.json")["records"]
    val_manifest = read_json(EVAL / "val548_manifest.json")
    metric_by_model_seed = {(item["model"], int(item["seed"])): item for item in all_metrics}
    caches: dict[tuple[str, int], dict[str, Any]] = {}
    cache_audit: list[dict[str, Any]] = []
    ones = np.ones((1, N_IMAGES), dtype=np.float64)
    for metric in all_metrics:
        key = (metric["model"], int(metric["seed"]))
        path = cache_dir / f"{model_slug(metric['model'])}_seed{metric['seed']}.pkl"
        cache = build_cache(metric, val_manifest, path)
        point = evaluate_distribution(cache, ones)
        differences = {name: float(point[name][0] - float(metric[name])) for name in ("AP", "APS", "Recall")}
        if any(abs(value) > 2e-8 for value in differences.values()):
            raise RuntimeError(f"Resampling cache point parity failed for {key}: {differences}")
        caches[key] = cache
        cache_audit.append({"model": key[0], "seed": key[1], "cache_path": str(path), "cache_sha256": sha256(path), "point_differences": differences, "status": "PASS"})
    write_json(OUT / "resampling_cache_audit.json", {"records": cache_audit, "status": "PASS"})

    counts_per_seed = {seed: bootstrap_counts(42042 + seed) for seed in (42, 43, 44)}
    counts_aggregate = bootstrap_counts(42042)
    masks = np.random.default_rng(52042).integers(0, 2, size=(ITERATIONS, N_IMAGES), dtype=np.int8).astype(np.float64)
    np.savez_compressed(OUT / "resampling_rng_draws.npz", per_seed_42=counts_per_seed[42], per_seed_43=counts_per_seed[43], per_seed_44=counts_per_seed[44], aggregate=counts_aggregate, randomization_masks=masks)

    per_seed_dist: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    aggregate_dist: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key, cache in caches.items():
        per_seed_dist[key] = evaluate_distribution(cache, counts_per_seed[key[1]])
        aggregate_dist[key] = evaluate_distribution(cache, counts_aggregate)
        np.savez_compressed(distribution_dir / f"standard_{model_slug(key[0])}_seed{key[1]}.npz", **{f"per_seed_{name}": values for name, values in per_seed_dist[key].items()}, **{f"aggregate_{name}": values for name, values in aggregate_dist[key].items()})

    paired_records: list[dict[str, Any]] = []
    paired_summary: list[dict[str, Any]] = []
    per_seed_bootstrap: list[dict[str, Any]] = []
    aggregate_bootstrap: list[dict[str, Any]] = []
    randomization_records: list[dict[str, Any]] = []
    random_distributions: dict[str, np.ndarray] = {}
    all_point_metrics = ["AP", "AP50", "AP75", "APS", "APM", "APL", "Precision", "Recall", "F1"]
    for comparison in lock["comparisons"]:
        cid, model_a, model_b = comparison["comparison_id"], comparison["model_a"], comparison["model_b"]
        for metric_name in all_point_metrics:
            differences = []
            for seed in (42, 43, 44):
                a = float(metric_by_model_seed[(model_a, seed)][metric_name])
                b = float(metric_by_model_seed[(model_b, seed)][metric_name])
                difference = a - b
                differences.append(difference)
                paired_records.append({"comparison_id": cid, "model_a": model_a, "model_b": model_b, "metric": metric_name, "seed": seed, "model_a_value": a, "model_b_value": b, "paired_difference": difference})
            paired_summary.append({"comparison_id": cid, "metric": metric_name, "seed_differences": differences, "mean_paired_difference": statistics.mean(differences), "sample_sd": statistics.stdev(differences), "minimum": min(differences), "maximum": max(differences), "sign_consistency": all(value > 0 for value in differences) or all(value < 0 for value in differences) or all(value == 0 for value in differences)})
        for seed in (42, 43, 44):
            for metric_name in ("AP", "APS", "Recall"):
                distribution = per_seed_dist[(model_a, seed)][metric_name] - per_seed_dist[(model_b, seed)][metric_name]
                low, high = percentile_ci(distribution)
                per_seed_bootstrap.append({"comparison_id": cid, "seed": seed, "metric": metric_name, "point_estimate": float(metric_by_model_seed[(model_a, seed)][metric_name]) - float(metric_by_model_seed[(model_b, seed)][metric_name]), "ci95_low": low, "ci95_high": high, "iterations": ITERATIONS, "rng_seed": 42042 + seed})
        for metric_name in ("AP", "APS", "Recall"):
            distribution = np.mean(np.stack([aggregate_dist[(model_a, seed)][metric_name] - aggregate_dist[(model_b, seed)][metric_name] for seed in (42, 43, 44)]), axis=0)
            low, high = percentile_ci(distribution)
            point = statistics.mean([float(metric_by_model_seed[(model_a, seed)][metric_name]) - float(metric_by_model_seed[(model_b, seed)][metric_name]) for seed in (42, 43, 44)])
            aggregate_bootstrap.append({"comparison_id": cid, "metric": metric_name, "aggregate_mean_point_estimate": point, "ci95_low": low, "ci95_high": high, "iterations": ITERATIONS, "rng_seed": 42042, "interpretation": "Image-sampling uncertainty conditional on the three fixed checkpoints"})
        for seed in (42, 43, 44):
            hybrid = combine_cache(caches[(model_a, seed)], caches[(model_b, seed)])
            dist_a = evaluate_distribution(hybrid, masks, orientation=0)
            dist_b = evaluate_distribution(hybrid, masks, orientation=1)
            for metric_name in ("AP", "APS"):
                random_distributions[f"{cid}_seed{seed}_{metric_name}"] = dist_a[metric_name] - dist_b[metric_name]
        for metric_name in ("AP", "APS"):
            distribution = np.mean(np.stack([random_distributions[f"{cid}_seed{seed}_{metric_name}"] for seed in (42, 43, 44)]), axis=0)
            observed = statistics.mean([float(metric_by_model_seed[(model_a, seed)][metric_name]) - float(metric_by_model_seed[(model_b, seed)][metric_name]) for seed in (42, 43, 44)])
            p_value = (1 + int(np.count_nonzero(np.abs(distribution) >= abs(observed)))) / (ITERATIONS + 1)
            randomization_records.append({"hypothesis_id": f"{cid}:{metric_name}", "comparison_id": cid, "family": comparison["family"], "metric": metric_name, "observed_mean_paired_difference": observed, "raw_p_value": p_value, "iterations": ITERATIONS, "rng_seed": 52042, "two_sided": True})

    holm_records = holm(randomization_records)
    np.savez_compressed(distribution_dir / "randomization_distributions.npz", **random_distributions)
    write_csv(OUT / "paired_seed_differences.csv", paired_records + paired_summary)
    write_json(OUT / "paired_seed_differences.json", {"per_seed_records": paired_records, "summaries": paired_summary, "conventional_n3_p_value": False})
    write_csv(OUT / "per_seed_bootstrap_results.csv", per_seed_bootstrap)
    write_json(OUT / "per_seed_bootstrap_results.json", {"records": per_seed_bootstrap, "status": "PASS"})
    write_csv(OUT / "aggregate_bootstrap_results.csv", aggregate_bootstrap)
    write_json(OUT / "aggregate_bootstrap_results.json", {"records": aggregate_bootstrap, "status": "PASS"})
    write_csv(OUT / "randomization_results.csv", randomization_records)
    write_json(OUT / "randomization_results.json", {"records": randomization_records, "status": "PASS"})
    write_csv(OUT / "holm_results.csv", holm_records)
    write_json(OUT / "holm_results.json", {"records": holm_records, "alpha": 0.05, "status": "PASS"})

    criteria: list[dict[str, Any]] = []
    aggregate_by_key = {(item["comparison_id"], item["metric"]): item for item in aggregate_bootstrap}
    holm_by_id = {item["hypothesis_id"]: item for item in holm_records}
    for comparison in lock["comparisons"]:
        cid = comparison["comparison_id"]
        ap = aggregate_by_key[(cid, "AP")]["aggregate_mean_point_estimate"]
        aps = aggregate_by_key[(cid, "APS")]["aggregate_mean_point_estimate"]
        recall = aggregate_by_key[(cid, "Recall")]["aggregate_mean_point_estimate"]
        a = metric_by_model_seed[(comparison["model_a"], 42)]
        b = metric_by_model_seed[(comparison["model_b"], 42)]
        criteria.append({
            "comparison_id": cid, "model_a": comparison["model_a"], "model_b": comparison["model_b"],
            "mean_delta_AP": ap, "mean_delta_APS": aps, "mean_delta_Recall": recall,
            "basic_AP_improvement_met": ap >= 0.002, "strong_AP_improvement_met": ap >= 0.003,
            "APS_improvement_met": aps >= 0.003, "recall_noninferiority_met": recall >= -0.002,
            "AP_ci95": [aggregate_by_key[(cid, "AP")]["ci95_low"], aggregate_by_key[(cid, "AP")]["ci95_high"]],
            "APS_ci95": [aggregate_by_key[(cid, "APS")]["ci95_low"], aggregate_by_key[(cid, "APS")]["ci95_high"]],
            "Recall_ci95": [aggregate_by_key[(cid, "Recall")]["ci95_low"], aggregate_by_key[(cid, "Recall")]["ci95_high"]],
            "AP_raw_p": holm_by_id[f"{cid}:AP"]["raw_p_value"], "AP_holm_p": holm_by_id[f"{cid}:AP"]["holm_adjusted_p_value"],
            "APS_raw_p": holm_by_id[f"{cid}:APS"]["raw_p_value"], "APS_holm_p": holm_by_id[f"{cid}:APS"]["holm_adjusted_p_value"],
            "params_difference_model_a_minus_b": int(a["Params"]) - int(b["Params"]),
            "flops_raw_difference_model_a_minus_b": int(a["FLOPs_raw"]) - int(b["FLOPs_raw"]),
        })
    write_json(OUT / "success_criteria.json", {"thresholds": {"recall_noninferiority": -0.002, "basic_AP": 0.002, "strong_AP": 0.003, "APS": 0.003}, "records": criteria, "status": "PASS"})
    validation = {
        "schema": "stage11_statistical_validation_v1", "created_utc": utc_now(), "status": "PASS",
        "checks": {"paired_seeds_preserved": True, "per_seed_bootstrap_10000": True, "aggregate_bootstrap_10000": True, "randomization_10000": True, "same_image_indices_within_pairs": True, "same_randomization_masks_across_seeds": True, "holm_families_locked": True, "conventional_n3_p_value": False, "cache_point_parity": True, "heldout1610_accessed": False},
        "device": str(DEVICE), "numeric_dtype": str(DTYPE),
    }
    write_json(OUT / "statistical_validation.json", validation)
    (OUT / ".complete").write_text("STAGE11_STATISTICS=PASS\nHELDOUT1610_ACCESSED=NO\n", encoding="ascii")
    print(json.dumps({"status": "STATISTICS_PASS", "bootstrap_iterations": ITERATIONS, "randomization_iterations": ITERATIONS, "holm_tests": len(holm_records), "heldout1610_accessed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
