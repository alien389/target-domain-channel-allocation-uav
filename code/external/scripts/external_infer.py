from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from external_evaluator import evaluate_external
from stage15_common import (
    CODE_ROOT,
    STAGE15_ROOT,
    YOLO_ROOT,
    model_slug,
    read_json,
    require,
    sha256,
    utc_now,
    write_json,
    write_text,
)


def load_manifest(dataset: str) -> tuple[list[dict[str, str]], Path, Path]:
    import csv

    if dataset == "carpk":
        prep_root = STAGE15_ROOT / "01_CARPK_PREP"
        manifest_path = prep_root / "CARPK_TEST_MANIFEST.csv"
        data_yaml = prep_root / "carpk_eval_data.yaml"
    else:
        prep_root = STAGE15_ROOT / "02_UAVDT_PREP"
        manifest_path = prep_root / "UAVDT_TEST_MANIFEST.csv"
        data_yaml = prep_root / "uavdt_eval_data.yaml"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, manifest_path, data_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["carpk", "uavdt"])
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=[42, 43, 44])
    args = parser.parse_args()
    protocol_path = STAGE15_ROOT / "STAGE15_PROTOCOL_LOCK.json"
    require(protocol_path.is_file(), "Stage15 protocol is not locked")
    protocol = read_json(protocol_path)
    require(protocol.get("status") == "LOCKED_PRE_INFERENCE", "Stage15 protocol lock status is invalid")
    checkpoint_records = protocol["checkpoints"]
    matches = [item for item in checkpoint_records if item["architecture"] == args.architecture and int(item["seed"]) == args.seed]
    require(len(matches) == 1, "Checkpoint identity is not unique in Stage15 lock")
    checkpoint_record = matches[0]
    checkpoint = Path(checkpoint_record["checkpoint_path"])
    require(sha256(checkpoint) == checkpoint_record["sha256"], "Checkpoint changed after Stage15 lock")

    rows, manifest_path, data_yaml = load_manifest(args.dataset)
    expected = 459 if args.dataset == "carpk" else 16592
    require(len(rows) == expected, f"{args.dataset} manifest count changed")
    locked_manifest = protocol["datasets"][args.dataset]["test_manifest_sha256"]
    require(sha256(manifest_path) == locked_manifest, f"{args.dataset} test manifest changed after lock")
    by_path = {str(Path(row["image_path"]).resolve()).lower(): row for row in rows}

    output_base = STAGE15_ROOT / ("03_INFERENCE_CARPK" if args.dataset == "carpk" else "04_INFERENCE_UAVDT")
    run_dir = output_base / model_slug(args.architecture) / f"seed{args.seed}"
    complete = run_dir / ".complete"
    if complete.is_file():
        metrics_path = run_dir / "metrics.json"
        require(metrics_path.is_file(), f"Incomplete committed run: {run_dir}")
        print(json.dumps({"status": "ALREADY_COMPLETE", "dataset": args.dataset, "architecture": args.architecture, "seed": args.seed}))
        return 0
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete external run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_capture = run_dir / "raw_full_precision_post_nms.jsonl"
    capture_temp = raw_capture.with_suffix(".jsonl.tmp")

    if str(YOLO_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLO_ROOT))
    import val
    from utils.callbacks import Callbacks

    allowed = {3} if args.dataset == "carpk" else {3, 5, 8}
    captured_paths: set[str] = set()
    transformations: dict[str, dict[str, Any]] = {}
    capture_handle = capture_temp.open("w", encoding="utf-8")

    def capture_image(pred, predn, path, names, im):
        normalized = str(Path(path).resolve()).lower()
        row = by_path.get(normalized)
        if row is None:
            raise RuntimeError(f"Inference emitted an image outside the locked manifest: {path}")
        detections = predn.detach().cpu().numpy()
        selected = detections[np.isin(np.rint(detections[:, 5]).astype(int), list(allowed))] if len(detections) else np.empty((0, 6))
        record = {
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "detections": [[float(value) for value in detection] for detection in selected.tolist()],
        }
        capture_handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n")
        captured_paths.add(normalized)

    def capture_batch(batch_i, im, targets, paths, shapes, preds):
        input_height, input_width = int(im.shape[2]), int(im.shape[3])
        for path, shape in zip(paths, shapes):
            normalized = str(Path(path).resolve()).lower()
            original, ratio_pad = shape
            ratio, pad = ratio_pad
            transformations[normalized] = {
                "original_shape": [int(original[0]), int(original[1])],
                "input_shape": [input_height, input_width],
                "letterbox_ratio": [float(ratio[0]), float(ratio[1])],
                "letterbox_pad": [float(pad[0]), float(pad[1])],
            }

    callbacks = Callbacks()
    callbacks.register_action("on_val_image_end", "stage15_capture_image", capture_image)
    callbacks.register_action("on_val_batch_end", "stage15_capture_batch", capture_batch)
    parameters = {
        "data": str(data_yaml),
        "weights": str(checkpoint),
        "batch_size": 16,
        "imgsz": 640,
        "conf_thres": 0.001,
        "iou_thres": 0.60,
        "max_det": 500,
        "task": "val",
        "device": "0",
        "workers": 4,
        "single_cls": False,
        "augment": False,
        "verbose": False,
        "save_txt": False,
        "save_hybrid": False,
        "save_conf": False,
        "save_json": False,
        "project": str(run_dir),
        "name": "yolov5_fp32_internal",
        "exist_ok": True,
        "half": False,
        "dnn": False,
        "plots": False,
    }
    write_json(run_dir / "config.json", parameters)
    start_utc = utc_now()
    start_time = time.perf_counter()
    try:
        val.run(**parameters, callbacks=callbacks)
    finally:
        capture_handle.flush()
        capture_handle.close()

    # val.py skips on_val_image_end for images with zero detections. Add explicit empty records.
    with capture_temp.open("a", encoding="utf-8") as handle:
        for row in rows:
            normalized = str(Path(row["image_path"]).resolve()).lower()
            if normalized not in captured_paths:
                handle.write(
                    json.dumps(
                        {"image_id": row["image_id"], "image_path": row["image_path"], "detections": []},
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                    + "\n"
                )
    capture_temp.replace(raw_capture)

    # Add exact per-image preprocessing metadata and canonicalize order.
    raw_by_id: dict[str, dict[str, Any]] = {}
    with raw_capture.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            raw_by_id[item["image_id"]] = item
    canonical_temp = raw_capture.with_suffix(".canonical.tmp")
    with canonical_temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            normalized = str(Path(row["image_path"]).resolve()).lower()
            transform = transformations.get(normalized)
            require(transform is not None, f"Missing preprocessing metadata: {row['image_id']}")
            item = raw_by_id[row["image_id"]]
            item.update(transform)
            handle.write(json.dumps(item, separators=(",", ":"), ensure_ascii=True) + "\n")
    canonical_temp.replace(raw_capture)

    metrics = evaluate_external(args.dataset, raw_capture, run_dir)
    elapsed = time.perf_counter() - start_time
    metrics.update(
        {
            "schema": "stage15_external_metrics_v1",
            "created_utc": utc_now(),
            "status": "PASS",
            "dataset": args.dataset.upper(),
            "architecture": args.architecture,
            "training_seed": args.seed,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_record["sha256"],
            "architecture_signature": checkpoint_record["architecture_signature"],
            "Params": int(checkpoint_record["params_exact"]),
            "FLOPs_raw": int(checkpoint_record["flops_raw_exact"]),
            "FLOPs_G": int(checkpoint_record["flops_raw_exact"]) / 1e9,
            "backend": "PyTorch",
            "precision": "FP32",
            "device": "0",
            "imgsz": 640,
            "batch": 16,
            "conf_threshold": 0.001,
            "nms_iou": 0.60,
            "max_det": 500,
            "tta": False,
            "multi_scale": False,
            "test_manifest_sha256": locked_manifest,
            "evaluator_version": protocol["evaluator"]["external_evaluator_sha256"],
            "start_time": start_utc,
            "end_time": utc_now(),
            "duration_seconds": elapsed,
            "zero_shot_verified": True,
            "training_executed": False,
            "checkpoint_selection_executed": False,
        }
    )
    require(all(math.isfinite(float(metrics[key])) for key in ("AP", "AP50", "AP75", "Precision", "Recall", "F1")), "Non-finite external metrics")
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "run_manifest.json",
        {
            "schema": "stage15_external_run_manifest_v1",
            "status": "PASS",
            "dataset": args.dataset,
            "architecture": args.architecture,
            "seed": args.seed,
            "checkpoint_sha256": checkpoint_record["sha256"],
            "raw_prediction_path": str(raw_capture),
            "raw_prediction_sha256": sha256(raw_capture),
            "prediction_path": str(run_dir / "predictions.jsonl"),
            "prediction_sha256": sha256(run_dir / "predictions.jsonl"),
            "metrics_path": str(run_dir / "metrics.json"),
            "metrics_sha256": sha256(run_dir / "metrics.json"),
        },
    )
    write_text(run_dir / "run.log", f"STATUS=PASS\nDATASET={args.dataset}\nARCHITECTURE={args.architecture}\nSEED={args.seed}\nDURATION_SECONDS={elapsed:.6f}\n")
    write_text(complete, "STAGE15_EXTERNAL_RUN=PASS\n")
    print(json.dumps({"status": "PASS", "dataset": args.dataset, "architecture": args.architecture, "seed": args.seed, "AP": metrics["AP"], "Recall": metrics["Recall"], "duration_seconds": elapsed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
