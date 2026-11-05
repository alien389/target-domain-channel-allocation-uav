from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import copy
import csv
import fnmatch
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE = PROJECT / "12_jetson_fp16_deployment"
YOLO = Path(_release_paths.expand('${YOLOV5_ROOT}'))
SOURCE_IMPLEMENTATION = Path(
    _release_paths.expand('${CODE_ROOT}\\arch modify\\09_ptq_tensorrt\\pareto_final_quantization\\scripts\\build_modelopt_qdq_ptq_engine.py')
)
CAL_LIST = Path(_release_paths.expand('${PAPER1_ROOT}\\deployment_drone_laptop_raspberry\\10_part1d_int8_build\\01_calibration_selection\\visdrone_train_calibration_1024_seed42_drone.txt'))
LOCK = json.loads((STAGE / "10_manifests/locked_model_manifest.json").read_text(encoding="utf-8"))
OUT = STAGE / "03_onnx/selective_qdq"
LOG = STAGE / "09_logs/selective_qdq"
MANIFEST_DIR = STAGE / "10_manifests"
PROTECTED_PATTERNS = [
    "model.0.conv.*",
    "model.17.cv3.conv.output_quantizer",
    "model.20.cv3.conv.output_quantizer",
    "model.23.cv3.conv.output_quantizer",
    "model.24.m.*",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def preprocess(image_path: Path, letterbox) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image = letterbox(image_bgr, new_shape=640, stride=32, auto=False)[0]
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32)
    image /= 255.0
    return image


def shape_of(value_info: onnx.ValueInfoProto) -> list[int | str]:
    result = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append("?")
    return result


if str(YOLO) not in sys.path:
    sys.path.insert(0, str(YOLO))
from models.experimental import attempt_load  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
import modelopt.torch.quantization as mtq  # noqa: E402


calibration_paths = [Path(line.strip()) for line in CAL_LIST.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
if len(calibration_paths) != 1024 or len(set(calibration_paths)) != 1024:
    raise RuntimeError("Calibration list must contain exactly 1024 unique paths")
if any(not path.is_file() for path in calibration_paths):
    raise FileNotFoundError("At least one calibration image is missing")

OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
rows = []
for model_record in LOCK["models"]:
    model_id = model_record["model_id"]
    weights = STAGE / "02_locked_inputs/checkpoints" / f"{model_id}_seed42_terminal_ema.pt"
    output = OUT / f"{model_id}_seed42_selective_qdq.onnx"
    metadata_path = OUT / f"{model_id}_seed42_selective_qdq_metadata.json"
    if output.exists() or metadata_path.exists():
        raise RuntimeError(f"Refusing to overwrite Selective QDQ artifact: {model_id}")
    if sha256(weights) != model_record["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch before QDQ: {model_id}")

    start = time.time()
    model = attempt_load(str(weights), device="cpu", inplace=True, fuse=False)
    model.eval()
    model.model[-1].export = True

    quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    for pattern in PROTECTED_PATTERNS:
        quant_cfg["quant_cfg"][pattern] = {"enable": False}

    processed = [0]

    def calibration_forward_loop(module: torch.nn.Module) -> None:
        with torch.no_grad():
            for image_path in calibration_paths:
                tensor = torch.from_numpy(preprocess(image_path, letterbox)[None])
                module(tensor)
                processed[0] += 1

    mtq.quantize(model, quant_cfg, forward_loop=calibration_forward_loop)
    if processed[0] != 1024:
        raise RuntimeError(f"Calibration count mismatch for {model_id}: {processed[0]}")

    total_quantizers = 0
    enabled_quantizers = []
    disabled_quantizers = []
    pattern_matches = {pattern: [] for pattern in PROTECTED_PATTERNS}
    for name, module in model.named_modules():
        if isinstance(module, mtq.nn.TensorQuantizer):
            total_quantizers += 1
            if getattr(module, "is_enabled", False):
                enabled_quantizers.append(name)
            else:
                disabled_quantizers.append(name)
            for pattern in PROTECTED_PATTERNS:
                if fnmatch.fnmatchcase(name, pattern):
                    pattern_matches[pattern].append(name)
    expected_pattern_counts = [3, 1, 1, 1, 9]
    actual_pattern_counts = [len(pattern_matches[pattern]) for pattern in PROTECTED_PATTERNS]
    if (
        total_quantizers != 284
        or actual_pattern_counts != expected_pattern_counts
        or len(enabled_quantizers) != 113
        or len(disabled_quantizers) != 171
    ):
        raise RuntimeError(
            f"Selective mapping mismatch for {model_id}: total={total_quantizers}, "
            f"patterns={actual_pattern_counts}, enabled={len(enabled_quantizers)}, "
            f"disabled={len(disabled_quantizers)}"
        )

    dummy = torch.zeros(1, 3, 640, 640)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        dynamo=False,
    )
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    counts = Counter(node.op_type for node in graph.graph.node)
    input_shape = shape_of(graph.graph.input[0])
    output_shape = shape_of(graph.graph.output[0])
    if input_shape != [1, 3, 640, 640] or output_shape != [1, 25200, 15]:
        raise RuntimeError(f"QDQ ONNX shape mismatch for {model_id}: {input_shape}, {output_shape}")
    if counts["QuantizeLinear"] == 0 or counts["DequantizeLinear"] == 0:
        raise RuntimeError(f"QDQ nodes missing for {model_id}")
    if counts["QuantizeLinear"] != 115 or counts["DequantizeLinear"] != 115:
        raise RuntimeError(
            f"QDQ node-count mismatch for {model_id}: "
            f"Q={counts['QuantizeLinear']}, DQ={counts['DequantizeLinear']}"
        )
    if any("NMS" in node.op_type.upper() or "NONMAX" in node.op_type.upper() for node in graph.graph.node):
        raise RuntimeError(f"Unexpected NMS in QDQ graph: {model_id}")

    metadata = {
        "schema": "meta_jetson_selective_qdq_metadata_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "paper_name": model_record["paper_name"],
        "source_checkpoint_sha256": model_record["checkpoint_sha256"],
        "source_implementation": str(SOURCE_IMPLEMENTATION),
        "source_implementation_sha256": sha256(SOURCE_IMPLEMENTATION),
        "quant_cfg": "INT8_DEFAULT_CFG",
        "protected_patterns": PROTECTED_PATTERNS,
        "pattern_matches": pattern_matches,
        "pattern_match_counts": dict(zip(PROTECTED_PATTERNS, actual_pattern_counts)),
        "total_quantizers": total_quantizers,
        "enabled_quantizers": len(enabled_quantizers),
        "disabled_quantizers": len(disabled_quantizers),
        "historical_default_disabled_quantizers": 156,
        "explicitly_protected_pattern_quantizers": 15,
        "disabled_quantizer_names": disabled_quantizers,
        "calibration_list": str(CAL_LIST),
        "calibration_list_sha256": sha256(CAL_LIST),
        "calibration_images": processed[0],
        "calibration_split": "VisDrone training only",
        "calibration_batch_size": 1,
        "imgsz": 640,
        "letterbox_stride": 32,
        "letterbox_auto": False,
        "opset": 17,
        "input_name": graph.graph.input[0].name,
        "input_shape": input_shape,
        "output_name": graph.graph.output[0].name,
        "output_shape": output_shape,
        "quantize_linear_nodes": counts["QuantizeLinear"],
        "dequantize_linear_nodes": counts["DequantizeLinear"],
        "onnx_path": _release_paths.expand(f'${{JETSON_PROJECT_ROOT}}/03_onnx/selective_qdq/{output.name}'),
        "onnx_sha256": sha256(output),
        "onnx_size_bytes": output.stat().st_size,
        "elapsed_seconds": time.time() - start,
        "status": "PASS",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (LOG / f"{model_id}.log").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "calibration_images": processed[0],
                "total_quantizers": total_quantizers,
                "disabled_quantizers": len(disabled_quantizers),
                "qdq_nodes": counts["QuantizeLinear"] + counts["DequantizeLinear"],
                "elapsed_seconds": metadata["elapsed_seconds"],
                "status": "PASS",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows.append(metadata)
    del model

csv_path = MANIFEST_DIR / "selective_qdq_manifest.csv"
json_path = MANIFEST_DIR / "selective_qdq_manifest.json"
csv_fields = [
    "model_id",
    "paper_name",
    "source_checkpoint_sha256",
    "quant_cfg",
    "calibration_list_sha256",
    "calibration_images",
    "total_quantizers",
    "enabled_quantizers",
    "disabled_quantizers",
    "quantize_linear_nodes",
    "dequantize_linear_nodes",
    "onnx_path",
    "onnx_sha256",
    "onnx_size_bytes",
    "elapsed_seconds",
    "status",
]
with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(
    json.dumps(
        {
            "schema": "meta_jetson_selective_qdq_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_count": len(rows),
            "fixed_policy": True,
            "protected_patterns": PROTECTED_PATTERNS,
            "calibration_images": 1024,
            "calibration_list_sha256": sha256(CAL_LIST),
            "models": rows,
            "status": "PASS",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"SELECTIVE_QDQ_MODELS={len(rows)}")
print(f"SELECTIVE_QDQ_MANIFEST_SHA256={sha256(json_path)}")
