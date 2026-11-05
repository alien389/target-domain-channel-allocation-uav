from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import argparse
import copy
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


YOLO = Path(_release_paths.expand('${YOLOV5_ROOT}'))
SOURCE_IMPLEMENTATION = Path(
    _release_paths.expand('${CODE_ROOT}\\arch modify\\09_ptq_tensorrt\\pareto_final_quantization\\scripts\\build_modelopt_qdq_ptq_engine.py')
)
PROTECTED_PATTERNS = [
    "model.0.conv.*",
    "model.17.cv3.conv.output_quantizer",
    "model.20.cv3.conv.output_quantizer",
    "model.23.cv3.conv.output_quantizer",
    "model.24.m.*",
]

if str(YOLO) not in sys.path:
    sys.path.insert(0, str(YOLO))

from models.experimental import attempt_load  # noqa: E402
from models.yolo import Detect  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
import modelopt.torch.quantization as mtq  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest().upper()


def preprocess(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image = letterbox(image_bgr, new_shape=640, stride=32, auto=False)[0]
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--calibration-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()

    if any(path.exists() for path in (args.output, args.metadata)):
        raise FileExistsError("Refusing to overwrite a Stage-6 QDQ repair artifact")
    if sha256(args.weights) != args.checkpoint_sha256.upper():
        raise RuntimeError("Stage-6 checkpoint SHA256 mismatch")
    calibration_paths = [
        Path(line.strip())
        for line in args.calibration_list.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(calibration_paths) != 1024 or len(set(calibration_paths)) != 1024:
        raise RuntimeError("Calibration list must contain exactly 1024 unique images")
    if any(not path.is_file() for path in calibration_paths):
        raise FileNotFoundError("At least one calibration image is missing")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = attempt_load(str(args.weights), device="cpu", inplace=True, fuse=False).float().eval()
    detect_modules = [module for module in model.modules() if isinstance(module, Detect)]
    if len(detect_modules) != 1:
        raise RuntimeError(f"Expected one Detect module, found {len(detect_modules)}")
    detect = detect_modules[0]
    expected = detect.anchors.detach().float() * detect.stride.detach().float().view(-1, 1, 1)
    cached_before = [grid.detach().float()[0, :, 0, 0, :].tolist() for grid in detect.anchor_grid]
    mismatch_before = any(
        not torch.allclose(detect.anchor_grid[index].detach().float()[0, :, 0, 0, :], expected[index], atol=1e-5, rtol=0.0)
        for index in range(detect.nl)
    )
    if not mismatch_before:
        raise RuntimeError("Stage-6 QDQ cache is already coherent; repair premise is absent")

    state_before = state_digest(model)
    detect.grid = [torch.empty(0) for _ in range(detect.nl)]
    detect.anchor_grid = [torch.empty(0) for _ in range(detect.nl)]
    detect.export = True
    detect.dynamic = False
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    cached_after = [grid.detach().float()[0, :, 0, 0, :].tolist() for grid in detect.anchor_grid]
    coherent_after = all(
        torch.allclose(detect.anchor_grid[index].detach().float()[0, :, 0, 0, :], expected[index], atol=1e-5, rtol=0.0)
        for index in range(detect.nl)
    )
    state_after = state_digest(model)
    if not coherent_after or state_before != state_after:
        raise RuntimeError("Grid refresh changed model state or failed anchor coherence")

    quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    for pattern in PROTECTED_PATTERNS:
        quant_cfg["quant_cfg"][pattern] = {"enable": False}
    processed = [0]

    def calibration_forward_loop(module: torch.nn.Module) -> None:
        with torch.no_grad():
            for image_path in calibration_paths:
                module(torch.from_numpy(preprocess(image_path)[None]))
                processed[0] += 1

    mtq.quantize(model, quant_cfg, forward_loop=calibration_forward_loop)
    if processed[0] != 1024:
        raise RuntimeError("Calibration did not process exactly 1024 images")

    total_quantizers = 0
    enabled_quantizers = []
    disabled_quantizers = []
    pattern_matches = {pattern: [] for pattern in PROTECTED_PATTERNS}
    for name, module in model.named_modules():
        if isinstance(module, mtq.nn.TensorQuantizer):
            total_quantizers += 1
            (enabled_quantizers if getattr(module, "is_enabled", False) else disabled_quantizers).append(name)
            for pattern in PROTECTED_PATTERNS:
                if fnmatch.fnmatchcase(name, pattern):
                    pattern_matches[pattern].append(name)
    pattern_counts = [len(pattern_matches[pattern]) for pattern in PROTECTED_PATTERNS]
    if total_quantizers != 284 or pattern_counts != [3, 1, 1, 1, 9]:
        raise RuntimeError("Selective quantizer mapping changed during Stage-6 repair")
    if len(enabled_quantizers) != 113 or len(disabled_quantizers) != 171:
        raise RuntimeError("Selective quantizer enable/disable counts changed")

    torch.onnx.export(
        model,
        torch.zeros(1, 3, 640, 640),
        str(args.output),
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        dynamo=False,
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    counts = Counter(node.op_type for node in graph.graph.node)
    input_shape = shape_of(graph.graph.input[0])
    output_shape = shape_of(graph.graph.output[0])
    if input_shape != [1, 3, 640, 640] or output_shape != [1, 25200, 15]:
        raise RuntimeError("Corrected Stage-6 QDQ ONNX contract mismatch")
    if counts["QuantizeLinear"] != 115 or counts["DequantizeLinear"] != 115:
        raise RuntimeError("Corrected Stage-6 QDQ node counts changed")

    metadata = {
        "schema": "stage12_stage6_selective_qdq_grid_refresh_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scientific_change": "NONE_EXPORT_CACHE_REPAIR_ONLY",
        "model_id": "source_locked_stage6_b6",
        "source_checkpoint": str(args.weights),
        "source_checkpoint_sha256": sha256(args.weights),
        "source_implementation": str(SOURCE_IMPLEMENTATION),
        "source_implementation_sha256": sha256(SOURCE_IMPLEMENTATION),
        "state_dict_sha256_before": state_before,
        "state_dict_sha256_after": state_after,
        "state_dict_unchanged": state_before == state_after,
        "expected_anchor_pixels": expected.tolist(),
        "cached_anchor_pixels_before": cached_before,
        "cached_anchor_pixels_after": cached_after,
        "anchor_grid_coherent_after": coherent_after,
        "quant_cfg": "INT8_DEFAULT_CFG",
        "protected_patterns": PROTECTED_PATTERNS,
        "pattern_matches": pattern_matches,
        "total_quantizers": total_quantizers,
        "enabled_quantizers": len(enabled_quantizers),
        "disabled_quantizers": len(disabled_quantizers),
        "disabled_quantizer_names": disabled_quantizers,
        "calibration_list": str(args.calibration_list),
        "calibration_list_sha256": sha256(args.calibration_list),
        "calibration_images": processed[0],
        "calibration_split": "VisDrone training only",
        "calibration_batch_size": 1,
        "imgsz": 640,
        "letterbox_stride": 32,
        "letterbox_auto": False,
        "opset": 17,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "quantize_linear_nodes": counts["QuantizeLinear"],
        "dequantize_linear_nodes": counts["DequantizeLinear"],
        "onnx_path": str(args.output),
        "onnx_sha256": sha256(args.output),
        "onnx_size_bytes": args.output.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("STAGE6_SELECTIVE_QDQ_GRID_REFRESH=PASS")
    print(f"QDQ_ONNX_SHA256={metadata['onnx_sha256']}")


if __name__ == "__main__":
    main()
