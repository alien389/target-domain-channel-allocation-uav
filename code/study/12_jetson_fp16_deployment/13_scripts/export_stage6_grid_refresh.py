from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch


YOLO = Path(_release_paths.expand('${YOLOV5_ROOT}'))
if str(YOLO) not in sys.path:
    sys.path.insert(0, str(YOLO))

from export import export_onnx  # noqa: E402
from models.experimental import attempt_load  # noqa: E402
from models.yolo import Detect  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402


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


def cache_audit(module: Detect) -> dict:
    expected = module.anchors.detach().float() * module.stride.detach().float().view(-1, 1, 1)
    cached = []
    matches = []
    for index, grid in enumerate(module.anchor_grid):
        if grid.numel():
            actual = grid.detach().float()[0, :, 0, 0, :]
            cached.append(actual.cpu().tolist())
            matches.append(bool(torch.allclose(actual, expected[index], rtol=0.0, atol=1e-5)))
        else:
            cached.append([])
            matches.append(False)
    return {
        "expected_anchor_pixels": expected.cpu().tolist(),
        "cached_anchor_pixels": cached,
        "per_scale_match": matches,
        "all_scales_match": all(matches),
        "grid_shapes": [list(grid.shape) for grid in module.grid],
        "anchor_grid_shapes": [list(grid.shape) for grid in module.anchor_grid],
    }


def preprocess(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = letterbox(image, new_shape=640, stride=32, auto=False)[0]
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    return image[None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--parity-image", required=True, type=Path)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite a grid-refresh repair artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = attempt_load(str(args.weights), device=device, inplace=True, fuse=True).float().eval()
    detect_modules = [module for module in model.modules() if isinstance(module, Detect)]
    if len(detect_modules) != 1:
        raise RuntimeError(f"Expected one Detect module, found {len(detect_modules)}")
    detect = detect_modules[0]
    before = cache_audit(detect)
    if before["all_scales_match"]:
        raise RuntimeError("Stage-6 cache is already coherent; the documented repair premise is absent")

    state_before = state_digest(model)
    detect.grid = [torch.empty(0, device=device) for _ in range(detect.nl)]
    detect.anchor_grid = [torch.empty(0, device=device) for _ in range(detect.nl)]
    detect.dynamic = False
    detect.export = True
    detect.inplace = False

    dummy = torch.zeros(1, 3, 640, 640, device=device)
    with torch.no_grad():
        model(dummy)
        model(dummy)
    after = cache_audit(detect)
    if not after["all_scales_match"]:
        raise RuntimeError("Detect anchor-grid refresh did not reproduce anchors multiplied by stride")
    state_after = state_digest(model)
    if state_after != state_before:
        raise RuntimeError("Model state_dict changed during grid refresh")

    exported, _ = export_onnx(model, dummy, args.output, opset=17, dynamic=False, simplify=False)
    exported = Path(exported)
    if exported != args.output:
        raise RuntimeError(f"Unexpected export destination: {exported}")
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    input_shape = [dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in graph.graph.output[0].type.tensor_type.shape.dim]
    if input_shape != [1, 3, 640, 640] or output_shape != [1, 25200, 15]:
        raise RuntimeError(f"Unexpected ONNX contract: input={input_shape}, output={output_shape}")

    # Compare CPU PyTorch and CPU ONNX Runtime so the audit measures export
    # parity rather than backend-specific CUDA convolution drift.
    parity_model = attempt_load(str(args.weights), device="cpu", inplace=True, fuse=True).float().eval()
    parity_detect = [module for module in parity_model.modules() if isinstance(module, Detect)][0]
    parity_detect.grid = [torch.empty(0) for _ in range(parity_detect.nl)]
    parity_detect.anchor_grid = [torch.empty(0) for _ in range(parity_detect.nl)]
    parity_detect.dynamic = False
    parity_detect.export = True
    parity_detect.inplace = False
    parity_input = preprocess(args.parity_image)
    with torch.no_grad():
        torch_output = parity_model(torch.from_numpy(parity_input))[0].detach().numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {session.get_inputs()[0].name: parity_input})[0]
    delta = np.abs(torch_output - onnx_output)
    parity = {
        "image": str(args.parity_image),
        "torch_shape": list(torch_output.shape),
        "onnx_shape": list(onnx_output.shape),
        "all_finite": bool(np.isfinite(torch_output).all() and np.isfinite(onnx_output).all()),
        "maximum_absolute_error": float(delta.max()),
        "mean_absolute_error": float(delta.mean()),
        "p99_absolute_error": float(np.quantile(delta, 0.99)),
        "allclose_rtol_1e_4_atol_1e_5": bool(
            np.allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5)
        ),
    }
    if not parity["all_finite"] or parity["mean_absolute_error"] > 1e-4:
        raise RuntimeError(f"PyTorch/ONNX parity failed: {parity}")

    report = {
        "schema": "stage12_stage6_detect_grid_refresh_export_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scientific_change": "NONE_EXPORT_CACHE_REPAIR_ONLY",
        "source_checkpoint": str(args.weights),
        "source_checkpoint_sha256": sha256(args.weights),
        "state_dict_sha256_before": state_before,
        "state_dict_sha256_after": state_after,
        "state_dict_unchanged": state_before == state_after,
        "cache_before": before,
        "cache_after": after,
        "output_onnx": str(args.output),
        "output_onnx_sha256": sha256(args.output),
        "output_onnx_size_bytes": args.output.stat().st_size,
        "opset": 17,
        "static_shape": [1, 3, 640, 640],
        "output_shape": [1, 25200, 15],
        "embedded_nms": False,
        "parity": parity,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("STAGE6_GRID_REFRESH_EXPORT=PASS")
    print(f"ONNX_SHA256={report['output_onnx_sha256']}")


if __name__ == "__main__":
    main()
