from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import onnx


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE = PROJECT / "12_jetson_fp16_deployment"
YOLO = Path(_release_paths.expand('${YOLOV5_ROOT}'))
PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))
EXPORTER = YOLO / "export.py"
LOCK = json.loads((STAGE / "10_manifests/locked_model_manifest.json").read_text(encoding="utf-8"))
OUT = STAGE / "03_onnx/fp32"
LOG = STAGE / "09_logs/onnx_export"
MANIFEST_DIR = STAGE / "10_manifests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    result = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append("?")
    return result


OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
rows = []
for model in LOCK["models"]:
    model_id = model["model_id"]
    checkpoint = STAGE / "02_locked_inputs/checkpoints" / f"{model_id}_seed42_terminal_ema.pt"
    if sha256(checkpoint) != model["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch before ONNX export: {model_id}")
    generated = checkpoint.with_suffix(".onnx")
    destination = OUT / f"{model_id}_seed42_fp32.onnx"
    if generated.exists() or destination.exists():
        raise RuntimeError(f"Refusing to overwrite an ONNX artifact for {model_id}")

    command = [
        str(PYTHON),
        str(EXPORTER),
        "--weights",
        str(checkpoint),
        "--imgsz",
        "640",
        "640",
        "--batch-size",
        "1",
        "--device",
        "0",
        "--include",
        "onnx",
        "--opset",
        "17",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(YOLO)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=YOLO, env=env, text=True, capture_output=True)
    ended = datetime.now(timezone.utc)
    (LOG / f"{model_id}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (LOG / f"{model_id}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    (LOG / f"{model_id}.command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"ONNX export failed for {model_id}; inspect {LOG}")
    if not generated.is_file():
        raise RuntimeError(f"Exporter did not create expected ONNX for {model_id}")
    generated.replace(destination)

    graph = onnx.load(destination)
    onnx.checker.check_model(graph)
    input_info = graph.graph.input[0]
    output_info = graph.graph.output[0]
    input_shape = tensor_shape(input_info)
    output_shape = tensor_shape(output_info)
    nms_nodes = [node.op_type for node in graph.graph.node if "NMS" in node.op_type.upper() or "NONMAX" in node.op_type.upper()]
    opset = max(item.version for item in graph.opset_import)
    if input_shape != [1, 3, 640, 640]:
        raise RuntimeError(f"Unexpected input shape for {model_id}: {input_shape}")
    if output_shape != [1, 25200, 15]:
        raise RuntimeError(f"Unexpected output shape for {model_id}: {output_shape}")
    if opset != 17 or nms_nodes:
        raise RuntimeError(f"ONNX contract failure for {model_id}: opset={opset}, nms={nms_nodes}")

    rows.append(
        {
            "model_id": model_id,
            "paper_name": model["paper_name"],
            "source_checkpoint_sha256": model["checkpoint_sha256"],
            "onnx_path": _release_paths.expand(f'${{JETSON_PROJECT_ROOT}}/03_onnx/fp32/{destination.name}'),
            "onnx_sha256": sha256(destination),
            "onnx_size_bytes": destination.stat().st_size,
            "input_name": input_info.name,
            "input_shape": json.dumps(input_shape, separators=(",", ":")),
            "output_name": output_info.name,
            "output_shape": json.dumps(output_shape, separators=(",", ":")),
            "opset": opset,
            "static_batch": True,
            "embedded_nms_nodes": len(nms_nodes),
            "exporter_path": str(EXPORTER),
            "exporter_sha256": sha256(EXPORTER),
            "command_path": _release_paths.expand(f'${{JETSON_PROJECT_ROOT}}/09_logs/onnx_export/{model_id}.command.txt'),
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "status": "PASS",
        }
    )

csv_path = MANIFEST_DIR / "canonical_onnx_manifest.csv"
json_path = MANIFEST_DIR / "canonical_onnx_manifest.json"
with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(
    json.dumps(
        {
            "schema": "meta_jetson_canonical_onnx_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_count": len(rows),
            "static_batch": 1,
            "input_shape": [1, 3, 640, 640],
            "output_shape": [1, 25200, 15],
            "opset": 17,
            "embedded_nms": False,
            "models": rows,
            "status": "PASS",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"CANONICAL_ONNX_EXPORTS={len(rows)}")
print(f"CANONICAL_ONNX_MANIFEST_SHA256={sha256(json_path)}")
