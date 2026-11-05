from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from pruning_research.core.dependency_graph import build_dependency_graph
from pruning_research.core.model_inspector import inspect_model_layers
from pruning_research.core.model_loader import load_detection_model
from pruning_research.profiling.deep_structure_profile import build_channel_topology, build_leaf_profiles
from pruning_research.profiling.profile_blocks import build_block_profile
from pruning_research.profiling.profile_structure import build_layer_profile
from pruning_research.registry.registry_manager import ensure_registry_files, register_experiment, register_model
from pruning_research.utils.hash_utils import stable_payload_hash
from pruning_research.utils.id_utils import build_experiment_id
from pruning_research.utils.io_utils import ensure_dir, save_csv, save_json


def _summarize_block_profile(block_rows):
    stage_summary: Dict[str, Dict[str, Any]] = {}
    for row in block_rows:
        stage = str(row["stage"])
        bucket = stage_summary.setdefault(
            stage,
            {
                "block_count": 0,
                "prune_allowed_count": 0,
                "params": 0,
                "flops_g": 0.0,
            },
        )
        bucket["block_count"] += 1
        bucket["prune_allowed_count"] += int(bool(row.get("prune_allowed", False)))
        bucket["params"] += int(row.get("params", 0) or 0)
        bucket["flops_g"] += float(row.get("flops_g", 0.0) or 0.0)

    top_blocks_by_flops = sorted(block_rows, key=lambda row: float(row.get("flops_g", 0.0) or 0.0), reverse=True)[:5]
    top_blocks_by_params = sorted(block_rows, key=lambda row: int(row.get("params", 0) or 0), reverse=True)[:5]

    return {
        "stage_summary": stage_summary,
        "top_blocks_by_flops": [
            {
                "block_name": row["block_name"],
                "block_label": row.get("block_label"),
                "stage": row["stage"],
                "flops_g": float(row.get("flops_g", 0.0) or 0.0),
            }
            for row in top_blocks_by_flops
        ],
        "top_blocks_by_params": [
            {
                "block_name": row["block_name"],
                "block_label": row.get("block_label"),
                "stage": row["stage"],
                "params": int(row.get("params", 0) or 0),
            }
            for row in top_blocks_by_params
        ],
    }


def run_structural_profile_export(
    weights: str,
    yolov5_root: str,
    output_dir: str,
    registry_dir: str,
    model_id: str,
    imgsz: int = 640,
    device: str = "cpu",
    fuse: bool = True,
) -> Dict[str, Any]:
    artifact = load_detection_model(
        weights=weights,
        yolov5_root=yolov5_root,
        device=device,
        fuse=fuse,
        model_id=model_id,
    )

    layer_rows = inspect_model_layers(artifact.model)
    dependency_map = build_dependency_graph(artifact.model, layer_rows)
    layer_profile, structure_summary = build_layer_profile(artifact.model, layer_rows, imgsz=imgsz)
    block_profile = build_block_profile(layer_profile)
    leaf_module_profile, leaf_invocation_trace, deep_structure_summary = build_leaf_profiles(
        artifact.model,
        layer_profile,
        imgsz=imgsz,
    )
    channel_topology, pruning_constraints = build_channel_topology(artifact.model, layer_profile)
    block_profile_summary = _summarize_block_profile(block_profile)

    profile_config = {
        "weights": artifact.weights_path,
        "yolov5_root": artifact.yolov5_root,
        "model_id": artifact.model_id,
        "imgsz": imgsz,
        "device": device,
        "fuse": bool(fuse),
    }
    config_hash = stable_payload_hash(profile_config)
    experiment_id = build_experiment_id("structural_profile", artifact.model_id)

    target_dir = ensure_dir(Path(output_dir) / artifact.model_id)
    layer_profile_path = save_csv(layer_profile, target_dir / "layer_profile.csv")
    block_profile_path = save_csv(block_profile, target_dir / "block_profile.csv")
    leaf_module_profile_path = save_csv(leaf_module_profile, target_dir / "leaf_module_profile.csv")
    leaf_invocation_trace_path = save_csv(leaf_invocation_trace, target_dir / "leaf_invocation_trace.csv")
    dependency_map_path = save_json(dependency_map, target_dir / "dependency_map.json")
    channel_topology_path = save_json(channel_topology, target_dir / "channel_topology.json")
    pruning_constraints_path = save_json(pruning_constraints, target_dir / "pruning_constraints.json")
    registry_paths = ensure_registry_files(registry_dir)

    model_structure_summary = {
        "experiment": {
            "experiment_id": experiment_id,
            "experiment_type": "structural_profile",
            "config_hash": config_hash,
            "status": "completed",
        },
        "model": artifact.to_metadata(),
        "profiling": {
            "imgsz": imgsz,
            "device": device,
            "fuse": fuse,
            **structure_summary,
        },
        "profiling_protocol": {
            "block_definition": "top_level_yolov5_module",
            "params_method": "sum(module.parameters()) on loaded runtime graph",
            "flops_method": "THOP MACs x2 on per-module runtime inputs for 1x3ximgszximgsz",
            "dependency_source": "YOLOv5 parse_model attributes (i, f, type, m)",
            "leaf_trace_method": "forward hooks on leaf runtime modules over a 1x3ximgszximgsz dummy input",
            "channel_topology_method": "module-specific semantic extraction for Conv, C3, SPPF, Concat, Upsample, and Detect",
            "scope_note": "This phase performs structural profiling and dependency extraction only; no pruning is applied.",
        },
        "block_profile_summary": block_profile_summary,
        "deep_structure_summary": deep_structure_summary,
        "pruning_constraints_summary": {
            "immutable_block_count": len(pruning_constraints["immutable_blocks"]),
            "prune_candidate_block_count": len(pruning_constraints["prune_candidate_blocks"]),
            "residual_sensitive_block_count": len(pruning_constraints["residual_sensitive_blocks"]),
            "concat_block_count": len(pruning_constraints["concat_blocks"]),
            "detect_input_block_count": len(pruning_constraints["detect_input_blocks"]),
        },
        "paths": {
            "layer_profile_csv": str(layer_profile_path),
            "block_profile_csv": str(block_profile_path),
            "leaf_module_profile_csv": str(leaf_module_profile_path),
            "leaf_invocation_trace_csv": str(leaf_invocation_trace_path),
            "dependency_map_json": str(dependency_map_path),
            "channel_topology_json": str(channel_topology_path),
            "pruning_constraints_json": str(pruning_constraints_path),
            "model_registry_csv": str(registry_paths["model_registry_csv"]),
            "experiment_registry_csv": str(registry_paths["experiment_registry_csv"]),
        },
    }
    summary_path = save_json(model_structure_summary, target_dir / "model_structure_summary.json")

    register_model(
        registry_dir,
        {
            "model_id": artifact.model_id,
            "parent_model_id": "",
            "stage": "baseline",
            "weights_path": artifact.weights_path,
            "checkpoint_type": artifact.checkpoint_type,
            "model_source": artifact.model_source,
            "fused": artifact.fused,
            "nc": artifact.nc,
            "stride": ",".join(str(v) for v in (artifact.to_metadata().get("stride") or [])),
            "status": "profiled",
            "notes": "structural_profile_ready_with_deep_topology",
        },
    )
    register_experiment(
        registry_dir,
        {
            "experiment_id": experiment_id,
            "experiment_type": "structural_profile",
            "model_id": artifact.model_id,
            "config_hash": config_hash,
            "status": "completed",
            "output_dir": str(target_dir),
            "summary_path": str(summary_path),
            "notes": "layer_profile, block_profile, dependency_map, leaf traces, and pruning constraints exported",
        },
    )

    return {
        "experiment_id": experiment_id,
        "config_hash": config_hash,
        "output_dir": str(target_dir),
        "layer_profile_csv": str(layer_profile_path),
        "block_profile_csv": str(block_profile_path),
        "leaf_module_profile_csv": str(leaf_module_profile_path),
        "leaf_invocation_trace_csv": str(leaf_invocation_trace_path),
        "dependency_map_json": str(dependency_map_path),
        "channel_topology_json": str(channel_topology_path),
        "pruning_constraints_json": str(pruning_constraints_path),
        "model_structure_summary_json": str(summary_path),
        "model_registry_csv": str(registry_paths["model_registry_csv"]),
        "experiment_registry_csv": str(registry_paths["experiment_registry_csv"]),
        "model_metadata": artifact.to_metadata(),
        "profiling_summary": structure_summary,
        "block_profile_summary": block_profile_summary,
        "deep_structure_summary": deep_structure_summary,
    }
