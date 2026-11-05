from __future__ import annotations

from typing import Any, Dict, List

import torch


def _infer_repeats(module: torch.nn.Module) -> int:
    if not hasattr(module, "m"):
        return 1
    inner = getattr(module, "m")
    if isinstance(inner, (torch.nn.Sequential, torch.nn.ModuleList, list, tuple)):
        return int(len(inner))
    return 1


def _normalize_sources(from_spec: Any, current_index: int) -> List[Any]:
    raw = list(from_spec) if isinstance(from_spec, (list, tuple)) else [from_spec]
    normalized: List[Any] = []
    for source in raw:
        if source == -1:
            normalized.append("input" if current_index == 0 else current_index - 1)
        else:
            normalized.append(int(source))
    return normalized


def _stage_for_index(model: torch.nn.Module, layer_index: int) -> str:
    yaml_def = getattr(model, "yaml", {}) or {}
    backbone_len = len(yaml_def.get("backbone", []))
    total_layers = len(getattr(model, "model", []))
    if layer_index < backbone_len:
        return "backbone"
    if layer_index == total_layers - 1:
        return "head"
    return "neck"


def _tensor_channels_from_conv(module: torch.nn.Module) -> Dict[str, Any]:
    module_type = type(module).__name__

    if module_type.startswith("C3") and hasattr(module, "cv1") and hasattr(module, "cv3"):
        return {
            "in_channels_static": int(module.cv1.conv.in_channels),
            "out_channels_static": int(module.cv3.conv.out_channels),
            "kernel_static": "1x1;3x3",
            "stride_static": "1x1",
            "groups_static": int(module.cv1.conv.groups),
        }

    if module_type == "SPPF" and hasattr(module, "cv1") and hasattr(module, "cv2"):
        return {
            "in_channels_static": int(module.cv1.conv.in_channels),
            "out_channels_static": int(module.cv2.conv.out_channels),
            "kernel_static": "1x1;5x5",
            "stride_static": "1x1",
            "groups_static": int(module.cv1.conv.groups),
        }

    if module_type == "Detect" and hasattr(module, "m"):
        branch_inputs = [str(int(conv.in_channels)) for conv in module.m]
        branch_outputs = [str(int(conv.out_channels)) for conv in module.m]
        return {
            "in_channels_static": ";".join(branch_inputs),
            "out_channels_static": ";".join(branch_outputs),
            "kernel_static": "1x1",
            "stride_static": "1x1",
            "groups_static": 1,
        }

    conv_layers = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        return {
            "in_channels_static": None,
            "out_channels_static": None,
            "kernel_static": None,
            "stride_static": None,
            "groups_static": None,
        }

    first_conv = conv_layers[0]
    last_conv = conv_layers[-1]
    kernel_values = sorted({f"{m.kernel_size[0]}x{m.kernel_size[1]}" for m in conv_layers})
    stride_values = sorted({f"{m.stride[0]}x{m.stride[1]}" for m in conv_layers})
    return {
        "in_channels_static": int(first_conv.in_channels),
        "out_channels_static": int(last_conv.out_channels),
        "kernel_static": ";".join(kernel_values),
        "stride_static": ";".join(stride_values),
        "groups_static": int(first_conv.groups),
    }


def inspect_model_layers(model: torch.nn.Module) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    for module in getattr(model, "model", []):
        module_type = type(module).__name__
        stage = _stage_for_index(model, int(module.i))
        static_info = _tensor_channels_from_conv(module)

        residual_connections = 0
        if hasattr(module, "m"):
            try:
                residual_connections = int(sum(1 for sub in module.m.modules() if getattr(sub, "add", False)))
            except Exception:
                residual_connections = 0

        detect_branches = 0
        if module_type == "Detect":
            try:
                detect_branches = int(len(module.m))
            except Exception:
                detect_branches = 0

        concat_inputs = 0
        if module_type == "Concat":
            sources = getattr(module, "f", [])
            concat_inputs = len(sources) if isinstance(sources, (list, tuple)) else 1

        prune_allowed = bool((module_type in {"Conv", "SPPF"} or module_type.startswith("C3")) and stage in {"backbone", "neck"})

        layer_row = {
            "layer_index": int(module.i),
            "layer_name": f"model.{int(module.i)}",
            "module_type": module_type,
            "module_path": getattr(module, "type", module_type),
            "stage": stage,
            "from_spec": getattr(module, "f", -1),
            "from_indices": _normalize_sources(getattr(module, "f", -1), int(module.i)),
            "from_indices_text": ";".join(str(x) for x in _normalize_sources(getattr(module, "f", -1), int(module.i))),
            "params": int(sum(p.numel() for p in module.parameters())),
            "params_parse_model": int(getattr(module, "np", 0)),
            "repeats": _infer_repeats(module),
            "residual_connections": residual_connections,
            "concat_inputs": concat_inputs,
            "detect_branches": detect_branches,
            "prune_allowed": prune_allowed,
            **static_info,
        }
        layers.append(layer_row)
    return layers
