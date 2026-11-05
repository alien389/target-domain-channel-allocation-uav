from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch


def _clone_profile_input(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, list):
        return [_clone_profile_input(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_profile_input(v) for v in x)
    return deepcopy(x)


def _shape_repr(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return "x".join(str(int(v)) for v in x.shape)
    if isinstance(x, list):
        return ";".join(_shape_repr(v) for v in x)
    if isinstance(x, tuple):
        return ";".join(_shape_repr(v) for v in x)
    return str(type(x).__name__)


def _channel_repr(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return int(x.shape[1]) if x.ndim >= 2 else None
    if isinstance(x, list):
        return ";".join(str(_channel_repr(v)) for v in x)
    if isinstance(x, tuple):
        return ";".join(str(_channel_repr(v)) for v in x)
    return None


def _profile_module_cost(module: torch.nn.Module, module_input: Any) -> Tuple[Any, Any, str]:
    try:
        from thop import profile
    except Exception:
        return None, None, "thop_unavailable"

    try:
        safe_input = _clone_profile_input(module_input)
        macs, _ = profile(module, inputs=(safe_input,), verbose=False)
        return float(macs / 1e9), float((2.0 * macs) / 1e9), "ok"
    except Exception as exc:
        return None, None, f"profile_failed:{type(exc).__name__}"


def _normalize_outputs_cache_value(x: Any) -> Any:
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], list):
        return x[1]
    return x


def build_layer_profile(
    model: torch.nn.Module,
    layer_rows: List[Dict[str, Any]],
    imgsz: int = 640,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    outputs: List[Any] = []
    profiled_rows: List[Dict[str, Any]] = []

    model.eval()
    for base_row, module in zip(layer_rows, getattr(model, "model", [])):
        if module.f != -1:
            module_input = outputs[module.f] if isinstance(module.f, int) else [x if j == -1 else outputs[j] for j in module.f]
        else:
            module_input = x

        input_shape = _shape_repr(module_input)
        input_channels = _channel_repr(module_input)
        macs_g, flops_g, profile_status = _profile_module_cost(module, module_input)
        output = module(module_input)
        cached_output = _normalize_outputs_cache_value(output)
        outputs.append(cached_output)
        x = cached_output

        row = dict(base_row)
        row.update(
            {
                "input_shape": input_shape,
                "output_shape": _shape_repr(output),
                "in_channels": input_channels if input_channels is not None else row.get("in_channels_static"),
                "out_channels": _channel_repr(output) if _channel_repr(output) is not None else row.get("out_channels_static"),
                "macs_g": macs_g,
                "flops_g": flops_g,
                "profile_status": profile_status,
            }
        )
        profiled_rows.append(row)

    total_params = sum(int(row.get("params", 0)) for row in profiled_rows)
    total_flops_g = sum(float(row["flops_g"]) for row in profiled_rows if row.get("flops_g") is not None)
    profile_failures = [row["layer_name"] for row in profiled_rows if row.get("profile_status") != "ok"]

    summary = {
        "total_layers": len(profiled_rows),
        "total_params": total_params,
        "total_flops_g": total_flops_g,
        "profile_failures": profile_failures,
    }
    return profiled_rows, summary

