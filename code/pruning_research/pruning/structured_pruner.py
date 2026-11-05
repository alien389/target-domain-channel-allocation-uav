from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from pruning_research.core.model_loader import load_detection_model
from pruning_research.utils.io_utils import ensure_dir, save_csv, save_json


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _block_index_from_name(block_name: str) -> int:
    try:
        return int(str(block_name).split(".")[-1])
    except Exception as exc:
        raise ValueError(f"Invalid block name: {block_name}") from exc


def _identity_indices(width: int) -> List[int]:
    return list(range(int(width)))


def _is_identity_keep(keep_indices: Sequence[int], width: int) -> bool:
    return list(keep_indices) == _identity_indices(width)


def _validate_indices(indices: Sequence[int], width: int, label: str) -> List[int]:
    normalized = sorted({int(idx) for idx in indices})
    for idx in normalized:
        if idx < 0 or idx >= int(width):
            raise IndexError(f"{label} index {idx} is outside width {width}")
    return normalized


def _keep_indices_from_scope(scope_info: Optional[Dict[str, Any]], default_width: int) -> List[int]:
    width = int(scope_info.get("width", default_width)) if scope_info else int(default_width)
    pruned = scope_info.get("selected_channel_indices", []) if scope_info else []
    pruned_set = {int(idx) for idx in pruned}
    keep = [idx for idx in range(width) if idx not in pruned_set]
    if not keep:
        raise ValueError(f"Scope pruning removed all channels for width={width}")
    return keep


def _resolve_source_index(layer_index: int, source_index: int) -> int:
    return layer_index - 1 if int(source_index) == -1 else int(source_index)


def _topology_width(value: Any) -> int:
    if isinstance(value, list) and value:
        return int(value[0])
    return int(value)


def _copy_bn_subset(old_bn: nn.BatchNorm2d, keep_out: Sequence[int]) -> nn.BatchNorm2d:
    keep_out = list(keep_out)
    new_bn = nn.BatchNorm2d(
        len(keep_out),
        eps=old_bn.eps,
        momentum=old_bn.momentum,
        affine=old_bn.affine,
        track_running_stats=old_bn.track_running_stats,
    ).to(device=old_bn.weight.device, dtype=old_bn.weight.dtype)

    with torch.no_grad():
        if old_bn.affine:
            new_bn.weight.copy_(old_bn.weight.data[keep_out].clone())
            new_bn.bias.copy_(old_bn.bias.data[keep_out].clone())
        if old_bn.track_running_stats:
            new_bn.running_mean.copy_(old_bn.running_mean.data[keep_out].clone())
            new_bn.running_var.copy_(old_bn.running_var.data[keep_out].clone())
            new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
    return new_bn


def _copy_conv2d_subset(
    old_conv: nn.Conv2d,
    in_keep: Optional[Sequence[int]] = None,
    out_keep: Optional[Sequence[int]] = None,
) -> nn.Conv2d:
    if old_conv.groups != 1:
        raise NotImplementedError(
            f"Grouped/depthwise convolution is not yet supported by structured_pruner: groups={old_conv.groups}"
        )

    in_keep = _validate_indices(in_keep or _identity_indices(old_conv.in_channels), old_conv.in_channels, "conv input")
    out_keep = _validate_indices(out_keep or _identity_indices(old_conv.out_channels), old_conv.out_channels, "conv output")

    new_conv = nn.Conv2d(
        in_channels=len(in_keep),
        out_channels=len(out_keep),
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=1,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.data[out_keep][:, in_keep].clone())
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias.data[out_keep].clone())
    return new_conv


def _copy_depthwise_conv2d_subset(old_conv: nn.Conv2d, channel_keep: Sequence[int]) -> nn.Conv2d:
    if old_conv.groups != old_conv.in_channels or old_conv.out_channels != old_conv.in_channels:
        raise NotImplementedError(
            "Only true depthwise channel-preserving convolutions are supported for C3Lite pruning."
        )

    channel_keep = _validate_indices(channel_keep, old_conv.in_channels, "depthwise channel")
    new_conv = nn.Conv2d(
        in_channels=len(channel_keep),
        out_channels=len(channel_keep),
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=len(channel_keep),
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.data[channel_keep].clone())
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias.data[channel_keep].clone())
    return new_conv


def _replace_conv_block_channels(module: nn.Module, in_keep: Sequence[int], out_keep: Sequence[int]) -> None:
    if not hasattr(module, "conv") or not hasattr(module, "bn"):
        raise TypeError(f"Module does not expose YOLOv5 Conv structure: {type(module).__name__}")
    module.conv = _copy_conv2d_subset(module.conv, in_keep=in_keep, out_keep=out_keep)
    module.bn = _copy_bn_subset(module.bn, keep_out=out_keep)


def _replace_dwconvlite_channels(module: nn.Module, channel_keep: Sequence[int]) -> None:
    if not (hasattr(module, "dw") and hasattr(module, "pw") and hasattr(module, "bn")):
        raise TypeError(f"Module does not expose DWConvLite structure: {type(module).__name__}")
    module.dw = _copy_depthwise_conv2d_subset(module.dw, channel_keep=channel_keep)
    module.pw = _copy_conv2d_subset(module.pw, in_keep=channel_keep, out_keep=channel_keep)
    module.bn = _copy_bn_subset(module.bn, keep_out=channel_keep)


def _prune_conv_block(module: nn.Module, input_keep: Sequence[int], output_keep: Sequence[int]) -> None:
    _replace_conv_block_channels(module, in_keep=input_keep, out_keep=output_keep)


def _prune_c3_block(
    module: nn.Module,
    input_keep: Sequence[int],
    main_keep: Sequence[int],
    skip_keep: Sequence[int],
    output_keep: Sequence[int],
) -> None:
    hidden_width = int(module.cv1.conv.out_channels)
    if int(module.cv2.conv.out_channels) != hidden_width:
        raise ValueError("C3 branch widths are inconsistent; cannot prune structurally.")
    if int(module.cv3.conv.in_channels) != hidden_width * 2:
        raise ValueError("C3 merge width is inconsistent with hidden branches; cannot prune structurally.")

    _replace_conv_block_channels(module.cv1, in_keep=input_keep, out_keep=main_keep)
    _replace_conv_block_channels(module.cv2, in_keep=input_keep, out_keep=skip_keep)

    for bottleneck in module.m:
        _replace_conv_block_channels(bottleneck.cv1, in_keep=main_keep, out_keep=main_keep)
        _replace_conv_block_channels(bottleneck.cv2, in_keep=main_keep, out_keep=main_keep)

    cv3_in_keep = list(main_keep) + [hidden_width + idx for idx in skip_keep]
    _replace_conv_block_channels(module.cv3, in_keep=cv3_in_keep, out_keep=output_keep)


def _prune_c3lite_block(
    module: nn.Module,
    input_keep: Sequence[int],
    main_keep: Sequence[int],
    skip_keep: Sequence[int],
    output_keep: Sequence[int],
) -> None:
    hidden_width = int(module.cv1.conv.out_channels)
    if int(module.cv2.conv.out_channels) != hidden_width:
        raise ValueError("C3Lite branch widths are inconsistent; cannot prune structurally.")
    if int(module.cv3.conv.in_channels) != hidden_width * 2:
        raise ValueError("C3Lite merge width is inconsistent with hidden branches; cannot prune structurally.")

    _replace_conv_block_channels(module.cv1, in_keep=input_keep, out_keep=main_keep)
    _replace_conv_block_channels(module.cv2, in_keep=input_keep, out_keep=skip_keep)

    for unit in module.m:
        _replace_dwconvlite_channels(unit, channel_keep=main_keep)

    cv3_in_keep = list(main_keep) + [hidden_width + idx for idx in skip_keep]
    _replace_conv_block_channels(module.cv3, in_keep=cv3_in_keep, out_keep=output_keep)


def _prune_sppf_block(
    module: nn.Module,
    input_keep: Sequence[int],
    hidden_keep: Sequence[int],
    output_keep: Sequence[int],
) -> None:
    hidden_width = int(module.cv1.conv.out_channels)
    if int(module.cv2.conv.in_channels) != hidden_width * 4:
        raise ValueError("SPPF merge width is inconsistent with hidden branch width; cannot prune structurally.")

    _replace_conv_block_channels(module.cv1, in_keep=input_keep, out_keep=hidden_keep)

    cv2_in_keep: List[int] = []
    for segment in range(4):
        offset = segment * hidden_width
        cv2_in_keep.extend(offset + idx for idx in hidden_keep)
    _replace_conv_block_channels(module.cv2, in_keep=cv2_in_keep, out_keep=output_keep)


def _adapt_detect_branch_inputs(module: nn.Module, input_keep_map: Dict[int, List[int]], layer_index: int) -> List[Dict[str, Any]]:
    branch_rows: List[Dict[str, Any]] = []
    for branch_index, source in enumerate(module.f):
        resolved_source = _resolve_source_index(layer_index, int(source))
        keep = input_keep_map[resolved_source]
        old_in_channels = int(module.m[branch_index].in_channels)
        changed = len(keep) != old_in_channels or not _is_identity_keep(keep, old_in_channels)
        if changed:
            module.m[branch_index] = _copy_conv2d_subset(module.m[branch_index], in_keep=keep, out_keep=None)
        branch_rows.append(
            {
                "branch_index": int(branch_index),
                "source_block": f"model.{resolved_source}",
                "input_channels_before": old_in_channels,
                "input_channels_after": len(keep),
                "changed": bool(changed),
            }
        )
    return branch_rows


def _refresh_runtime_metadata(model: nn.Module) -> None:
    for module in model.model:
        module.np = sum(parameter.numel() for parameter in module.parameters())


def _shape_repr(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.shape]
    if isinstance(value, list):
        return [_shape_repr(item) for item in value]
    if isinstance(value, tuple):
        return [_shape_repr(item) for item in value]
    return str(type(value).__name__)


def _validate_forward(model: nn.Module, device: str, imgsz: int) -> Dict[str, Any]:
    device_t = torch.device(device)
    model = model.to(device_t).eval()
    with torch.no_grad():
        sample = torch.zeros(1, 3, int(imgsz), int(imgsz), device=device_t)
        output = model(sample)
    primary = output[0] if isinstance(output, tuple) else output
    return {
        "forward_ok": True,
        "device": str(device_t),
        "imgsz": int(imgsz),
        "output_shape": _shape_repr(primary),
        "raw_output_shape": _shape_repr(output),
    }


def _derive_pruned_model_id(base_model_id: str, plan_payload: Dict[str, Any], explicit_model_id: Optional[str] = None) -> str:
    if explicit_model_id and explicit_model_id.strip():
        return explicit_model_id.strip()
    strategy_id = str(plan_payload.get("strategy", {}).get("id", "plan")).strip()
    target_label = str(plan_payload.get("target_label", "")).strip()
    if not target_label:
        base_ratio = int(round(float(plan_payload.get("base_ratio", 0.0)) * 100))
        target_label = f"r{base_ratio:02d}"
    return f"{base_model_id}_{strategy_id}_{target_label}_structured"


def _report_row(
    block_name: str,
    module_type: str,
    input_before: int,
    input_after: int,
    output_before: int,
    output_after: int,
    note: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "block_name": block_name,
        "module_type": module_type,
        "input_channels_before": int(input_before),
        "input_channels_after": int(input_after),
        "output_channels_before": int(output_before),
        "output_channels_after": int(output_after),
        "changed": bool(input_before != input_after or output_before != output_after),
        "note": note,
    }
    if extra:
        row.update(extra)
    return row


def _block_plan_has_active_prune(block_plan: Optional[Dict[str, Any]]) -> bool:
    if not block_plan:
        return False
    return any(int(scope.get("applied_prune_count_rounded", 0)) > 0 for scope in block_plan.get("scopes", {}).values())


def apply_structured_pruning(
    weights: str,
    yolov5_root: str,
    plan_json: str,
    profile_dir: str,
    output_dir: str,
    weights_output_dir: str,
    model_id: Optional[str] = None,
    pruned_model_id: Optional[str] = None,
    device: str = "cpu",
    validate: bool = True,
    validate_imgsz: int = 640,
) -> Dict[str, Any]:
    plan_path = Path(plan_json).resolve()
    profile_root = Path(profile_dir).resolve()
    output_root = Path(output_dir).resolve()
    weights_root = Path(weights_output_dir).resolve()

    plan_payload = _load_json(plan_path)
    topology_payload = _load_json(profile_root / "channel_topology.json")

    requested_model_id = model_id or str(plan_payload.get("model_id", "")).strip() or Path(weights).stem
    resolved_pruned_model_id = _derive_pruned_model_id(requested_model_id, plan_payload, explicit_model_id=pruned_model_id)

    artifact = load_detection_model(
        weights=weights,
        yolov5_root=yolov5_root,
        device=device,
        fuse=False,
        model_id=requested_model_id,
    )
    model = artifact.model.float().eval()
    original_params = int(sum(parameter.numel() for parameter in model.parameters()))

    try:
        from models.common import C3, C3Lite, Concat, Conv, SPPF
        from models.yolo import Detect
    except Exception as exc:
        raise RuntimeError("Failed to import YOLOv5 modules required for structural pruning.") from exc

    topology_by_index = {int(row["block_index"]): row for row in topology_payload.get("block_topology", [])}
    original_output_width_by_index = {
        int(row["block_index"]): _topology_width(row["output_channels"]) for row in topology_payload.get("block_topology", [])
    }
    block_plan_by_name = {str(block["block_name"]): block for block in plan_payload.get("blocks", [])}

    active_unsupported: List[Dict[str, Any]] = []
    for block in plan_payload.get("blocks", []):
        active_scope_count = sum(int(scope.get("applied_prune_count_rounded", 0)) for scope in block.get("scopes", {}).values())
        if active_scope_count <= 0:
            continue
        if str(block.get("module_type")) not in {"Conv", "C3", "C3Lite", "SPPF"}:
            active_unsupported.append(
                {
                    "block_name": block.get("block_name"),
                    "module_type": block.get("module_type"),
                }
            )
    if active_unsupported:
        raise NotImplementedError(f"Plan contains active blocks not yet supported by structured_pruner: {active_unsupported}")

    block_rows: List[Dict[str, Any]] = []
    output_keep_map: Dict[int, List[int]] = {}

    for layer_index, module in enumerate(model.model):
        block_name = f"model.{layer_index}"
        topology_row = topology_by_index[layer_index]
        block_plan = block_plan_by_name.get(block_name)
        block_has_active_prune = _block_plan_has_active_prune(block_plan)
        old_output_width = int(original_output_width_by_index[layer_index])

        if layer_index == 0:
            input_keep = _identity_indices(int(module.conv.in_channels) if hasattr(module, "conv") else 3)
        elif isinstance(module.f, int):
            input_keep = list(output_keep_map[_resolve_source_index(layer_index, int(module.f))])
        else:
            input_keep = []
            for source in module.f:
                resolved_source = _resolve_source_index(layer_index, int(source))
                input_keep.extend(output_keep_map[resolved_source])

        if isinstance(module, nn.Upsample):
            output_keep_map[layer_index] = list(input_keep)
            block_rows.append(
                _report_row(
                    block_name=block_name,
                    module_type="Upsample",
                    input_before=old_output_width,
                    input_after=len(input_keep),
                    output_before=old_output_width,
                    output_after=len(input_keep),
                    note="pass_through_channel_propagation",
                )
            )
            continue

        if isinstance(module, Concat):
            concat_keep: List[int] = []
            offset = 0
            for source in module.f:
                resolved_source = _resolve_source_index(layer_index, int(source))
                source_keep = output_keep_map[resolved_source]
                concat_keep.extend(offset + idx for idx in source_keep)
                offset += int(original_output_width_by_index[resolved_source])
            output_keep_map[layer_index] = concat_keep
            block_rows.append(
                _report_row(
                    block_name=block_name,
                    module_type="Concat",
                    input_before=old_output_width,
                    input_after=len(concat_keep),
                    output_before=old_output_width,
                    output_after=len(concat_keep),
                    note="concat_segment_propagation",
                )
            )
            continue

        if isinstance(module, Detect):
            branch_rows = _adapt_detect_branch_inputs(module, output_keep_map, layer_index)
            block_rows.append(
                {
                    "block_name": block_name,
                    "module_type": "Detect",
                    "input_channels_before": int(sum(int(branch["input_channels_before"]) for branch in branch_rows)),
                    "input_channels_after": int(sum(int(branch["input_channels_after"]) for branch in branch_rows)),
                    "output_channels_before": int(old_output_width),
                    "output_channels_after": int(old_output_width),
                    "changed": any(bool(branch["changed"]) for branch in branch_rows),
                    "note": "detect_branch_input_adaptation",
                    "branch_updates": json.dumps(branch_rows, ensure_ascii=False),
                }
            )
            continue

        if isinstance(module, Conv):
            output_keep = _keep_indices_from_scope(block_plan.get("scopes", {}).get("output") if block_plan else None, old_output_width)
            input_before = int(module.conv.in_channels)
            output_before = int(module.conv.out_channels)
            _prune_conv_block(module, input_keep=input_keep, output_keep=output_keep)
            output_keep_map[layer_index] = output_keep
            block_rows.append(
                _report_row(
                    block_name=block_name,
                    module_type="Conv",
                    input_before=input_before,
                    input_after=len(input_keep),
                    output_before=output_before,
                    output_after=len(output_keep),
                    note="planned_output_prune_and_input_propagation" if block_has_active_prune else "input_propagation_only",
                )
            )
            continue

        if isinstance(module, C3Lite):
            scopes = block_plan.get("scopes", {}) if block_plan else {}
            main_keep = _keep_indices_from_scope(scopes.get("main_hidden"), int(module.cv1.conv.out_channels))
            skip_keep = _keep_indices_from_scope(scopes.get("skip_hidden"), int(module.cv2.conv.out_channels))
            output_keep = _keep_indices_from_scope(scopes.get("output"), old_output_width)
            input_before = int(module.cv1.conv.in_channels)
            output_before = int(module.cv3.conv.out_channels)
            main_before = int(module.cv1.conv.out_channels)
            skip_before = int(module.cv2.conv.out_channels)
            _prune_c3lite_block(
                module,
                input_keep=input_keep,
                main_keep=main_keep,
                skip_keep=skip_keep,
                output_keep=output_keep,
            )
            output_keep_map[layer_index] = output_keep
            block_rows.append(
                _report_row(
                    block_name=block_name,
                    module_type="C3Lite",
                    input_before=input_before,
                    input_after=len(input_keep),
                    output_before=output_before,
                    output_after=len(output_keep),
                    note="planned_c3lite_structural_prune" if block_has_active_prune else "input_propagation_only",
                    extra={
                        "main_hidden_before": main_before,
                        "main_hidden_after": len(main_keep),
                        "skip_hidden_before": skip_before,
                        "skip_hidden_after": len(skip_keep),
                    },
                )
            )
            continue

        if isinstance(module, C3):
            scopes = block_plan.get("scopes", {}) if block_plan else {}
            main_keep = _keep_indices_from_scope(scopes.get("main_hidden"), int(module.cv1.conv.out_channels))
            skip_keep = _keep_indices_from_scope(scopes.get("skip_hidden"), int(module.cv2.conv.out_channels))
            output_keep = _keep_indices_from_scope(scopes.get("output"), old_output_width)
            input_before = int(module.cv1.conv.in_channels)
            output_before = int(module.cv3.conv.out_channels)
            main_before = int(module.cv1.conv.out_channels)
            skip_before = int(module.cv2.conv.out_channels)
            _prune_c3_block(
                module,
                input_keep=input_keep,
                main_keep=main_keep,
                skip_keep=skip_keep,
                output_keep=output_keep,
            )
            output_keep_map[layer_index] = output_keep
            block_rows.append(
                _report_row(
                    block_name=block_name,
                    module_type="C3",
                    input_before=input_before,
                    input_after=len(input_keep),
                    output_before=output_before,
                    output_after=len(output_keep),
                    note="planned_c3_structural_prune" if block_has_active_prune else "input_propagation_only",
                    extra={
                        "main_hidden_before": main_before,
                        "main_hidden_after": len(main_keep),
                        "skip_hidden_before": skip_before,
                        "skip_hidden_after": len(skip_keep),
                    },
                )
            )
            continue

        if isinstance(module, SPPF):
            scopes = block_plan.get("scopes", {}) if block_plan else {}
            hidden_keep = _keep_indices_from_scope(scopes.get("hidden"), int(module.cv1.conv.out_channels))
            output_keep = _keep_indices_from_scope(scopes.get("output"), old_output_width)
            input_before = int(module.cv1.conv.in_channels)
            output_before = int(module.cv2.conv.out_channels)
            hidden_before = int(module.cv1.conv.out_channels)
            _prune_sppf_block(
                module,
                input_keep=input_keep,
                hidden_keep=hidden_keep,
                output_keep=output_keep,
            )
            output_keep_map[layer_index] = output_keep
            block_rows.append(
                _report_row(
                    block_name=block_name,
                    module_type="SPPF",
                    input_before=input_before,
                    input_after=len(input_keep),
                    output_before=output_before,
                    output_after=len(output_keep),
                    note="planned_sppf_structural_prune" if block_has_active_prune else "input_propagation_only",
                    extra={
                        "hidden_before": hidden_before,
                        "hidden_after": len(hidden_keep),
                    },
                )
            )
            continue

        raise NotImplementedError(f"Unsupported top-level module in structured_pruner: {type(module).__name__}")

    model.eval()
    _refresh_runtime_metadata(model)

    forward_validation = _validate_forward(model, device=device, imgsz=validate_imgsz) if validate else {"forward_ok": False, "skipped": True}

    results_dir = ensure_dir(output_root / requested_model_id / f"{plan_path.stem}_applied")
    model_weights_dir = ensure_dir(weights_root / requested_model_id)
    checkpoint_path = model_weights_dir / f"{resolved_pruned_model_id}.pt"

    model.model_id = resolved_pruned_model_id
    model.pruning_metadata = {
        "source_weights": str(Path(weights).resolve()),
        "source_model_id": requested_model_id,
        "plan_json": str(plan_path),
        "profile_dir": str(profile_root),
        "execution_scope": "Conv/C3/C3Lite/SPPF structural pruning with propagated adaptation through Upsample/Concat/Detect consumers",
    }

    checkpoint_model = copy.deepcopy(model).cpu().float().eval()
    checkpoint_payload = {
        "model": checkpoint_model,
        "optimizer": None,
        "ema": None,
        "updates": None,
        "epoch": -1,
        "best_fitness": None,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    reload_validation: Dict[str, Any]
    if validate:
        reloaded_artifact = load_detection_model(
            weights=str(checkpoint_path),
            yolov5_root=yolov5_root,
            device=device,
            fuse=False,
            model_id=resolved_pruned_model_id,
        )
        reload_validation = _validate_forward(reloaded_artifact.model, device=device, imgsz=validate_imgsz)
    else:
        reload_validation = {"forward_ok": False, "skipped": True}

    block_csv = save_csv(block_rows, results_dir / "structured_pruning_blocks.csv")
    applied_plan_snapshot = save_json(plan_payload, results_dir / "applied_plan_snapshot.json")

    pruned_params = int(sum(parameter.numel() for parameter in model.parameters()))

    summary = {
        "source_model_id": requested_model_id,
        "pruned_model_id": resolved_pruned_model_id,
        "plan_label": str(plan_payload.get("plan_label", "")).strip() or plan_path.stem.replace("pruning_plan_", ""),
        "target_label": str(plan_payload.get("target_label", "")).strip(),
        "search_role": str(plan_payload.get("search_role", "")).strip(),
        "source_weights": str(Path(weights).resolve()),
        "pruned_weights": str(checkpoint_path),
        "plan_json": str(plan_path),
        "applied_plan_snapshot_json": str(applied_plan_snapshot),
        "profile_dir": str(profile_root),
        "results_dir": str(results_dir),
        "structured_pruning_blocks_csv": str(block_csv),
        "device": device,
        "validate_imgsz": int(validate_imgsz),
        "original_params": original_params,
        "pruned_params": pruned_params,
        "param_reduction": int(original_params - pruned_params),
        "param_reduction_ratio": float((original_params - pruned_params) / original_params) if original_params else 0.0,
        "active_plan_blocks": [
            block["block_name"]
            for block in plan_payload.get("blocks", [])
            if sum(int(scope.get("applied_prune_count_rounded", 0)) for scope in block.get("scopes", {}).values()) > 0
        ],
        "validation": {
            "in_memory_forward": forward_validation,
            "reloaded_forward": reload_validation,
        },
        "protocol": {
            "load_mode": "unfused_yolov5_runtime_graph",
            "save_mode": "checkpoint_dict_with_pruned_model_object",
            "supported_active_module_types": ["Conv", "C3", "C3Lite", "SPPF"],
            "propagation_support": ["Upsample", "Concat", "Detect input adaptation"],
            "note": "This stage applies the pruning plan structurally and updates downstream consumer input channels to preserve graph validity.",
        },
    }
    summary_json = save_json(summary, results_dir / "structured_pruning_summary.json")
    summary["structured_pruning_summary_json"] = str(summary_json)
    return summary
