from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def _shape_repr(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return "x".join(str(int(v)) for v in x.shape)
    if isinstance(x, (list, tuple)):
        return ";".join(_shape_repr(v) for v in x)
    if x is None:
        return ""
    return str(type(x).__name__)


def _channel_repr(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return int(x.shape[1]) if x.ndim >= 2 else None
    if isinstance(x, (list, tuple)):
        return ";".join("" if _channel_repr(v) is None else str(_channel_repr(v)) for v in x)
    return None


def _is_leaf_module(module: torch.nn.Module) -> bool:
    return len(list(module.children())) == 0


def _parse_top_level_name(module_path: str) -> Tuple[Optional[str], Optional[int]]:
    parts = module_path.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return f"model.{parts[1]}", int(parts[1])
    return None, None


def _parse_channels_value(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    text = str(value).strip()
    if not text:
        return []
    if ";" in text:
        return [int(part) for part in text.split(";") if part.strip()]
    return [int(text)]


def _module_static_attrs(module: torch.nn.Module) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "kernel": None,
        "stride": None,
        "padding": None,
        "dilation": None,
        "groups": None,
        "in_channels_static": None,
        "out_channels_static": None,
    }

    if isinstance(module, torch.nn.Conv2d):
        attrs.update(
            {
                "kernel": f"{module.kernel_size[0]}x{module.kernel_size[1]}",
                "stride": f"{module.stride[0]}x{module.stride[1]}",
                "padding": f"{module.padding[0]}x{module.padding[1]}",
                "dilation": f"{module.dilation[0]}x{module.dilation[1]}",
                "groups": int(module.groups),
                "in_channels_static": int(module.in_channels),
                "out_channels_static": int(module.out_channels),
            }
        )
    elif isinstance(module, torch.nn.BatchNorm2d):
        attrs["out_channels_static"] = int(module.num_features)
    elif isinstance(module, torch.nn.MaxPool2d):
        kernel = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        attrs.update(
            {
                "kernel": f"{kernel[0]}x{kernel[1]}",
                "stride": f"{stride[0]}x{stride[1]}",
                "padding": f"{padding[0]}x{padding[1]}",
            }
        )
    elif isinstance(module, torch.nn.Upsample):
        attrs["stride"] = f"scale={module.scale_factor}"

    return attrs


def _build_named_module_map(model: torch.nn.Module) -> Dict[int, str]:
    return {id(module): name for name, module in model.named_modules() if name}


def _leaf_module_rows(model: torch.nn.Module, layer_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stage_by_index = {int(row["layer_index"]): row["stage"] for row in layer_rows}
    layer_name_by_index = {int(row["layer_index"]): row["layer_name"] for row in layer_rows}
    rows: List[Dict[str, Any]] = []

    for name, module in model.named_modules():
        if not name or not _is_leaf_module(module):
            continue

        top_level_name, top_level_index = _parse_top_level_name(name)
        parent_path = name.rsplit(".", 1)[0] if "." in name else ""
        static_attrs = _module_static_attrs(module)
        rows.append(
            {
                "module_path": name,
                "parent_path": parent_path,
                "top_level_block": top_level_name,
                "top_level_index": top_level_index,
                "top_level_stage": stage_by_index.get(top_level_index),
                "top_level_layer_name": layer_name_by_index.get(top_level_index),
                "module_type": type(module).__name__,
                "params": int(sum(p.numel() for p in module.parameters())),
                "trainable_params": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
                "buffers": int(sum(b.numel() for b in module.buffers())),
                **static_attrs,
            }
        )

    rows.sort(key=lambda row: row["module_path"])
    return rows


def build_leaf_profiles(
    model: torch.nn.Module,
    layer_rows: List[Dict[str, Any]],
    imgsz: int = 640,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    module_rows = _leaf_module_rows(model, layer_rows)
    module_row_by_path = {row["module_path"]: row for row in module_rows}
    module_name_map = _build_named_module_map(model)
    invocations: List[Dict[str, Any]] = []
    call_counters: Counter[str] = Counter()
    global_trace_index = 0
    handles = []

    def hook_fn(module: torch.nn.Module, module_input: Tuple[Any, ...], module_output: Any) -> None:
        nonlocal global_trace_index
        module_path = module_name_map.get(id(module))
        if not module_path:
            return
        call_counters[module_path] += 1
        global_trace_index += 1
        module_row = module_row_by_path.get(module_path, {})
        first_arg = module_input[0] if len(module_input) == 1 else list(module_input)
        invocations.append(
            {
                "trace_index": global_trace_index,
                "module_path": module_path,
                "module_call_index": call_counters[module_path],
                "top_level_block": module_row.get("top_level_block"),
                "top_level_index": module_row.get("top_level_index"),
                "top_level_stage": module_row.get("top_level_stage"),
                "module_type": type(module).__name__,
                "input_shape": _shape_repr(first_arg),
                "output_shape": _shape_repr(module_output),
                "input_channels": _channel_repr(first_arg),
                "output_channels": _channel_repr(module_output),
            }
        )

    for name, module in model.named_modules():
        if name and _is_leaf_module(module):
            handles.append(module.register_forward_hook(hook_fn))

    try:
        device = next(model.parameters()).device
        x = torch.zeros(1, 3, imgsz, imgsz, device=device)
        model.eval()
        with torch.no_grad():
            _ = model(x)
    finally:
        for handle in handles:
            handle.remove()

    first_runtime_by_path: Dict[str, Dict[str, Any]] = {}
    reuse_counts = Counter(inv["module_path"] for inv in invocations)
    for invocation in invocations:
        first_runtime_by_path.setdefault(invocation["module_path"], invocation)

    for row in module_rows:
        runtime = first_runtime_by_path.get(row["module_path"], {})
        row["input_shape"] = runtime.get("input_shape", "")
        row["output_shape"] = runtime.get("output_shape", "")
        row["input_channels"] = runtime.get("input_channels")
        row["output_channels"] = runtime.get("output_channels")
        row["call_count"] = int(reuse_counts.get(row["module_path"], 0))
        row["reused_module"] = bool(reuse_counts.get(row["module_path"], 0) > 1)

    summary = {
        "leaf_module_count": len(module_rows),
        "leaf_invocation_count": len(invocations),
        "reused_leaf_module_count": int(sum(1 for row in module_rows if row.get("reused_module"))),
        "module_type_summary": dict(Counter(row["module_type"] for row in module_rows)),
    }
    return module_rows, invocations, summary


def _source_segments_from_row(row: Dict[str, Any], layer_rows_by_index: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    cursor = 0
    for source in row.get("from_indices", []):
        if isinstance(source, str):
            source_channels = [3] if source == "input" else []
            source_name = source
        else:
            source_row = layer_rows_by_index.get(int(source), {})
            source_name = source_row.get("layer_name", f"model.{source}")
            source_channels = _parse_channels_value(source_row.get("out_channels"))
            if len(source_channels) != 1:
                source_channels = _parse_channels_value(source_row.get("out_channels_static"))
        width = int(source_channels[0]) if source_channels else 0
        segments.append(
            {
                "source": source_name,
                "source_index": source,
                "channels": width,
                "range_start": cursor,
                "range_end": cursor + width - 1 if width > 0 else cursor - 1,
            }
        )
        cursor += width
    return segments


def build_channel_topology(model: torch.nn.Module, layer_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    layer_rows_by_index = {int(row["layer_index"]): row for row in layer_rows}
    consumer_map: Dict[Any, List[str]] = defaultdict(list)
    for row in layer_rows:
        for source in row.get("from_indices", []):
            consumer_map[source].append(row["layer_name"])

    block_topology: List[Dict[str, Any]] = []
    residual_blocks: List[str] = []
    concat_blocks: List[str] = []
    detect_input_blocks: List[str] = []
    immutable_blocks: List[str] = []
    prune_candidates: List[str] = []
    channel_alignment_rules: List[Dict[str, Any]] = []

    for row in layer_rows:
        block_index = int(row["layer_index"])
        block_name = row["layer_name"]
        module = model.model[block_index]
        module_type = row["module_type"]
        block_info: Dict[str, Any] = {
            "block_index": block_index,
            "block_name": block_name,
            "module_type": module_type,
            "stage": row["stage"],
            "from_indices": row.get("from_indices", []),
            "consumers": consumer_map.get(block_index, []),
            "input_channels": _parse_channels_value(row.get("in_channels")),
            "output_channels": _parse_channels_value(row.get("out_channels")),
            "prune_allowed": bool(row.get("prune_allowed", False)),
            "dependency_group": row.get("dependency_group"),
        }

        if block_info["prune_allowed"]:
            prune_candidates.append(block_name)
        else:
            immutable_blocks.append(block_name)

        if module_type == "Conv":
            block_info["internal_topology"] = {
                "kind": "conv_block",
                "fused_runtime": not hasattr(module, "bn"),
                "output_channel_unit": int(module.conv.out_channels),
                "consumer_targets": consumer_map.get(block_index, []),
            }
            channel_alignment_rules.append(
                {
                    "rule_type": "conv_output_propagation",
                    "block_name": block_name,
                    "description": "Output channels propagate directly to all downstream consumer inputs.",
                }
            )

        elif module_type.startswith("C3"):
            hidden_channels = int(module.cv1.conv.out_channels)
            is_c3lite = any(hasattr(unit, "dw") and hasattr(unit, "pw") for unit in module.m)
            transform_units = []
            for unit_idx, unit in enumerate(module.m):
                unit_payload: Dict[str, Any] = {"unit_name": f"{block_name}.m.{unit_idx}"}
                if is_c3lite and hasattr(unit, "dw") and hasattr(unit, "pw"):
                    unit_payload.update(
                        {
                            "unit_type": type(unit).__name__,
                            "dw_in_channels": int(unit.dw.in_channels),
                            "dw_out_channels": int(unit.dw.out_channels),
                            "dw_groups": int(unit.dw.groups),
                            "pw_in_channels": int(unit.pw.in_channels),
                            "pw_out_channels": int(unit.pw.out_channels),
                            "residual_add": False,
                        }
                    )
                else:
                    unit_payload.update(
                        {
                            "unit_type": type(unit).__name__,
                            "cv1_in_channels": int(unit.cv1.conv.in_channels),
                            "cv1_out_channels": int(unit.cv1.conv.out_channels),
                            "cv2_in_channels": int(unit.cv2.conv.in_channels),
                            "cv2_out_channels": int(unit.cv2.conv.out_channels),
                            "residual_add": bool(getattr(unit, "add", False)),
                        }
                    )
                transform_units.append(unit_payload)

            residual_sensitive = any(unit["residual_add"] for unit in transform_units)
            if residual_sensitive:
                residual_blocks.append(block_name)

            block_info["internal_topology"] = {
                "kind": "c3lite_csp_split_dwconv_merge" if is_c3lite else "c3_csp_split_merge",
                "input_channels": int(module.cv1.conv.in_channels),
                "hidden_channels": hidden_channels,
                "output_channels": int(module.cv3.conv.out_channels),
                "main_branch": {
                    "entry_conv": f"{block_name}.cv1",
                    "entry_channels": hidden_channels,
                    "transform_units": transform_units,
                },
                "skip_branch": {
                    "entry_conv": f"{block_name}.cv2",
                    "entry_channels": int(module.cv2.conv.out_channels),
                },
                "merge_conv": {
                    "name": f"{block_name}.cv3",
                    "input_channels": int(module.cv3.conv.in_channels),
                    "output_channels": int(module.cv3.conv.out_channels),
                },
                "internal_concat_segments": [
                    {"source": "main_branch", "channels": hidden_channels, "range_start": 0, "range_end": hidden_channels - 1},
                    {
                        "source": "skip_branch",
                        "channels": int(module.cv2.conv.out_channels),
                        "range_start": hidden_channels,
                        "range_end": hidden_channels + int(module.cv2.conv.out_channels) - 1,
                    },
                ],
            }
            channel_alignment_rules.append(
                {
                    "rule_type": "c3_hidden_branch_alignment",
                    "block_name": block_name,
                    "description": "C3-family main branch and skip branch feed a concatenation before cv3; channel changes must preserve cv3 input partitioning.",
                    "hidden_channels": hidden_channels,
                }
            )
            if residual_sensitive:
                channel_alignment_rules.append(
                    {
                        "rule_type": "residual_add_constraint",
                        "block_name": block_name,
                        "description": "Residual bottlenecks require identical tensor channel dimensions across the add path.",
                    }
                )

        elif module_type == "SPPF":
            hidden_channels = int(module.cv1.conv.out_channels)
            block_info["internal_topology"] = {
                "kind": "sppf_reused_pool_concat",
                "input_channels": int(module.cv1.conv.in_channels),
                "hidden_channels": hidden_channels,
                "output_channels": int(module.cv2.conv.out_channels),
                "pool_reuse_count": 3,
                "concat_segments": [
                    {"source": "cv1", "channels": hidden_channels, "range_start": 0, "range_end": hidden_channels - 1},
                    {"source": "pool_1", "channels": hidden_channels, "range_start": hidden_channels, "range_end": 2 * hidden_channels - 1},
                    {"source": "pool_2", "channels": hidden_channels, "range_start": 2 * hidden_channels, "range_end": 3 * hidden_channels - 1},
                    {"source": "pool_3", "channels": hidden_channels, "range_start": 3 * hidden_channels, "range_end": 4 * hidden_channels - 1},
                ],
            }
            channel_alignment_rules.append(
                {
                    "rule_type": "sppf_concat_constraint",
                    "block_name": block_name,
                    "description": "SPPF concatenates four equal hidden-channel tensors before cv2; hidden channel width drives the full merge width.",
                    "hidden_channels": hidden_channels,
                }
            )

        elif module_type == "Concat":
            segments = _source_segments_from_row(row, layer_rows_by_index)
            concat_blocks.append(block_name)
            block_info["internal_topology"] = {
                "kind": "cross_block_concat",
                "output_channel_segments": segments,
                "output_total_channels": int(sum(segment["channels"] for segment in segments)),
            }
            channel_alignment_rules.append(
                {
                    "rule_type": "concat_segment_tracking",
                    "block_name": block_name,
                    "description": "Concatenated output channels are partitioned by source block; later pruning must preserve segment provenance.",
                    "segments": segments,
                }
            )

        elif module_type == "Upsample":
            block_info["internal_topology"] = {
                "kind": "spatial_only_resize",
                "channel_change": "none",
            }

        elif module_type == "Detect":
            strides = [int(v) for v in getattr(module, "stride", [])]
            branch_info = []
            for branch_idx, (source, conv) in enumerate(zip(row.get("from_indices", []), module.m)):
                if isinstance(source, str):
                    source_name = source
                else:
                    source_name = layer_rows_by_index[int(source)]["layer_name"]
                    detect_input_blocks.append(source_name)
                branch_info.append(
                    {
                        "branch_index": branch_idx,
                        "source": source_name,
                        "input_channels": int(conv.in_channels),
                        "output_channels": int(conv.out_channels),
                        "stride": strides[branch_idx] if branch_idx < len(strides) else None,
                    }
                )
            block_info["internal_topology"] = {
                "kind": "detect_head_multi_branch",
                "nc": int(module.nc),
                "na": int(module.na),
                "no": int(module.no),
                "nl": int(module.nl),
                "branch_info": branch_info,
            }
            channel_alignment_rules.append(
                {
                    "rule_type": "detect_input_contract",
                    "block_name": block_name,
                    "description": "Each detect branch expects the exact source feature width wired during model build.",
                    "branches": branch_info,
                }
            )

        else:
            block_info["internal_topology"] = {"kind": "generic_block"}

        block_topology.append(block_info)

    topology = {
        "block_topology": block_topology,
        "consumer_map": {str(key): value for key, value in consumer_map.items()},
    }
    pruning_constraints = {
        "immutable_blocks": immutable_blocks,
        "prune_candidate_blocks": prune_candidates,
        "residual_sensitive_blocks": residual_blocks,
        "concat_blocks": concat_blocks,
        "detect_input_blocks": sorted(set(detect_input_blocks)),
        "channel_alignment_rules": channel_alignment_rules,
    }
    return topology, pruning_constraints
