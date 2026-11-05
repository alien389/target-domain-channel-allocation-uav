from __future__ import annotations

from typing import Any, Dict, List


def _structural_role(row: Dict[str, Any]) -> str:
    stage = row["stage"]
    module_type = row["module_type"]
    stride = str(row.get("stride_static") or "")

    if stage == "backbone":
        if module_type == "Conv":
            return "backbone_downsample_conv" if "2x2" in stride else "backbone_transition_conv"
        if module_type.startswith("C3"):
            return "backbone_c3"
        if module_type == "SPPF":
            return "backbone_sppf"
    if stage == "neck":
        if module_type == "Upsample":
            return "neck_upsample"
        if module_type == "Concat":
            return "neck_concat"
        if module_type == "Conv":
            return "neck_downsample_conv" if "2x2" in stride else "neck_transition_conv"
        if module_type.startswith("C3"):
            return "neck_fusion_c3"
    if stage == "head" and module_type == "Detect":
        return "detect_head"
    return f"{stage}_{module_type.lower()}"


def build_block_profile(layer_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total_params = sum(int(row.get("params", 0)) for row in layer_rows)
    total_flops_g = sum(float(row.get("flops_g", 0.0) or 0.0) for row in layer_rows)
    stage_counters: Dict[str, int] = {}

    block_rows: List[Dict[str, Any]] = []
    for row in layer_rows:
        params = int(row.get("params", 0))
        flops_g = float(row.get("flops_g", 0.0) or 0.0)
        stage = row["stage"]
        prune_allowed = bool(row.get("prune_allowed", False))
        structural_role = _structural_role(row)
        stage_counters[stage] = stage_counters.get(stage, 0) + 1
        stage_block_index = stage_counters[stage]

        if not prune_allowed:
            prune_priority_candidate = "frozen"
        elif stage == "neck":
            prune_priority_candidate = "high"
        elif stage == "backbone":
            prune_priority_candidate = "medium"
        else:
            prune_priority_candidate = "low"

        block_rows.append(
            {
                "block_name": row["layer_name"],
                "block_label": f"{structural_role}_{stage_block_index:02d}",
                "layer_index": row["layer_index"],
                "module_type": row["module_type"],
                "stage": stage,
                "stage_block_index": stage_block_index,
                "structural_role": structural_role,
                "params": params,
                "flops_g": flops_g,
                "params_share_pct": (params / total_params * 100.0) if total_params else None,
                "flops_share_pct": (flops_g / total_flops_g * 100.0) if total_flops_g else None,
                "in_channels": row.get("in_channels"),
                "out_channels": row.get("out_channels"),
                "channels_signature": f"{row.get('in_channels')}->{row.get('out_channels')}",
                "dependency_group": row.get("dependency_group"),
                "prune_allowed": prune_allowed,
                "prune_priority_candidate": prune_priority_candidate,
            }
        )
    return block_rows
