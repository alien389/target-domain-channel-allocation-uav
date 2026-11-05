from __future__ import annotations

from typing import Any, Dict, List

import torch


def _dependency_group_for_layer(layer_row: Dict[str, Any]) -> str:
    module_type = layer_row["module_type"]
    layer_index = layer_row["layer_index"]
    if module_type == "Detect":
        return f"detect_head_{layer_index}"
    if module_type == "Concat":
        return f"concat_group_{layer_index}"
    if int(layer_row.get("residual_connections", 0)) > 0:
        return f"residual_block_{layer_index}"
    return f"forward_path_{layer_index}"


def build_dependency_graph(model: torch.nn.Module, layer_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    concat_groups: List[Dict[str, Any]] = []
    detect_dependencies: List[Dict[str, Any]] = []
    residual_groups: List[Dict[str, Any]] = []

    layer_by_index = {row["layer_index"]: row for row in layer_rows}

    for row in layer_rows:
        dependency_group = _dependency_group_for_layer(row)
        row["dependency_group"] = dependency_group

        nodes.append(
            {
                "layer_index": row["layer_index"],
                "layer_name": row["layer_name"],
                "module_type": row["module_type"],
                "stage": row["stage"],
                "dependency_group": dependency_group,
                "prune_allowed": row["prune_allowed"],
            }
        )

        for source in row["from_indices"]:
            relation = "forward"
            if row["module_type"] == "Concat":
                relation = "concat"
            elif row["module_type"] == "Detect":
                relation = "detect_input"
            edges.append({"source": source, "target": row["layer_index"], "relation": relation})

        if row["module_type"] == "Concat":
            concat_groups.append(
                {
                    "group_id": dependency_group,
                    "target_layer": row["layer_index"],
                    "sources": row["from_indices"],
                }
            )

        if row["module_type"] == "Detect":
            detect_dependencies.append(
                {
                    "group_id": dependency_group,
                    "detect_layer": row["layer_index"],
                    "sources": row["from_indices"],
                    "branch_count": row.get("detect_branches", 0),
                }
            )

        if int(row.get("residual_connections", 0)) > 0:
            residual_groups.append(
                {
                    "group_id": dependency_group,
                    "layer_index": row["layer_index"],
                    "module_type": row["module_type"],
                    "residual_connections": int(row["residual_connections"]),
                }
            )

    stage_summary: Dict[str, int] = {}
    for row in layer_rows:
        stage_summary[row["stage"]] = stage_summary.get(row["stage"], 0) + 1

    return {
        "nodes": nodes,
        "edges": edges,
        "concat_groups": concat_groups,
        "detect_dependencies": detect_dependencies,
        "residual_groups": residual_groups,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "concat_group_count": len(concat_groups),
            "detect_dependency_count": len(detect_dependencies),
            "residual_group_count": len(residual_groups),
            "stage_summary": stage_summary,
        },
    }

