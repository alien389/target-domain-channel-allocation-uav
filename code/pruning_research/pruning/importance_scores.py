from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_block_index(block_name: str) -> int:
    return int(str(block_name).split(".")[-1])


def rank_channel_scores(scores: Sequence[float], scope_type: str, anchor_module: str) -> Dict[str, Any]:
    numeric_scores = [float(score) for score in scores]
    ordered = sorted(range(len(numeric_scores)), key=lambda idx: (numeric_scores[idx], idx))
    return {
        "scope_type": scope_type,
        "anchor_module": anchor_module,
        "width": len(numeric_scores),
        "indices_low_to_high": ordered,
        "scores": numeric_scores,
        "scores_low_to_high": [numeric_scores[idx] for idx in ordered],
        "min_score": min(numeric_scores) if numeric_scores else None,
        "mean_score": mean(numeric_scores) if numeric_scores else None,
        "max_score": max(numeric_scores) if numeric_scores else None,
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _l1_out_scores(conv: torch.nn.Conv2d) -> List[float]:
    weights = conv.weight.detach().float().abs()
    return weights.sum(dim=(1, 2, 3)).cpu().tolist()


def _l1_in_scores(conv: torch.nn.Conv2d) -> List[float]:
    weights = conv.weight.detach().float().abs()
    return weights.sum(dim=(0, 2, 3)).cpu().tolist()


def _add_scores(left: Sequence[float], right: Sequence[float]) -> List[float]:
    return [float(a) + float(b) for a, b in zip(left, right)]


def _is_c3lite_module(module: torch.nn.Module) -> bool:
    return type(module).__name__ == "C3Lite" or any(hasattr(unit, "dw") and hasattr(unit, "pw") for unit in getattr(module, "m", []))


def _c3lite_main_hidden_scores(module: torch.nn.Module) -> List[float]:
    hidden = int(module.cv1.conv.out_channels)
    cv3_in = _l1_in_scores(module.cv3.conv)
    main_scores = _l1_out_scores(module.cv1.conv)

    for unit in module.m:
        if not (hasattr(unit, "dw") and hasattr(unit, "pw")):
            raise TypeError(f"Expected DWConvLite-like unit inside C3Lite, got {type(unit).__name__}")
        main_scores = _add_scores(main_scores, _l1_out_scores(unit.dw))
        main_scores = _add_scores(main_scores, _l1_in_scores(unit.pw))
        main_scores = _add_scores(main_scores, _l1_out_scores(unit.pw))

    return _add_scores(main_scores, cv3_in[:hidden])


def _deterministic_random_scores(width: int, base_seed: int, block_name: str, scope_name: str) -> List[float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(base_seed, block_name, scope_name))
    return torch.rand(int(width), generator=generator, dtype=torch.float32).cpu().tolist()


def iter_prunable_topology_rows(
    channel_topology: Dict[str, Any],
    pruning_constraints: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    topology_rows = list(channel_topology.get("block_topology", []))
    if not pruning_constraints:
        return [row for row in topology_rows if _as_bool(row.get("prune_allowed", False))]

    allowed = {str(name) for name in pruning_constraints.get("prune_candidate_blocks", [])}
    return [row for row in topology_rows if str(row.get("block_name")) in allowed]


def resolve_scope_modules(model: torch.nn.Module, topology_row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    block_name = str(topology_row["block_name"])
    block_index = int(topology_row["block_index"])
    module_type = str(topology_row["module_type"])
    module = model.model[block_index]

    if module_type == "Conv":
        return {
            "output": {
                "scope_type": "output",
                "anchor_module": f"{block_name}.conv",
                "module": module,
                "width": int(module.conv.out_channels),
            }
        }

    if module_type.startswith("C3"):
        return {
            "main_hidden": {
                "scope_type": "main_hidden",
                "anchor_module": f"{block_name}.cv1.conv",
                "module": module.cv1,
                "width": int(module.cv1.conv.out_channels),
            },
            "skip_hidden": {
                "scope_type": "skip_hidden",
                "anchor_module": f"{block_name}.cv2.conv",
                "module": module.cv2,
                "width": int(module.cv2.conv.out_channels),
            },
            "output": {
                "scope_type": "output",
                "anchor_module": f"{block_name}.cv3.conv",
                "module": module.cv3,
                "width": int(module.cv3.conv.out_channels),
            },
        }

    if module_type == "SPPF":
        return {
            "hidden": {
                "scope_type": "hidden",
                "anchor_module": f"{block_name}.cv1.conv",
                "module": module.cv1,
                "width": int(module.cv1.conv.out_channels),
            },
            "output": {
                "scope_type": "output",
                "anchor_module": f"{block_name}.cv2.conv",
                "module": module.cv2,
                "width": int(module.cv2.conv.out_channels),
            },
        }

    raise KeyError(f"Unsupported prunable module type for scope resolution: {module_type}")


def build_runtime_l1_importance_bank(
    model: torch.nn.Module,
    channel_topology: Dict[str, Any],
    pruning_constraints: Dict[str, Any],
) -> Dict[str, Any]:
    importance_bank: Dict[str, Any] = {}

    for topology_row in iter_prunable_topology_rows(channel_topology, pruning_constraints):
        block_name = str(topology_row["block_name"])
        block_index = _parse_block_index(block_name)
        module_type = str(topology_row["module_type"])
        module = model.model[block_index]

        if module_type == "Conv":
            scores = _l1_out_scores(module.conv)
            scope_rankings = {
                "output": rank_channel_scores(scores=scores, scope_type="output", anchor_module=f"{block_name}.conv")
            }
        elif module_type.startswith("C3"):
            hidden = int(module.cv1.conv.out_channels)
            cv3_in = _l1_in_scores(module.cv3.conv)

            if _is_c3lite_module(module):
                main_scores = _c3lite_main_hidden_scores(module)
            else:
                main_scores = _l1_out_scores(module.cv1.conv)
                for bottleneck in module.m:
                    main_scores = _add_scores(main_scores, _l1_out_scores(bottleneck.cv1.conv))
                    main_scores = _add_scores(main_scores, _l1_out_scores(bottleneck.cv2.conv))
                main_scores = _add_scores(main_scores, cv3_in[:hidden])

            skip_scores = _l1_out_scores(module.cv2.conv)
            skip_scores = _add_scores(skip_scores, cv3_in[hidden : 2 * hidden])

            output_scores = _l1_out_scores(module.cv3.conv)
            scope_rankings = {
                "main_hidden": rank_channel_scores(
                    scores=main_scores,
                    scope_type="main_hidden",
                    anchor_module=f"{block_name}.cv1.conv",
                ),
                "skip_hidden": rank_channel_scores(
                    scores=skip_scores,
                    scope_type="skip_hidden",
                    anchor_module=f"{block_name}.cv2.conv",
                ),
                "output": rank_channel_scores(
                    scores=output_scores,
                    scope_type="output",
                    anchor_module=f"{block_name}.cv3.conv",
                ),
            }
        elif module_type == "SPPF":
            hidden = int(module.cv1.conv.out_channels)
            cv2_in = _l1_in_scores(module.cv2.conv)
            hidden_scores = _l1_out_scores(module.cv1.conv)
            for repeat in range(4):
                start = repeat * hidden
                stop = start + hidden
                hidden_scores = [a + b for a, b in zip(hidden_scores, cv2_in[start:stop])]

            output_scores = _l1_out_scores(module.cv2.conv)
            scope_rankings = {
                "hidden": rank_channel_scores(
                    scores=hidden_scores,
                    scope_type="hidden",
                    anchor_module=f"{block_name}.cv1.conv",
                ),
                "output": rank_channel_scores(
                    scores=output_scores,
                    scope_type="output",
                    anchor_module=f"{block_name}.cv2.conv",
                ),
            }
        else:
            raise KeyError(f"Unsupported runtime L1 module type: {module_type}")

        importance_bank[block_name] = {
            "block_name": block_name,
            "stage": str(topology_row["stage"]),
            "module_type": module_type,
            "scope_rankings": scope_rankings,
        }

    return {
        "importance_metric": "l1_weight_proxy",
        "importance_source": "runtime_l1_weight_proxy",
        "importance_bank": importance_bank,
    }


def build_runtime_random_importance_bank(
    model: torch.nn.Module,
    channel_topology: Dict[str, Any],
    pruning_constraints: Dict[str, Any],
    search_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    base_seed = int(search_cfg.get("random_seed", search_cfg.get("seed", 42)))
    importance_bank: Dict[str, Any] = {}

    for topology_row in iter_prunable_topology_rows(channel_topology, pruning_constraints):
        block_name = str(topology_row["block_name"])
        scope_rankings: Dict[str, Any] = {}

        for scope_name, scope_info in resolve_scope_modules(model, topology_row).items():
            scores = _deterministic_random_scores(
                width=int(scope_info["width"]),
                base_seed=base_seed,
                block_name=block_name,
                scope_name=scope_name,
            )
            scope_rankings[scope_name] = rank_channel_scores(
                scores=scores,
                scope_type=str(scope_info["scope_type"]),
                anchor_module=str(scope_info["anchor_module"]),
            )

        importance_bank[block_name] = {
            "block_name": block_name,
            "stage": str(topology_row["stage"]),
            "module_type": str(topology_row["module_type"]),
            "scope_rankings": scope_rankings,
        }

    return {
        "importance_metric": "random_channel_proxy",
        "importance_source": f"deterministic_random_seed:{base_seed}",
        "importance_bank": importance_bank,
    }


def _resolve_importance_json_path(search_cfg: Dict[str, Any]) -> Path:
    explicit_json = str(search_cfg.get("importance_json", "")).strip()
    if explicit_json:
        path = Path(explicit_json).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured importance_json does not exist: {path}")
        return path

    explicit_dir = str(search_cfg.get("importance_dir", "")).strip()
    if explicit_dir:
        root = Path(explicit_dir).resolve()
        candidates = [
            root / "aggregated_taylor_importance.json",
            root / "aggregated" / "aggregated_taylor_importance.json",
            root / "aggregated_importance_scores.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve aggregated Taylor importance JSON under: {root}")

    raise FileNotFoundError("Taylor importance requested, but neither importance_json nor importance_dir was provided.")


def load_importance_bank_from_json(path: str | Path) -> Dict[str, Any]:
    importance_path = Path(path).resolve()
    payload = json.loads(importance_path.read_text(encoding="utf-8"))
    raw_bank = payload.get("importance_bank", payload)

    normalized_bank: Dict[str, Any] = {}
    for block_name, block_payload in raw_bank.items():
        scope_rankings: Dict[str, Any] = {}
        for scope_name, scope_payload in dict(block_payload.get("scope_rankings", {})).items():
            scores = [float(value) for value in scope_payload.get("scores", [])]
            width = int(scope_payload.get("width", len(scores)))
            indices = list(scope_payload.get("indices_low_to_high", []))
            if not indices:
                if not scores:
                    raise ValueError(
                        f"Importance scope {block_name}/{scope_name} does not contain scores or indices_low_to_high."
                    )
                indices = sorted(range(len(scores)), key=lambda idx: (scores[idx], idx))
            scope_rankings[scope_name] = {
                "scope_type": str(scope_payload.get("scope_type", scope_name)),
                "anchor_module": str(scope_payload.get("anchor_module", "")),
                "width": width,
                "indices_low_to_high": [int(idx) for idx in indices],
                "scores": scores,
                "scores_low_to_high": [
                    float(value)
                    for value in scope_payload.get(
                        "scores_low_to_high",
                        [scores[idx] for idx in indices] if scores else [],
                    )
                ],
                "min_score": float(scope_payload.get("min_score", min(scores) if scores else 0.0)),
                "mean_score": float(scope_payload.get("mean_score", mean(scores) if scores else 0.0)),
                "max_score": float(scope_payload.get("max_score", max(scores) if scores else 0.0)),
            }

        normalized_bank[block_name] = {
            "block_name": str(block_payload.get("block_name", block_name)),
            "stage": str(block_payload.get("stage", "")),
            "module_type": str(block_payload.get("module_type", "")),
            "scope_rankings": scope_rankings,
        }

    return {
        "importance_metric": str(payload.get("importance_metric", "external_importance")),
        "importance_source": str(importance_path),
        "importance_bank": normalized_bank,
    }


def build_importance_bank(
    model: torch.nn.Module,
    channel_topology: Dict[str, Any],
    pruning_constraints: Dict[str, Any],
    search_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    metric = str(search_cfg.get("importance_metric", "l1_weight_proxy")).strip().lower()

    if metric in {"l1", "l1_weight", "l1_weight_proxy", "default"}:
        return build_runtime_l1_importance_bank(
            model=model,
            channel_topology=channel_topology,
            pruning_constraints=pruning_constraints,
        )

    if metric in {"taylor", "taylor_first_order", "taylor_first_order_mean_abs"}:
        importance_json = _resolve_importance_json_path(search_cfg)
        importance_result = load_importance_bank_from_json(importance_json)
        importance_result["importance_metric"] = "taylor_first_order"
        return importance_result

    if metric in {"random", "random_control", "random_channel_proxy"}:
        return build_runtime_random_importance_bank(
            model=model,
            channel_topology=channel_topology,
            pruning_constraints=pruning_constraints,
            search_cfg=search_cfg,
        )

    raise KeyError(f"Unsupported importance_metric: {search_cfg.get('importance_metric')}")
