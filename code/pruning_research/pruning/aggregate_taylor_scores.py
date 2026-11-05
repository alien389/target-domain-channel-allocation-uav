from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pruning_research.pruning.importance_scores import rank_channel_scores
from pruning_research.utils.io_utils import ensure_dir, save_csv, save_json


def load_run_scores(path: str | Path) -> Dict[str, Any]:
    run_path = Path(path).resolve()
    return json.loads(run_path.read_text(encoding="utf-8"))


def _discover_run_jsons(input_root: str | Path) -> List[Path]:
    root = Path(input_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Taylor input root does not exist: {root}")

    direct_file = root / "raw_taylor_scores.json"
    if direct_file.exists():
        return [direct_file]

    discovered = sorted(root.glob("**/raw_taylor_scores.json"))
    if not discovered:
        raise FileNotFoundError(f"No raw_taylor_scores.json files were found under: {root}")
    return discovered


def aggregate_scores(run_payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not run_payloads:
        raise ValueError("No Taylor run payloads were provided for aggregation.")

    aggregated_bank: Dict[str, Any] = {}
    run_sources: List[str] = []

    for payload in run_payloads:
        run_sources.append(str(payload.get("raw_taylor_scores_json", payload.get("source_path", "unknown"))))
        for block_name, block_payload in payload["importance_bank"].items():
            block_state = aggregated_bank.setdefault(
                block_name,
                {
                    "block_name": block_name,
                    "stage": block_payload["stage"],
                    "module_type": block_payload["module_type"],
                    "scope_accumulators": {},
                },
            )
            for scope_name, scope_payload in block_payload["scope_rankings"].items():
                scope_scores = [float(value) for value in scope_payload["scores"]]
                scope_state = block_state["scope_accumulators"].setdefault(
                    scope_name,
                    {
                        "scope_type": scope_payload["scope_type"],
                        "anchor_module": scope_payload["anchor_module"],
                        "width": int(scope_payload["width"]),
                        "score_sum": [0.0 for _ in scope_scores],
                        "run_count": 0,
                    },
                )
                if int(scope_state["width"]) != len(scope_scores):
                    raise ValueError(
                        f"Inconsistent Taylor scope width for {block_name}/{scope_name}: "
                        f"{scope_state['width']} vs {len(scope_scores)}"
                    )
                scope_state["score_sum"] = [
                    float(existing + current) for existing, current in zip(scope_state["score_sum"], scope_scores)
                ]
                scope_state["run_count"] += 1

    normalized_bank: Dict[str, Any] = {}
    for block_name, block_state in aggregated_bank.items():
        scope_rankings: Dict[str, Any] = {}
        for scope_name, scope_state in block_state["scope_accumulators"].items():
            run_count = int(scope_state["run_count"])
            mean_scores = [float(score / run_count) for score in scope_state["score_sum"]]
            max_score = max(mean_scores) if mean_scores else 0.0
            normalized_scores = [float(score / max_score) if max_score > 0 else 0.0 for score in mean_scores]
            ranked_scope = rank_channel_scores(
                scores=mean_scores,
                scope_type=str(scope_state["scope_type"]),
                anchor_module=str(scope_state["anchor_module"]),
            )
            ranked_scope["scores"] = mean_scores
            ranked_scope["normalized_scores"] = normalized_scores
            ranked_scope["run_count"] = run_count
            scope_rankings[scope_name] = ranked_scope

        normalized_bank[block_name] = {
            "block_name": block_name,
            "stage": block_state["stage"],
            "module_type": block_state["module_type"],
            "scope_rankings": scope_rankings,
        }

    reference = run_payloads[0]
    return {
        "model_id": reference.get("model_id"),
        "weights": reference.get("weights"),
        "importance_metric": "taylor_first_order",
        "aggregation_method": "mean_across_runs",
        "score_reduction": reference.get("score_reduction"),
        "source_runs": run_sources,
        "importance_bank": normalized_bank,
    }


def normalize_scores(aggregated_payload: Dict[str, Any]) -> Dict[str, Any]:
    return aggregated_payload


def export_aggregated_scores(aggregated_payload: Dict[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    target_dir = ensure_dir(Path(output_dir).resolve())

    scope_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    for block_name, block_payload in aggregated_payload["importance_bank"].items():
        for scope_name, scope_payload in block_payload["scope_rankings"].items():
            scope_rows.append(
                {
                    "block_name": block_name,
                    "stage": block_payload["stage"],
                    "module_type": block_payload["module_type"],
                    "scope_name": scope_name,
                    "anchor_module": scope_payload["anchor_module"],
                    "width": int(scope_payload["width"]),
                    "run_count": int(scope_payload.get("run_count", 0)),
                    "min_score": scope_payload["min_score"],
                    "mean_score": scope_payload["mean_score"],
                    "max_score": scope_payload["max_score"],
                }
            )

            rank_map = {
                int(channel_index): rank + 1
                for rank, channel_index in enumerate(scope_payload["indices_low_to_high"])
            }
            for channel_index, channel_score in enumerate(scope_payload.get("scores", [])):
                normalized_scores = scope_payload.get("normalized_scores", [])
                channel_rows.append(
                    {
                        "block_name": block_name,
                        "stage": block_payload["stage"],
                        "module_type": block_payload["module_type"],
                        "scope_name": scope_name,
                        "anchor_module": scope_payload["anchor_module"],
                        "channel_index": int(channel_index),
                        "score": float(channel_score),
                        "normalized_score": float(normalized_scores[channel_index]) if normalized_scores else 0.0,
                        "rank_low_to_high": int(rank_map[int(channel_index)]),
                    }
                )

    aggregated_json = save_json(aggregated_payload, target_dir / "aggregated_taylor_importance.json")
    scope_csv = save_csv(scope_rows, target_dir / "aggregated_taylor_importance.csv")
    channel_csv = save_csv(channel_rows, target_dir / "aggregated_taylor_channel_scores.csv")
    summary = {
        "model_id": aggregated_payload.get("model_id"),
        "aggregated_taylor_importance_json": str(aggregated_json),
        "aggregated_taylor_importance_csv": str(scope_csv),
        "aggregated_taylor_channel_scores_csv": str(channel_csv),
        "run_count": len(aggregated_payload.get("source_runs", [])),
        "output_dir": str(target_dir),
    }
    summary_json = save_json(summary, target_dir / "aggregated_taylor_importance_summary.json")
    summary["aggregated_taylor_importance_summary_json"] = str(summary_json)
    return summary


def aggregate_taylor_runs(input_root: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    run_jsons = _discover_run_jsons(input_root)
    run_payloads = []
    for run_json in run_jsons:
        payload = load_run_scores(run_json)
        payload["source_path"] = str(run_json)
        run_payloads.append(payload)

    aggregated_payload = aggregate_scores(run_payloads)
    normalized_payload = normalize_scores(aggregated_payload)
    return export_aggregated_scores(normalized_payload, output_dir=output_dir)
