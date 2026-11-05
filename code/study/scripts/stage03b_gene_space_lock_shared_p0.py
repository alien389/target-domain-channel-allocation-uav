from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import csv
import hashlib
import importlib.util
import json
import math
import shutil
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE01 = PROJECT / "01_data_split"
AMENDMENT = PROJECT / "01A_protocol_amendment"
STAGE03A = PROJECT / "03a_taylor_dependency_gene_space"
R1 = STAGE03A / "gene_space_R1"
OUTPUT = PROJECT / "03b_search_initialization"
SCRIPT_PATH = PROJECT / "scripts" / "stage03b_gene_space_lock_shared_p0.py"

GENE_JSON = R1 / "gene_space_manifest_R1.json"
GENE_XLSX = R1 / "gene_space_manifest_R1.xlsx"
DEPENDENCY_GRAPH = STAGE03A / "dependency_graph.json"
TAYLOR_RANKING = STAGE03A / "taylor_ranking_manifest.json"
TAYLOR_CHANNELS = STAGE03A / "normalized_channel_taylor_scores.csv"
FEASIBLE_ARCHIVE = R1 / "feasible_signatures_verified.csv"
GLOBAL_TAYLOR = R1 / "global_taylor_trajectory.csv"
TARGETMANUAL = R1 / "targetmanual_candidate.json"
R1_VALIDATION = R1 / "revision_validation.json"
R1_STATE = R1 / "revision_state.json"
STAGE03A_STATE = STAGE03A / "stage03a_state.json"

SPLIT_HASH_SOURCE = STAGE01 / "split_hashes.json"
TRAINING_PROTOCOL_SOURCE = AMENDMENT / "revised_frozen_hyperparameters.yaml"
STAGE01_STATE = STAGE01 / "stage01_state.json"

REPLICATES = {
    1042: {"p0_seed": 104200, "ap_aps": 104201, "ap_only": 104202, "random": 104203},
    2042: {"p0_seed": 204200, "ap_aps": 204201, "ap_only": 204202, "random": 204203},
    3042: {"p0_seed": 304200, "ap_aps": 304201, "ap_only": 304202, "random": 304203},
}
METHODS = (
    "NSGA-II AP+APS",
    "Evolutionary AP-only",
    "Warm-started constrained Random Search",
)
TRAINING_SEED = 42
CHANNEL_QUANTUM = 8
NEIGHBOR_DELTAS = (-16, -8, 8, 16)


def load_r1_module():
    path = PROJECT / "scripts" / "stage03a_r1_gene_space_repair.py"
    spec = importlib.util.spec_from_file_location("stage03a_r1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import R1 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1M = load_r1_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_signature(group_ids: list[str], widths: tuple[int, ...]) -> str:
    return ";".join(f"{key}={value}" for key, value in zip(group_ids, widths))


def signature_sha(group_ids: list[str], widths: tuple[int, ...]) -> str:
    return hash_text(canonical_signature(group_ids, widths))


def parse_widths(row: dict[str, Any], group_ids: list[str]) -> tuple[int, ...]:
    return tuple(int(row[group_id]) for group_id in group_ids)


def source_snapshot() -> dict[str, str]:
    paths = [
        GENE_JSON,
        GENE_XLSX,
        DEPENDENCY_GRAPH,
        TAYLOR_RANKING,
        TAYLOR_CHANNELS,
        FEASIBLE_ARCHIVE,
        GLOBAL_TAYLOR,
        TARGETMANUAL,
        R1_VALIDATION,
        R1_STATE,
        STAGE03A_STATE,
        SPLIT_HASH_SOURCE,
        TRAINING_PROTOCOL_SOURCE,
        STAGE01_STATE,
        R1M.SEARCH_PARENT,
    ]
    return {str(path): sha256(path) for path in paths}


def lock_inputs(
    gene_payload: dict[str, Any],
    approved_archive: Path,
) -> dict[str, Any]:
    groups = gene_payload["groups"]
    payload = {
        "created_utc": utc_now(),
        "lock_name": "Stage03B Gene-Space Lock",
        "approval_token_consumed": "APPROVE_STAGE03A_GENE_SPACE",
        "gene_space_version": "R1",
        "dependency_group_count": len(groups),
        "genome_definition": "retained width per dependency group only",
        "channel_identity_definition": "approved target-domain Taylor ranking only",
        "channel_quantum": gene_payload["channel_quantum"],
        "budget": gene_payload["budget"],
        "groups": [
            {
                "group_id": row["group_id"],
                "block_name": row["block_name"],
                "scope_name": row["scope_name"],
                "protection_class": row["protection_class"],
                "exact_minimum_width": row["exact_minimum_width"],
                "exact_maximum_width": row["exact_maximum_width"],
                "allowed_widths": row["allowed_widths"],
                "channel_quantum": row["channel_quantum"],
            }
            for row in groups
        ],
        "locked_sources": {
            "gene_space_manifest_R1_json": {
                "path": str(GENE_JSON),
                "sha256": sha256(GENE_JSON),
            },
            "gene_space_manifest_R1_xlsx": {
                "path": str(GENE_XLSX),
                "sha256": sha256(GENE_XLSX),
            },
            "dependency_graph": {
                "path": str(DEPENDENCY_GRAPH),
                "sha256": sha256(DEPENDENCY_GRAPH),
            },
            "approved_taylor_ranking": {
                "path": str(TAYLOR_RANKING),
                "sha256": sha256(TAYLOR_RANKING),
            },
            "approved_taylor_channel_scores": {
                "path": str(TAYLOR_CHANNELS),
                "sha256": sha256(TAYLOR_CHANNELS),
            },
            "verified_feasible_archive": {
                "path": str(approved_archive),
                "sha256": sha256(approved_archive),
            },
            "global_taylor_trajectory": {
                "path": str(GLOBAL_TAYLOR),
                "sha256": sha256(GLOBAL_TAYLOR),
            },
            "searchparent_checkpoint": {
                "path": str(R1M.SEARCH_PARENT),
                "sha256": sha256(R1M.SEARCH_PARENT),
            },
        },
        "immutable_after_lock": [
            "allowed widths",
            "protection classes",
            "width floors",
            "budget limits",
            "Taylor ranking",
            "dependency topology",
        ],
        "future_change_policy": "A new protocol amendment is required; overwrite is prohibited.",
        "status": "PASS",
    }
    path = OUTPUT / "gene_space_lock.json"
    json_write(path, payload)
    text_write(OUTPUT / "gene_space_lock.sha256", f"{sha256(path)}  {path.name}\n")
    return payload


def make_reference(
    name: str,
    role: str,
    groups: list[dict[str, Any]],
    widths: tuple[int, ...],
    complexity_model,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_ids = [row["group_id"] for row in groups]
    metrics = complexity_model.calculate(dict(zip(group_ids, widths)))
    payload = {
        "created_utc": utc_now(),
        "name": name,
        "role": role,
        "signature_sha256": signature_sha(group_ids, widths),
        "architecture_signature": canonical_signature(group_ids, widths),
        "dependency_group_count": len(widths),
        "widths": dict(zip(group_ids, widths)),
        "fused_params": metrics["params"],
        "fused_flops_g": metrics["flops_g"],
        "search_parent_checkpoint": str(R1M.SEARCH_PARENT),
        "search_parent_sha256": sha256(R1M.SEARCH_PARENT),
        "channel_selection": "approved target-domain Taylor ranking",
        "training_performed": False,
        "status": "LOCKED",
    }
    if extra:
        payload.update(extra)
    return payload


def write_references(
    groups: list[dict[str, Any]],
    complexity_model,
    target_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    references = {
        "searchparent_original": make_reference(
            "SearchParent original signature",
            "parent_reference_only",
            groups,
            tuple(int(row["original_width"]) for row in groups),
            complexity_model,
        ),
        "stage1_allocation_anchor": make_reference(
            "Stage1-Allocation-Anchor",
            "historical_width_allocation_reference",
            groups,
            tuple(int(row["stage1_width"]) for row in groups),
            complexity_model,
            {
                "old_source_checkpoint_used_as_search_candidate": False,
                "old_source_checkpoint_context_only": True,
            },
        ),
        "stage6_allocation_anchor": make_reference(
            "Stage6-Allocation-Anchor",
            "shared_p0_anchor_using_current_parent",
            groups,
            tuple(int(row["stage6_width"]) for row in groups),
            complexity_model,
            {
                "old_stage6_checkpoint_used_as_search_candidate": False,
                "old_stage6_checkpoint_context_only": str(
                    R1M.ANCHORS["Stage-6"]["checkpoint"]
                ),
                "old_stage6_checkpoint_sha256_context_only": R1M.ANCHORS[
                    "Stage-6"
                ]["sha256"],
            },
        ),
        "stage10_allocation_anchor": make_reference(
            "Stage10-Allocation-Anchor",
            "historical_width_allocation_reference",
            groups,
            tuple(int(row["stage10_width"]) for row in groups),
            complexity_model,
            {
                "old_source_checkpoint_used_as_search_candidate": False,
                "old_source_checkpoint_context_only": True,
            },
        ),
        "targetmanual_b6_reference": make_reference(
            "TargetManual-B6-reference",
            "shared_p0_anchor_using_current_parent",
            groups,
            tuple(int(target_payload["widths"][row["group_id"]]) for row in groups),
            complexity_model,
            {
                "selection_source": str(TARGETMANUAL),
                "trajectory_step": target_payload["trajectory_step"],
            },
        ),
    }
    root = OUTPUT / "reference_signatures"
    for filename, payload in references.items():
        json_write(root / f"{filename}.json", payload)
    return references


def construct_signature(
    label: str,
    groups: list[dict[str, Any]],
    widths: tuple[int, ...],
    taylor_bank: dict[str, dict[str, Any]],
    complexity_model,
    final_checkpoint: Path | None = None,
) -> dict[str, Any]:
    from pruning_research.pruning.structured_pruner import apply_structured_pruning

    short_id = hash_text(label)[:10].lower()
    scratch = OUTPUT / "_s" / short_id
    plan_path = scratch / "p.json"
    R1M.json_write(
        plan_path,
        R1M.build_pruning_plan(label, groups, widths, taylor_bank),
    )
    result = apply_structured_pruning(
        weights=str(R1M.SEARCH_PARENT),
        yolov5_root=str(R1M.YOLOV5_ROOT),
        plan_json=str(plan_path),
        profile_dir=str(R1M.PROFILE_DIR),
        output_dir=str(scratch / "r"),
        weights_output_dir=str(scratch / "w"),
        model_id="s3b",
        pruned_model_id=f"s3b_{short_id}",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        validate=True,
        validate_imgsz=R1M.IMGSZ,
    )
    generated = Path(result["pruned_weights"])
    if final_checkpoint is not None:
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, final_checkpoint)
        checkpoint = final_checkpoint
    else:
        checkpoint = generated
    physical = R1M.physical_complexity(checkpoint)
    analytic = complexity_model.calculate(
        dict(zip([row["group_id"] for row in groups], widths))
    )
    test = {
        "signature_label": label,
        "signature_sha256": signature_sha(
            [row["group_id"] for row in groups], widths
        ),
        "fused_params_expected": analytic["params"],
        "fused_params_actual": physical["fused_params"],
        "params_match": analytic["params"] == physical["fused_params"],
        "fused_flops_g_expected": analytic["flops_g"],
        "fused_flops_g_actual": physical["flops_g"],
        "flops_match": math.isclose(
            analytic["flops_g"],
            physical["flops_g"],
            rel_tol=0.0,
            abs_tol=R1M.FLOPS_TOLERANCE_G,
        ),
        "nc": physical["nc"],
        "input_shape": "1x3x640x640",
        "detect_scales": physical["detect_scales"],
        "nonfinite_values": physical["nonfinite_values"],
        "in_memory_forward": result["validation"]["in_memory_forward"]["forward_ok"],
        "reloaded_forward": result["validation"]["reloaded_forward"]["forward_ok"],
        "raw_checkpoint": str(final_checkpoint) if final_checkpoint else "SMOKE_TEST_TEMPORARY",
        "raw_checkpoint_sha256": sha256(final_checkpoint)
        if final_checkpoint
        else physical["checkpoint_sha256"],
    }
    test["status"] = (
        "PASS"
        if test["params_match"]
        and test["flops_match"]
        and test["nc"] == 10
        and test["detect_scales"] == 3
        and test["nonfinite_values"] == 0
        and test["in_memory_forward"]
        and test["reloaded_forward"]
        else "FAIL"
    )
    resolved = scratch.resolve()
    expected_parent = (OUTPUT / "_s").resolve()
    if resolved.parent != expected_parent:
        raise RuntimeError(f"Unsafe scratch path: {resolved}")
    shutil.rmtree(resolved)
    if test["status"] != "PASS":
        raise RuntimeError(f"Structural reconstruction failed: {label}")
    return test


def write_channel_map(
    path: Path,
    groups: list[dict[str, Any]],
    widths: tuple[int, ...],
    taylor_bank: dict[str, dict[str, Any]],
) -> None:
    rows = []
    for group, retained_width in zip(groups, widths):
        original = int(group["original_width"])
        prune_count = original - retained_width
        ranking = taylor_bank[group["group_id"]]
        removed = ranking["indices_low_to_high"][:prune_count]
        removed_set = set(removed)
        retained = [index for index in range(original) if index not in removed_set]
        rank_by_index = {
            index: rank + 1
            for rank, index in enumerate(ranking["indices_low_to_high"])
        }
        rows.append(
            {
                "group_id": group["group_id"],
                "block_name": group["block_name"],
                "scope_name": group["scope_name"],
                "source_tensor": group["source_modules"],
                "dependent_tensors": group["dependent_modules"],
                "original_width": original,
                "retained_width": retained_width,
                "retained_channel_indices": ";".join(map(str, retained)),
                "retained_taylor_ranks": ";".join(
                    str(rank_by_index[index]) for index in retained
                ),
                "removed_channel_indices": ";".join(map(str, removed)),
                "removed_taylor_ranks": ";".join(
                    str(rank_by_index[index]) for index in removed
                ),
                "taylor_rank_source": str(TAYLOR_RANKING),
                "taylor_rank_source_sha256": sha256(TAYLOR_RANKING),
                "status": "PASS",
            }
        )
    csv_write(path, rows)


def metrics_for(
    widths: tuple[int, ...],
    group_ids: list[str],
    complexity_model,
) -> dict[str, Any]:
    return complexity_model.calculate(dict(zip(group_ids, widths)))


def is_in_budget(metrics: dict[str, Any], budget: dict[str, Any]) -> bool:
    return (
        int(budget["params_lower_inclusive"])
        <= int(metrics["params"])
        <= int(budget["params_upper_inclusive"])
        and float(budget["flops_lower_g_inclusive"])
        <= float(metrics["flops_g"])
        <= float(budget["flops_upper_g_inclusive"])
    )


def enumerate_neighbors(
    anchor: tuple[int, ...],
    groups: list[dict[str, Any]],
    complexity_model,
    budget: dict[str, Any],
) -> list[dict[str, Any]]:
    group_ids = [row["group_id"] for row in groups]
    allowed = [set(R1M.parse_allowed(row["allowed_widths"])) for row in groups]
    candidates: dict[tuple[int, ...], dict[str, Any]] = {}
    for gene_count in (1, 2):
        for indices in combinations(range(len(groups)), gene_count):
            for deltas in product(NEIGHBOR_DELTAS, repeat=gene_count):
                widths = list(anchor)
                valid = True
                for index, delta in zip(indices, deltas):
                    widths[index] += delta
                    if widths[index] not in allowed[index]:
                        valid = False
                        break
                if not valid:
                    continue
                value = tuple(widths)
                metrics = metrics_for(value, group_ids, complexity_model)
                if not is_in_budget(metrics, budget):
                    continue
                candidates[value] = {
                    "widths": value,
                    "changed_gene_count": gene_count,
                    "changed_genes": ";".join(group_ids[index] for index in indices),
                    "deltas": ";".join(map(str, deltas)),
                    "fused_params": metrics["params"],
                    "fused_flops_g": metrics["flops_g"],
                    "signature_sha256": signature_sha(group_ids, value),
                }
    return sorted(candidates.values(), key=lambda row: row["signature_sha256"])


def select_seeded_neighbors(
    anchor_name: str,
    candidates: list[dict[str, Any]],
    count: int,
    rng: np.random.Generator,
    excluded: set[str],
) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if row["signature_sha256"] not in excluded]
    if len(eligible) < count:
        raise RuntimeError(
            f"{anchor_name} has only {len(eligible)} eligible neighbors."
        )
    selected_indices = rng.choice(len(eligible), size=count, replace=False)
    rows = []
    for order, index in enumerate(selected_indices, start=1):
        row = dict(eligible[int(index)])
        row["signature_id"] = f"{anchor_name}-N{order}"
        row["p0_role"] = f"{anchor_name}_neighbor"
        rows.append(row)
        excluded.add(row["signature_sha256"])
    return rows


def maximin_distance(
    candidates: np.ndarray,
    selected: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    distances = np.abs(
        (candidates[:, None, :] - selected[None, :, :]) / ranges[None, None, :]
    ).mean(axis=2)
    return distances.min(axis=1)


def select_maximin(
    archive: list[dict[str, str]],
    group_ids: list[str],
    groups: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    excluded = {row["signature_sha256"] for row in selected_rows}
    available = [row for row in archive if row["signature_sha256"] not in excluded]
    matrix = np.asarray(
        [[int(row[group_id]) for group_id in group_ids] for row in available],
        dtype=np.float64,
    )
    selected_matrix = np.asarray(
        [list(row["widths"]) for row in selected_rows], dtype=np.float64
    )
    ranges = np.asarray(
        [
            max(
                1,
                int(group["exact_maximum_width"])
                - int(group["exact_minimum_width"]),
            )
            for group in groups
        ],
        dtype=np.float64,
    )
    chosen: list[dict[str, Any]] = []
    active = np.ones(len(available), dtype=bool)
    for order in range(1, count + 1):
        active_indices = np.flatnonzero(active)
        distances = maximin_distance(
            matrix[active_indices], selected_matrix, ranges
        )
        ranked = sorted(
            range(len(active_indices)),
            key=lambda local: (
                -float(distances[local]),
                float(available[int(active_indices[local])]["fused_flops_g"]),
                int(available[int(active_indices[local])]["fused_params"]),
                available[int(active_indices[local])]["signature_id"],
            ),
        )
        local = ranked[0]
        global_index = int(active_indices[local])
        source = available[global_index]
        row = {
            "signature_id": f"MAXIMIN-{order:02d}",
            "p0_role": "maximin_diverse_archive",
            "signature_sha256": source["signature_sha256"],
            "widths": tuple(int(source[group_id]) for group_id in group_ids),
            "fused_params": int(source["fused_params"]),
            "fused_flops_g": float(source["fused_flops_g"]),
            "archive_signature_id": source["signature_id"],
            "minimum_normalized_manhattan_distance": float(distances[local]),
            "changed_gene_count": "",
            "changed_genes": "",
            "deltas": "",
        }
        chosen.append(row)
        selected_matrix = np.vstack((selected_matrix, matrix[global_index]))
        active[global_index] = False
    return chosen


def anchor_p0_row(
    signature_id: str,
    role: str,
    reference: dict[str, Any],
    group_ids: list[str],
) -> dict[str, Any]:
    return {
        "signature_id": signature_id,
        "p0_role": role,
        "signature_sha256": reference["signature_sha256"],
        "widths": tuple(int(reference["widths"][key]) for key in group_ids),
        "fused_params": int(reference["fused_params"]),
        "fused_flops_g": float(reference["fused_flops_g"]),
        "archive_signature_id": "",
        "minimum_normalized_manhattan_distance": "",
        "changed_gene_count": 0,
        "changed_genes": "",
        "deltas": "",
    }


def evaluation_id_payload(
    architecture_signature: str,
    dependency_hash: str,
    split_hash: str,
    training_protocol_hash: str,
) -> tuple[str, str]:
    fields = {
        "architecture_signature": architecture_signature,
        "raw_parent_checkpoint_hash": sha256(R1M.SEARCH_PARENT),
        "Taylor_ranking_hash": sha256(TAYLOR_RANKING),
        "dependency_manifest_hash": dependency_hash,
        "data_split_hash": split_hash,
        "training_protocol_hash": training_protocol_hash,
        "training_seed": str(TRAINING_SEED),
    }
    canonical = "\n".join(f"{key}={value}" for key, value in fields.items())
    return hash_text(canonical), canonical


def construct_p0(
    groups: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    archive: list[dict[str, str]],
    complexity_model,
    budget: dict[str, Any],
    dependency_hash: str,
    split_hash: str,
    training_protocol_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    group_ids = [row["group_id"] for row in groups]
    target = tuple(
        int(references["targetmanual_b6_reference"]["widths"][key])
        for key in group_ids
    )
    stage6 = tuple(
        int(references["stage6_allocation_anchor"]["widths"][key])
        for key in group_ids
    )
    target_neighbors = enumerate_neighbors(target, groups, complexity_model, budget)
    stage6_neighbors = enumerate_neighbors(stage6, groups, complexity_model, budget)
    membership: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    p0_hashes: dict[int, str] = {}
    p0_root = OUTPUT / "p0"
    for replicate, seeds in REPLICATES.items():
        rng = np.random.default_rng(np.random.SeedSequence(seeds["p0_seed"]))
        selected: list[dict[str, Any]] = []
        excluded: set[str] = set()

        target_row = anchor_p0_row(
            "TARGETMANUAL-EXACT",
            "exact_targetmanual",
            references["targetmanual_b6_reference"],
            group_ids,
        )
        selected.append(target_row)
        excluded.add(target_row["signature_sha256"])
        selected.extend(
            select_seeded_neighbors(
                "TARGETMANUAL",
                target_neighbors,
                3,
                rng,
                excluded,
            )
        )

        stage6_row = anchor_p0_row(
            "STAGE6-ALLOCATION-EXACT",
            "exact_stage6_allocation",
            references["stage6_allocation_anchor"],
            group_ids,
        )
        if stage6_row["signature_sha256"] in excluded:
            raise RuntimeError("TargetManual and Stage6 signatures unexpectedly match.")
        selected.append(stage6_row)
        excluded.add(stage6_row["signature_sha256"])
        selected.extend(
            select_seeded_neighbors(
                "STAGE6",
                stage6_neighbors,
                3,
                rng,
                excluded,
            )
        )
        selected.extend(
            select_maximin(archive, group_ids, groups, selected, count=8)
        )
        if len(selected) != 16 or len(
            {row["signature_sha256"] for row in selected}
        ) != 16:
            raise RuntimeError(f"Replicate {replicate} P0 is not 16 unique rows.")

        output_rows = []
        for slot, row in enumerate(selected, start=1):
            widths = tuple(row["widths"])
            signature = canonical_signature(group_ids, widths)
            eval_id, _ = evaluation_id_payload(
                signature,
                dependency_hash,
                split_hash,
                training_protocol_hash,
            )
            output_row = {
                "replicate_base": replicate,
                "p0_sub_seed": seeds["p0_seed"],
                "p0_slot": slot,
                "signature_id": row["signature_id"],
                "p0_role": row["p0_role"],
                "signature_sha256": row["signature_sha256"],
                "evaluation_id": eval_id,
                "fused_params": row["fused_params"],
                "fused_flops_g": row["fused_flops_g"],
                "changed_gene_count": row.get("changed_gene_count", ""),
                "changed_genes": row.get("changed_genes", ""),
                "deltas": row.get("deltas", ""),
                "minimum_normalized_manhattan_distance": row.get(
                    "minimum_normalized_manhattan_distance", ""
                ),
                "shared_methods": ";".join(METHODS),
                **dict(zip(group_ids, widths)),
            }
            output_rows.append(output_row)
            membership.append(output_row)
            all_rows.append({**row, "replicate": replicate, "evaluation_id": eval_id})
        csv_path = p0_root / f"replicate_{replicate}_p0.csv"
        csv_write(csv_path, output_rows)
        ordered = "\n".join(row["signature_sha256"] for row in output_rows)
        p0_hashes[replicate] = hash_text(ordered)
    csv_write(p0_root / "p0_membership_manifest.csv", membership)
    return membership, all_rows, p0_hashes


def write_unique_p0(
    all_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    dependency_hash: str,
    split_hash: str,
    training_protocol_hash: str,
) -> list[dict[str, Any]]:
    group_ids = [row["group_id"] for row in groups]
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_sha.setdefault(row["signature_sha256"], []).append(row)
    output = []
    for digest, rows in sorted(by_sha.items()):
        widths = tuple(rows[0]["widths"])
        signature = canonical_signature(group_ids, widths)
        evaluation_id, canonical = evaluation_id_payload(
            signature,
            dependency_hash,
            split_hash,
            training_protocol_hash,
        )
        output.append(
            {
                "unique_signature_index": len(output) + 1,
                "signature_sha256": digest,
                "evaluation_id": evaluation_id,
                "replicates": ";".join(
                    map(str, sorted({int(row["replicate"]) for row in rows}))
                ),
                "membership_count": len(rows),
                "roles": ";".join(sorted({row["p0_role"] for row in rows})),
                "fused_params": rows[0]["fused_params"],
                "fused_flops_g": rows[0]["fused_flops_g"],
                "training_seed": TRAINING_SEED,
                "evaluation_identity_canonical_sha256": hash_text(canonical),
                **dict(zip(group_ids, widths)),
            }
        )
    csv_write(OUTPUT / "p0" / "p0_unique_signatures.csv", output)
    return output


def write_rng_and_cache_schemas(
    p0_hashes: dict[int, str],
    dependency_hash: str,
    split_hash: str,
    training_protocol_hash: str,
) -> None:
    rng_payload = {
        "created_utc": utc_now(),
        "stream_mechanism": "numpy.random.SeedSequence with PCG64 default_rng",
        "replicates": {
            str(replicate): {
                "search_replicate_base": replicate,
                "p0_sub_seed": seeds["p0_seed"],
                "method_streams": {
                    "NSGA-II AP+APS": seeds["ap_aps"],
                    "Evolutionary AP-only": seeds["ap_only"],
                    "Warm-started constrained Random Search": seeds["random"],
                },
                "shared_ordered_p0_sha256": p0_hashes[replicate],
                "all_three_methods_use_same_p0": True,
            }
            for replicate, seeds in REPLICATES.items()
        },
        "status": "PASS",
    }
    json_write(OUTPUT / "rng_manifest.json", rng_payload)
    eval_schema = {
        "schema_name": "global_physical_evaluation_id_v1",
        "hash": "SHA256",
        "encoding": "UTF-8",
        "field_separator": "LF",
        "ordered_fields": [
            "architecture_signature",
            "raw_parent_checkpoint_hash",
            "Taylor_ranking_hash",
            "dependency_manifest_hash",
            "data_split_hash",
            "training_protocol_hash",
            "training_seed",
        ],
        "fixed_values": {
            "raw_parent_checkpoint_hash": sha256(R1M.SEARCH_PARENT),
            "Taylor_ranking_hash": sha256(TAYLOR_RANKING),
            "dependency_manifest_hash": dependency_hash,
            "data_split_hash": split_hash,
            "training_protocol_hash": training_protocol_hash,
            "training_seed": TRAINING_SEED,
        },
        "excluded_fields": [
            "method",
            "search replicate",
            "generation",
            "population position",
        ],
        "deduplication_rule": "Identical evaluation_id is trained/evaluated physically once.",
        "status": "PASS",
    }
    json_write(OUTPUT / "evaluation_id_schema.json", eval_schema)
    record_schema = {
        "schema_name": "search_record_id_v1",
        "hash": "SHA256",
        "encoding": "UTF-8",
        "field_separator": "LF",
        "ordered_fields": [
            "method",
            "replicate",
            "generation",
            "population_index",
            "evaluation_id",
        ],
        "purpose": "Tracks a logical search appearance without changing physical evaluation identity.",
        "status": "PASS",
    }
    json_write(OUTPUT / "search_record_id_schema.json", record_schema)


def smoke_test_p0(
    unique_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    taylor_bank: dict[str, dict[str, Any]],
    complexity_model,
    raw_tests: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    group_ids = [row["group_id"] for row in groups]
    cached = {
        raw_tests["target"]["signature_sha256"]: raw_tests["target"],
        raw_tests["stage6"]["signature_sha256"]: raw_tests["stage6"],
    }
    rows = []
    for unique in unique_rows:
        digest = unique["signature_sha256"]
        if digest in cached:
            test = cached[digest]
            reuse = "RAW_RECONSTRUCTION_REUSED"
        else:
            widths = tuple(int(unique[group_id]) for group_id in group_ids)
            test = construct_signature(
                f"P0_{unique['unique_signature_index']:03d}_{digest[:8]}",
                groups,
                widths,
                taylor_bank,
                complexity_model,
            )
            reuse = "FRESH_STRUCTURAL_SMOKE_TEST"
        rows.append(
            {
                "unique_signature_index": unique["unique_signature_index"],
                "signature_sha256": digest,
                "evaluation_id": unique["evaluation_id"],
                "test_mode": reuse,
                "fused_params_expected": unique["fused_params"],
                "fused_params_actual": test["fused_params_actual"],
                "params_match": int(unique["fused_params"])
                == int(test["fused_params_actual"]),
                "fused_flops_g_expected": unique["fused_flops_g"],
                "fused_flops_g_actual": test["fused_flops_g_actual"],
                "flops_match": math.isclose(
                    float(unique["fused_flops_g"]),
                    float(test["fused_flops_g_actual"]),
                    rel_tol=0.0,
                    abs_tol=R1M.FLOPS_TOLERANCE_G,
                ),
                "nc": test["nc"],
                "detect_scales": test["detect_scales"],
                "nonfinite_values": test["nonfinite_values"],
                "forward_pass": test["in_memory_forward"]
                and test["reloaded_forward"],
                "dual_budget_feasible": (
                    R1M.PARAMS_LOWER
                    <= int(unique["fused_params"])
                    <= R1M.PARAMS_UPPER
                    and R1M.FLOPS_LOWER_G
                    <= float(unique["fused_flops_g"])
                    <= R1M.FLOPS_UPPER_G
                ),
                "status": test["status"],
            }
        )
    if not all(
        row["status"] == "PASS"
        and row["params_match"]
        and row["flops_match"]
        and row["dual_budget_feasible"]
        for row in rows
    ):
        raise RuntimeError("One or more P0 structural smoke tests failed.")
    R1M.xlsx_write(
        OUTPUT / "structural_smoke_tests.xlsx",
        [("P0 Structural Smoke Tests", rows)],
    )
    return rows


def build_manifest() -> dict[str, Any]:
    files = [
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "stage03b_output_manifest.json"
    ]
    rows = [
        {
            "relative_path": str(path.relative_to(OUTPUT)),
            "absolute_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "scientific_role": (
                "raw checkpoint"
                if path.suffix == ".pt"
                else "Stage03B locked evidence or validation artifact"
            ),
        }
        for path in sorted(files)
    ]
    payload = {
        "created_utc": utc_now(),
        "output_root": str(OUTPUT),
        "file_count_excluding_manifest": len(rows),
        "files": rows,
    }
    json_write(OUTPUT / "stage03b_output_manifest.json", payload)
    return payload


def write_report_and_validation(
    source_unchanged: bool,
    references: dict[str, dict[str, Any]],
    raw_tests: dict[str, dict[str, Any]],
    membership: list[dict[str, Any]],
    unique_rows: list[dict[str, Any]],
    p0_hashes: dict[int, str],
    smoke_rows: list[dict[str, Any]],
    lock_payload: dict[str, Any],
) -> dict[str, Any]:
    p0_counts = {
        replicate: sum(int(row["replicate_base"]) == replicate for row in membership)
        for replicate in REPLICATES
    }
    p0_unique = {
        replicate: len(
            {
                row["signature_sha256"]
                for row in membership
                if int(row["replicate_base"]) == replicate
            }
        )
        for replicate in REPLICATES
    }
    stage01_state = json_load(STAGE01_STATE)
    checks = {
        "gene_space_lock": lock_payload["status"] == "PASS",
        "gene_group_count_35": lock_payload["dependency_group_count"] == 35,
        "targetmanual_raw_reconstruction": raw_tests["target"]["status"] == "PASS",
        "stage6_anchor_raw_reconstruction": raw_tests["stage6"]["status"] == "PASS",
        "stage6_exact_params": raw_tests["stage6"]["fused_params_actual"]
        == 2_515_679,
        "stage6_exact_flops": math.isclose(
            raw_tests["stage6"]["fused_flops_g_actual"],
            7.8540032,
            rel_tol=0.0,
            abs_tol=R1M.FLOPS_TOLERANCE_G,
        ),
        "p0_1042_16_valid_unique": p0_counts[1042] == 16
        and p0_unique[1042] == 16,
        "p0_2042_16_valid_unique": p0_counts[2042] == 16
        and p0_unique[2042] == 16,
        "p0_3042_16_valid_unique": p0_counts[3042] == 16
        and p0_unique[3042] == 16,
        "p0_method_memberships_48": len(membership) == 48,
        "shared_p0_method_mapping": len(p0_hashes) == 3
        and all(p0_hashes.values()),
        "global_evaluation_cache_schema": json_load(
            OUTPUT / "evaluation_id_schema.json"
        )["status"]
        == "PASS",
        "structural_smoke_tests": len(smoke_rows) == len(unique_rows)
        and all(row["status"] == "PASS" for row in smoke_rows),
        "official_val548_locked": (
            not bool(stage01_state.get("official_val_accessed", False))
            and not bool(
                stage01_state.get("official_val_used_for_evaluation", False)
            )
            and not bool(
                stage01_state.get("official_val_used_for_stratification", False)
            )
        ),
        "source_inputs_unchanged": source_unchanged,
        "candidate_training_not_executed": True,
        "candidate_evaluation_not_executed": True,
        "search_not_executed": True,
        "recovery_not_executed": True,
        "stage04_not_started": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    protocol_audit = {
        "created_utc": utc_now(),
        "approval_token": "APPROVE_STAGE03A_GENE_SPACE",
        "r1_validation_status": json_load(R1_VALIDATION)["status"],
        "r1_state_status": json_load(R1_STATE)["status"],
        "stage03a_state_before_stage03b": json_load(STAGE03A_STATE),
        "locked_source_hashes": source_snapshot(),
        "source_inputs_unchanged": source_unchanged,
        "official_val548_locked": checks["official_val548_locked"],
        "old_stage6_checkpoint_used_as_search_candidate": False,
        "stage6_search_candidate_parent": str(R1M.SEARCH_PARENT),
        "stage6_search_candidate_taylor": str(TAYLOR_RANKING),
        "training_executed": False,
        "evaluation_executed": False,
        "search_executed": False,
        "status": status,
    }
    json_write(OUTPUT / "protocol_lock_audit.json", protocol_audit)
    validation = {
        "created_utc": utc_now(),
        "stage": "Stage03B",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "p0_method_memberships": len(membership),
        "unique_p0_architecture_signatures": len(unique_rows),
        "unique_physical_evaluations_planned": len(unique_rows),
        "structural_smoke_tests": len(smoke_rows),
        "status": status,
    }
    json_write(OUTPUT / "stage03b_validation.json", validation)
    state = {
        "stage": "Stage03B",
        "status": "complete_waiting_for_stage03b_approval"
        if status == "PASS"
        else "failed",
        "timestamp_utc": utc_now(),
        "gene_space_lock": status,
        "targetmanual_raw_reconstruction": raw_tests["target"]["status"],
        "stage6_anchor_raw_reconstruction": raw_tests["stage6"]["status"],
        "p0_replicates": {
            str(replicate): f"{p0_counts[replicate]}_VALID_UNIQUE"
            if p0_unique[replicate] == 16
            else "FAIL"
            for replicate in REPLICATES
        },
        "shared_p0_method_mapping": (
            "PASS" if checks["shared_p0_method_mapping"] else "FAIL"
        ),
        "global_evaluation_cache_schema": (
            "PASS" if checks["global_evaluation_cache_schema"] else "FAIL"
        ),
        "structural_smoke_tests": (
            "PASS" if checks["structural_smoke_tests"] else "FAIL"
        ),
        "official_val548_locked": checks["official_val548_locked"],
        "candidate_training_executed": False,
        "candidate_evaluation_executed": False,
        "search_executed": False,
        "approval_ready": status == "PASS",
        "stage04_started": False,
        "required_next_approval_token": "APPROVE_STAGE03B_SEARCH_INITIALIZATION"
        if status == "PASS"
        else "NONE",
    }
    json_write(OUTPUT / "stage03b_state.json", state)
    report = f"""# Stage03B Gene-Space Lock and Shared-P0 Construction

## Status

`{status}`

## Locked Search Initialization

- Gene-space revision: R1.
- Dependency groups: 35.
- Feasible archive rows: 5,000.
- P0 method memberships: {len(membership)}.
- Unique P0 architecture signatures: {len(unique_rows)}.
- Unique physical evaluations planned: {len(unique_rows)}.
- Structural smoke tests: {len(smoke_rows)}.
- TargetManual raw Params/FLOPs: {raw_tests['target']['fused_params_actual']:,} / {raw_tests['target']['fused_flops_g_actual']:.7f} G.
- Stage6-Allocation raw Params/FLOPs: {raw_tests['stage6']['fused_params_actual']:,} / {raw_tests['stage6']['fused_flops_g_actual']:.7f} G.
- Old Stage-6 checkpoint used as search candidate: false.
- Official val548 accessed: false.
- Recovery/candidate training/search executed: false.
- Stage04 started: false.
"""
    text_write(OUTPUT / "stage03b_report.md", report)
    if status == "PASS":
        text_write(
            OUTPUT / ".complete",
            "STAGE03B_GENE_SPACE_LOCK_AND_SHARED_P0_CONSTRUCTION=PASS\n",
        )
    return validation


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if json_load(R1_VALIDATION)["status"] != "PASS":
        raise RuntimeError("R1 validation is not PASS.")
    if not json_load(R1_STATE)["approval_ready"]:
        raise RuntimeError("R1 is not approval-ready.")
    if sha256(R1M.SEARCH_PARENT) != R1M.SEARCH_PARENT_SHA:
        raise RuntimeError("SearchParent hash mismatch.")
    source_before = source_snapshot()

    approved_archive = OUTPUT / "approved_archive_5000.csv"
    shutil.copy2(FEASIBLE_ARCHIVE, approved_archive)
    if sha256(approved_archive) != sha256(FEASIBLE_ARCHIVE):
        raise RuntimeError("Approved archive copy hash mismatch.")
    text_write(
        OUTPUT / "approved_archive_5000.sha256",
        f"{sha256(approved_archive)}  {approved_archive.name}\n",
    )

    gene_payload = json_load(GENE_JSON)
    groups = gene_payload["groups"]
    if len(groups) != 35:
        raise RuntimeError("Locked gene-space does not contain 35 groups.")
    lock_payload = lock_inputs(gene_payload, approved_archive)
    source_artifact = R1M.load_source_model(R1M.SEARCH_PARENT, "stage03b_parent")
    complexity_model = R1M.ExactComplexityModel(source_artifact.model, groups)
    taylor_bank, taylor_manifest = R1M.taylor_scope_bank(groups)
    if taylor_manifest["checkpoint_sha256"] != R1M.SEARCH_PARENT_SHA:
        raise RuntimeError("Taylor ranking is not tied to the approved SearchParent.")
    target_payload = json_load(TARGETMANUAL)
    references = write_references(groups, complexity_model, target_payload)

    group_ids = [row["group_id"] for row in groups]
    target_widths = tuple(
        int(references["targetmanual_b6_reference"]["widths"][key])
        for key in group_ids
    )
    stage6_widths = tuple(
        int(references["stage6_allocation_anchor"]["widths"][key])
        for key in group_ids
    )
    raw_tests = {
        "target": construct_signature(
            "TargetManual-B6-reference raw",
            groups,
            target_widths,
            taylor_bank,
            complexity_model,
            OUTPUT / "raw_checkpoints" / "targetmanual_b6_raw.pt",
        ),
        "stage6": construct_signature(
            "Stage6-Allocation-Anchor raw",
            groups,
            stage6_widths,
            taylor_bank,
            complexity_model,
            OUTPUT / "raw_checkpoints" / "stage6_allocation_anchor_raw.pt",
        ),
    }
    write_channel_map(
        OUTPUT / "channel_maps" / "targetmanual_channel_map.csv",
        groups,
        target_widths,
        taylor_bank,
    )
    write_channel_map(
        OUTPUT / "channel_maps" / "stage6_anchor_channel_map.csv",
        groups,
        stage6_widths,
        taylor_bank,
    )

    archive = read_csv(approved_archive)
    if len(archive) != 5000 or len(
        {row["signature_sha256"] for row in archive}
    ) != 5000:
        raise RuntimeError("Approved archive is not 5000 unique signatures.")
    dependency_hash = sha256(DEPENDENCY_GRAPH)
    split_hash = sha256(SPLIT_HASH_SOURCE)
    training_protocol_hash = sha256(TRAINING_PROTOCOL_SOURCE)
    membership, all_rows, p0_hashes = construct_p0(
        groups,
        references,
        archive,
        complexity_model,
        gene_payload["budget"],
        dependency_hash,
        split_hash,
        training_protocol_hash,
    )
    unique_rows = write_unique_p0(
        all_rows,
        groups,
        dependency_hash,
        split_hash,
        training_protocol_hash,
    )
    write_rng_and_cache_schemas(
        p0_hashes,
        dependency_hash,
        split_hash,
        training_protocol_hash,
    )
    smoke_rows = smoke_test_p0(
        unique_rows,
        groups,
        taylor_bank,
        complexity_model,
        raw_tests,
    )
    scratch_root = OUTPUT / "_s"
    if scratch_root.exists():
        if any(scratch_root.iterdir()):
            raise RuntimeError("Stage03B scratch root is not empty after smoke tests.")
        scratch_root.rmdir()
    source_after = source_snapshot()
    source_unchanged = source_before == source_after
    if not source_unchanged:
        changed = [
            path
            for path, digest in source_before.items()
            if source_after.get(path) != digest
        ]
        raise RuntimeError(f"Locked source changed during Stage03B: {changed}")
    validation = write_report_and_validation(
        source_unchanged,
        references,
        raw_tests,
        membership,
        unique_rows,
        p0_hashes,
        smoke_rows,
        lock_payload,
    )
    build_manifest()
    if validation["status"] != "PASS":
        raise RuntimeError(f"Stage03B validation failed: {validation['failed_checks']}")

    print("STAGE03B GENE-SPACE LOCK AND SHARED-P0 CONSTRUCTION")
    print("GENE_SPACE_LOCK=PASS")
    print("TARGETMANUAL_RAW_RECONSTRUCTION=PASS")
    print("STAGE6_ANCHOR_RAW_RECONSTRUCTION=PASS")
    for replicate in REPLICATES:
        print(f"P0_REPLICATE_{replicate}=16_VALID_UNIQUE")
    print("SHARED_P0_METHOD_MAPPING=PASS")
    print("GLOBAL_EVALUATION_CACHE_SCHEMA=PASS")
    print("STRUCTURAL_SMOKE_TESTS=PASS")
    print("OFFICIAL_VAL548_LOCKED=YES")
    print("CANDIDATE_TRAINING_EXECUTED=NO")
    print("APPROVAL_READY=YES")
    print("STAGE04_STARTED=NO")
    print(f"P0_METHOD_MEMBERSHIPS={len(membership)}")
    print(f"UNIQUE_P0_ARCHITECTURES={len(unique_rows)}")
    print(f"OUTPUT_ROOT={OUTPUT}")


if __name__ == "__main__":
    main()
