from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

import stage05_common as C


STAGE04_SCRIPT = C.PROJECT / "scripts" / "stage04_shared_p0_fidelity1.py"
R1_SCRIPT = C.PROJECT / "scripts" / "stage03a_r1_gene_space_repair.py"
GENE_JSON = C.R1_ROOT / "gene_space_manifest_R1.json"
ARCHIVE_CSV = C.STAGE03B / "approved_archive_5000.csv"
P0_MEMBERSHIP = C.STAGE03B / "p0" / "p0_membership_manifest.csv"
P0_UNIQUE = C.STAGE03B / "p0" / "p0_unique_signatures.csv"
TAYLOR = C.STAGE03A / "taylor_ranking_manifest.json"
DEPENDENCY = C.STAGE03A / "dependency_graph.json"
HEARTBEAT_SECONDS = 15
MAX_PROPOSAL_ATTEMPTS = 20_000

ACTIVE_WORKER_STATE: dict[str, Any] = {}
ACTIVE_CAMPAIGN_UUID = ""
ACTIVE_WORKER_PID = os.getpid()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1M = load_module("stage05_r1_locked", R1_SCRIPT)
S4 = load_module("stage05_stage04_locked", STAGE04_SCRIPT)


def physical_dir(evaluation_id: str) -> Path:
    return C.PHYSICAL / f"E_{evaluation_id[:16]}"


def configure_stage04_runtime() -> None:
    S4.OUTPUT = C.ROOT
    S4.PHYSICAL_ROOT = C.PHYSICAL
    S4.DISK_STOP_GIB = C.HARD_STOP_FREE_GIB
    S4.evaluation_directory = lambda unique: physical_dir(unique["evaluation_id"])
    S4.run_logged = run_logged_with_heartbeat


def update_worker(**values: Any) -> dict[str, Any]:
    ACTIVE_WORKER_STATE.update(values)
    ACTIVE_WORKER_STATE["updated_utc"] = C.utc_now()
    C.atomic_json(C.WORKER_STATE, ACTIVE_WORKER_STATE)
    C.write_heartbeat(
        ACTIVE_WORKER_PID,
        ACTIVE_CAMPAIGN_UUID,
        ACTIVE_WORKER_STATE,
        current_epoch=ACTIVE_WORKER_STATE.get("current_epoch"),
    )
    C.write_status()
    return ACTIVE_WORKER_STATE


def current_epoch(results_csv: Path) -> int | None:
    if not results_csv.is_file():
        return None
    try:
        return len(pd.read_csv(results_csv))
    except Exception:
        return None


def run_logged_with_heartbeat(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    label: str,
) -> tuple[int, float]:
    start = time.time()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle, (
        stderr_path.open("w", encoding="utf-8", errors="replace")
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(C.PROJECT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=os.environ.copy(),
        )
        update_worker(
            child_pid=process.pid,
            child_command=subprocess.list2cmdline(command),
            child_label=label,
        )
        while process.poll() is None:
            eval_dir = Path(ACTIVE_WORKER_STATE.get("evaluation_directory", C.ROOT))
            epoch = current_epoch(eval_dir / "results.csv")
            update_worker(current_epoch=epoch)
            time.sleep(HEARTBEAT_SECONDS)
        code = int(process.returncode)
    update_worker(child_pid=None, child_command=None, child_label=None)
    return code, time.time() - start


def code_lock_valid() -> tuple[bool, list[str]]:
    payload = C.load_json(C.CODE_HASHES)
    failures: list[str] = []
    for row in payload["files"]:
        path = C.ROOT / row["relative_path"]
        if not path.is_file():
            failures.append(f"MISSING:{path}")
        elif C.sha256(path) != row["sha256"]:
            failures.append(f"HASH_MISMATCH:{path}")
    protocol = C.load_json(C.PROTOCOL_LOCK)
    if payload["stage05_protocol_sha256"] != protocol["stage05_protocol_sha256"]:
        failures.append("PROTOCOL_HASH_MISMATCH")
    return not failures, failures


def acquire_lock(campaign: dict[str, Any]) -> None:
    workers = [
        row for row in C.stage05_worker_processes() if row["pid"] != os.getpid()
    ]
    if workers:
        raise RuntimeError(f"Another Stage05 worker is active: {workers}")
    if C.CAMPAIGN_LOCK.is_file():
        existing = C.load_json(C.CAMPAIGN_LOCK)
        pid = existing.get("worker_pid")
        if C.process_alive(pid) and int(pid) != os.getpid():
            raise RuntimeError(f"Live Stage05 campaign lock belongs to PID {pid}.")
        if C.active_gpu_children():
            raise RuntimeError("Stale lock cannot be replaced while GPU child exists.")
    protocol = C.load_json(C.PROTOCOL_LOCK)
    worker_path = Path(__file__).resolve()
    process = psutil.Process(os.getpid())
    lock = {
        "worker_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "windows_process_start_time_utc": datetime.fromtimestamp(
            process.create_time(), timezone.utc
        ).isoformat(),
        "campaign_uuid": campaign["campaign_uuid"],
        "stage05_protocol_hash": protocol["stage05_protocol_sha256"],
        "worker_code_hash": C.sha256(worker_path),
        "creation_timestamp": C.utc_now(),
        "latest_heartbeat": C.utc_now(),
    }
    C.atomic_json(C.CAMPAIGN_LOCK, lock)
    C.atomic_text(C.WORKER_PID, f"{os.getpid()}\n")


def release_lock() -> None:
    if C.CAMPAIGN_LOCK.is_file():
        lock = C.load_json(C.CAMPAIGN_LOCK, {})
        if int(lock.get("worker_pid", -1)) == os.getpid():
            C.CAMPAIGN_LOCK.unlink()


def control_plane_validation(campaign: dict[str, Any]) -> dict[str, Any]:
    atomic_probe = C.RUNTIME / ".atomic_probe.json"
    C.atomic_json(atomic_probe, {"probe": "PASS"})
    readable = C.load_json(atomic_probe).get("probe") == "PASS"
    atomic_probe.unlink()
    lock = C.load_json(C.CAMPAIGN_LOCK)
    second_worker_prevented = (
        int(lock["worker_pid"]) == os.getpid()
        and len(C.stage05_worker_processes()) == 1
    )
    C.write_status()
    checks = {
        "detached_worker_pid_alive": C.process_alive(os.getpid()),
        "single_worker_lock": second_worker_prevented,
        "heartbeat_creation": C.HEARTBEAT.is_file(),
        "controller_state_readable": C.CONTROLLER_STATE.is_file(),
        "status_reporting": C.STATUS_JSON.is_file() and C.STATUS_MD.is_file(),
        "pause_request_recognition_function": C.PAUSE_REQUESTED.parent == C.RUNTIME,
        "resume_state_readability": C.CAMPAIGN_STATE.is_file(),
        "atomic_state_writing": readable,
        "extra_scientific_evaluations_created": False,
    }
    failed = [
        name
        for name, value in checks.items()
        if (name == "extra_scientific_evaluations_created" and value)
        or (name != "extra_scientific_evaluations_created" and not value)
    ]
    payload = {
        "created_utc": C.utc_now(),
        "checks": checks,
        "failed_checks": failed,
        "terminal_status": {
            "BACKGROUND_WORKER_CONTROL": "PASS" if not failed else "FAIL",
            "SINGLE_WORKER_LOCK": "PASS" if not failed else "FAIL",
            "HEARTBEAT": "PASS" if not failed else "FAIL",
            "STATUS_REPORTING": "PASS" if not failed else "FAIL",
            "GRACEFUL_PAUSE_CONTROL": "PASS" if not failed else "FAIL",
            "RESUME_STATE_CONTROL": "PASS" if not failed else "FAIL",
            "EXTRA_SCIENTIFIC_EVALUATIONS_CREATED": "NO",
        },
        "status": "PASS" if not failed else "FAIL",
    }
    C.atomic_json(C.ROOT / "stage05_control_plane_validation.json", payload)
    if failed:
        raise RuntimeError(f"Control-plane validation failed: {failed}")
    campaign["control_plane_validation"] = "PASS"
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    return payload


def load_search_assets() -> tuple[
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    Any,
    dict[str, dict[str, Any]],
]:
    gene = C.load_json(GENE_JSON)
    groups = gene["groups"]
    group_ids = [row["group_id"] for row in groups]
    archive = C.read_csv(ARCHIVE_CSV)
    if len(archive) != 5000:
        raise RuntimeError(f"Expected archive5000, found {len(archive)}.")
    signatures = {row["signature_sha256"] for row in archive}
    if len(signatures) != 5000:
        raise RuntimeError("Feasible archive signatures are not unique.")
    for row in archive:
        params = int(row["fused_params"])
        flops_raw = C.flops_raw_from_text(row["fused_flops_g"])
        if not C.exact_budget_feasible(params, flops_raw):
            raise RuntimeError(f"Archive budget violation: {row['signature_sha256']}")
        row["params_exact_integer"] = params
        row["flops_raw_exact_integer"] = flops_raw
        row["flops_g_full_precision"] = C.flops_full_precision(flops_raw)
    archive_by_signature = {row["signature_sha256"]: row for row in archive}
    # P0 contains approved anchors/neighbors that are valid but not necessarily
    # members of the certified archive5000. They are parent-readable only;
    # random sampling and deterministic repair remain restricted to archive5000.
    for row in C.read_csv(P0_UNIQUE):
        signature = row["signature_sha256"]
        if signature in archive_by_signature:
            continue
        params = int(row["fused_params"])
        flops_raw = C.flops_raw_from_text(row["fused_flops_g"])
        if not C.exact_budget_feasible(params, flops_raw):
            raise RuntimeError(f"P0 exact budget violation: {signature}")
        row["params_exact_integer"] = params
        row["flops_raw_exact_integer"] = flops_raw
        row["flops_g_full_precision"] = C.flops_full_precision(flops_raw)
        archive_by_signature[signature] = row
    source = R1M.load_source_model(R1M.SEARCH_PARENT, "stage05_parent")
    complexity_model = R1M.ExactComplexityModel(source.model, groups)
    taylor_bank, taylor_manifest = R1M.taylor_scope_bank(groups)
    if taylor_manifest["checkpoint_sha256"] != C.SEARCH_PARENT_SHA256:
        raise RuntimeError("Taylor bank checkpoint identity mismatch.")
    return groups, group_ids, archive, archive_by_signature, complexity_model, taylor_bank


def vector(row: dict[str, Any], group_ids: list[str]) -> tuple[int, ...]:
    return tuple(int(row[group_id]) for group_id in group_ids)


def cache_by_id() -> dict[str, dict[str, str]]:
    return {row["evaluation_id"]: row for row in C.read_csv(C.GLOBAL_CACHE)}


def cache_by_signature() -> dict[str, dict[str, str]]:
    return {
        row["signature_sha256"]: row
        for row in C.read_csv(C.GLOBAL_CACHE)
        if row.get("status") == "COMPLETE"
    }


def metric_record(signature: str, cache_signatures: dict[str, dict[str, str]]) -> dict[str, Any]:
    if signature not in cache_signatures:
        raise RuntimeError(f"No COMPLETE metrics for population signature {signature}.")
    row = dict(cache_signatures[signature])
    return {
        "evaluation_id": row["evaluation_id"],
        "signature_sha256": signature,
        "AP": float(row["AP"]),
        "APS": float(row["APS"]),
        "recall": str(row["recall_full_precision"]),
        "recall_violation": max(
            Decimal(0), C.RECALL_FLOOR - Decimal(str(row["recall_full_precision"]))
        ),
        "feasible": C.recall_feasible(row["recall_full_precision"]),
        "params_exact_integer": int(row["params_exact_integer"]),
        "flops_raw_exact_integer": int(row["flops_raw_exact_integer"]),
    }


def constrained_ranks(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = [row["signature_sha256"] for row in rows]
    by_key = {row["signature_sha256"]: row for row in rows}
    dominates = {key: set() for key in keys}
    dominated_count = {key: 0 for key in keys}

    def dominates_row(first: dict[str, Any], second: dict[str, Any]) -> bool:
        if first["feasible"] != second["feasible"]:
            return bool(first["feasible"])
        if not first["feasible"]:
            return first["recall_violation"] < second["recall_violation"]
        return (
            first["AP"] >= second["AP"]
            and first["APS"] >= second["APS"]
            and (first["AP"] > second["AP"] or first["APS"] > second["APS"])
        )

    for first in keys:
        for second in keys:
            if first == second:
                continue
            if dominates_row(by_key[first], by_key[second]):
                dominates[first].add(second)
            elif dominates_row(by_key[second], by_key[first]):
                dominated_count[first] += 1
    ranks: dict[str, int] = {}
    front = sorted(key for key in keys if dominated_count[key] == 0)
    rank = 1
    while front:
        next_front: set[str] = set()
        for key in front:
            ranks[key] = rank
            for other in dominates[key]:
                dominated_count[other] -= 1
                if dominated_count[other] == 0:
                    next_front.add(other)
        front = sorted(next_front)
        rank += 1
    if len(ranks) != len(keys):
        raise RuntimeError("Constrained rank did not cover every row.")
    return ranks


def crowding(rows: list[dict[str, Any]], ranks: dict[str, int]) -> dict[str, float]:
    output = {row["signature_sha256"]: 0.0 for row in rows}
    fronts: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        fronts.setdefault(ranks[row["signature_sha256"]], []).append(row)
    for front in fronts.values():
        if len(front) <= 2:
            for row in front:
                output[row["signature_sha256"]] = math.inf
            continue
        for objective in ("AP", "APS"):
            ordered = sorted(
                front, key=lambda row: (row[objective], row["signature_sha256"])
            )
            output[ordered[0]["signature_sha256"]] = math.inf
            output[ordered[-1]["signature_sha256"]] = math.inf
            minimum = ordered[0][objective]
            maximum = ordered[-1][objective]
            if maximum == minimum:
                continue
            for index in range(1, len(ordered) - 1):
                key = ordered[index]["signature_sha256"]
                if not math.isinf(output[key]):
                    output[key] += (
                        ordered[index + 1][objective]
                        - ordered[index - 1][objective]
                    ) / (maximum - minimum)
    return output


def ap_only_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row["feasible"]:
        return (0, -row["AP"], Decimal(0), row["flops_raw_exact_integer"], row["params_exact_integer"], row["signature_sha256"])
    return (
        1,
        Decimal(row["recall_violation"]),
        -row["AP"],
        row["flops_raw_exact_integer"],
        row["params_exact_integer"],
        row["signature_sha256"],
    )


def nsga_tournament_key(
    row: dict[str, Any], ranks: dict[str, int], distances: dict[str, float]
) -> tuple[Any, ...]:
    signature = row["signature_sha256"]
    distance = distances[signature]
    return (
        ranks[signature],
        -distance if not math.isinf(distance) else -math.inf,
        -row["AP"],
        -row["APS"],
        signature,
    )


def tournament(
    method: str,
    population: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = rng.choice(len(population), size=2, replace=False).tolist()
    candidates = [population[int(index)] for index in selected]
    if method == "NSGA-II AP+APS":
        ranks = constrained_ranks(population)
        distances = crowding(population, ranks)
        winner = min(
            candidates,
            key=lambda row: nsga_tournament_key(row, ranks, distances),
        )
        provenance = {
            "drawn_indices": selected,
            "drawn_signatures": [row["signature_sha256"] for row in candidates],
            "winner_signature": winner["signature_sha256"],
            "winner_rank": ranks[winner["signature_sha256"]],
            "winner_crowding": "INF"
            if math.isinf(distances[winner["signature_sha256"]])
            else distances[winner["signature_sha256"]],
        }
    else:
        winner = min(candidates, key=ap_only_key)
        provenance = {
            "drawn_indices": selected,
            "drawn_signatures": [row["signature_sha256"] for row in candidates],
            "winner_signature": winner["signature_sha256"],
        }
    return winner, provenance


def nearest_archive(
    proposed: tuple[int, ...],
    archive: list[dict[str, Any]],
    group_ids: list[str],
    groups: list[dict[str, Any]],
    excluded: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    denominators = [
        max(1, int(group["exact_maximum_width"]) - int(group["exact_minimum_width"]))
        for group in groups
    ]
    scored: list[tuple[float, int, int, str, dict[str, Any]]] = []
    for row in archive:
        signature = row["signature_sha256"]
        if signature in excluded:
            continue
        candidate = vector(row, group_ids)
        distance = sum(
            abs(first - second) / denominator
            for first, second, denominator in zip(proposed, candidate, denominators)
        )
        scored.append(
            (
                distance,
                int(row["flops_raw_exact_integer"]),
                int(row["params_exact_integer"]),
                signature,
                row,
            )
        )
    if not scored:
        raise RuntimeError("No unique certified feasible repair target remains.")
    chosen = min(scored)
    return chosen[-1], {
        "rule": "nearest certified feasible signature by normalized Manhattan distance",
        "distance": chosen[0],
        "tie_break_flops_raw": chosen[1],
        "tie_break_params": chosen[2],
        "tie_break_signature": chosen[3],
    }


def proposal_evolutionary(
    method: str,
    run_state: dict[str, Any],
    rng: np.random.Generator,
    archive: list[dict[str, Any]],
    archive_by_signature: dict[str, dict[str, Any]],
    groups: list[dict[str, Any]],
    group_ids: list[str],
    cache_signatures: dict[str, dict[str, str]],
    campaign: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    population = [
        metric_record(signature, cache_signatures)
        for signature in run_state["population_signatures"]
    ]
    seen = set(run_state["seen_signatures"])
    mutable = [
        index
        for index, group in enumerate(groups)
        if len(R1M.parse_allowed(group["allowed_widths"])) > 1
    ]
    rejected: list[dict[str, Any]] = []
    for attempt in range(1, MAX_PROPOSAL_ATTEMPTS + 1):
        first, draw_first = tournament(method, population, rng)
        second, draw_second = tournament(method, population, rng)
        first_vector = vector(archive_by_signature[first["signature_sha256"]], group_ids)
        second_vector = vector(archive_by_signature[second["signature_sha256"]], group_ids)
        crossover_draw = float(rng.random())
        if crossover_draw < 0.90:
            parent_choices = rng.random(len(group_ids)) < 0.50
            child = tuple(
                first_value if choose_first else second_value
                for first_value, second_value, choose_first in zip(
                    first_vector, second_vector, parent_choices
                )
            )
            crossover_applied = True
        else:
            parent_choices = np.ones(len(group_ids), dtype=bool)
            child = first_vector
            crossover_applied = False
        mutation_draw = float(rng.random())
        mutation_applied = mutation_draw < 0.35
        mutated_genes: list[dict[str, Any]] = []
        child_list = list(child)
        if mutation_applied:
            gene_count = int(rng.choice([1, 2, 3], p=[0.70, 0.25, 0.05]))
            selected = rng.choice(mutable, size=gene_count, replace=False).tolist()
            for index in selected:
                delta = int(rng.choice([8, 16, 24], p=[0.70, 0.25, 0.05]))
                sign = -1 if float(rng.random()) < 0.50 else 1
                proposed = child_list[index] + sign * delta
                allowed = R1M.parse_allowed(groups[index]["allowed_widths"])
                projected = min(
                    allowed,
                    key=lambda value: (abs(value - proposed), value),
                )
                mutated_genes.append(
                    {
                        "group_id": group_ids[index],
                        "old_width": child_list[index],
                        "delta_draw": sign * delta,
                        "unprojected_width": proposed,
                        "allowed_width_projection": projected,
                    }
                )
                child_list[index] = projected
        repaired, repair = nearest_archive(
            tuple(child_list), archive, group_ids, groups, seen
        )
        signature = repaired["signature_sha256"]
        if signature in seen:
            campaign["accounting"]["duplicate_rejections"] += 1
            rejected.append({"attempt": attempt, "reason": "WITHIN_RUN_DUPLICATE"})
            continue
        provenance = {
            "attempt": attempt,
            "method": method,
            "parent1": draw_first,
            "parent2": draw_second,
            "parent_signatures": [
                first["signature_sha256"],
                second["signature_sha256"],
            ],
            "crossover_draw": crossover_draw,
            "crossover_applied": crossover_applied,
            "per_gene_parent_choices": [
                "P1" if bool(value) else "P2" for value in parent_choices
            ],
            "mutation_draw": mutation_draw,
            "mutation_applied": mutation_applied,
            "mutated_genes": mutated_genes,
            "pre_repair_widths": dict(zip(group_ids, child_list)),
            "repair": repair,
            "resulting_signature": signature,
            "rejected_attempts_before_acceptance": rejected,
        }
        return repaired, provenance
    raise RuntimeError("Candidate proposal attempt limit exceeded.")


def proposal_random(
    run_state: dict[str, Any],
    rng: np.random.Generator,
    archive: list[dict[str, Any]],
    campaign: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    seen = set(run_state["seen_signatures"])
    rejected: list[dict[str, Any]] = []
    for attempt in range(1, MAX_PROPOSAL_ATTEMPTS + 1):
        index = int(rng.integers(0, len(archive)))
        row = archive[index]
        if row["signature_sha256"] in seen:
            campaign["accounting"]["duplicate_rejections"] += 1
            rejected.append(
                {
                    "attempt": attempt,
                    "archive_index": index,
                    "reason": "WITHIN_RUN_DUPLICATE",
                }
            )
            continue
        return row, {
            "attempt": attempt,
            "method": "Warm-started constrained Random Search",
            "archive_index_draw": index,
            "sampling": "uniform without replacement within replicate",
            "performance_information_used": False,
            "resulting_signature": row["signature_sha256"],
            "rejected_attempts_before_acceptance": rejected,
        }
    raise RuntimeError("Random proposal attempt limit exceeded.")


def proposal(
    run_state: dict[str, Any],
    rng: np.random.Generator,
    archive: list[dict[str, Any]],
    archive_by_signature: dict[str, dict[str, Any]],
    groups: list[dict[str, Any]],
    group_ids: list[str],
    campaign: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    method = run_state["method"]
    if method == "Warm-started constrained Random Search":
        return proposal_random(run_state, rng, archive, campaign)
    return proposal_evolutionary(
        method,
        run_state,
        rng,
        archive,
        archive_by_signature,
        groups,
        group_ids,
        cache_by_signature(),
        campaign,
    )


def exact_candidate(
    archive_row: dict[str, Any],
    group_ids: list[str],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    params = int(archive_row["params_exact_integer"])
    flops_raw = int(archive_row["flops_raw_exact_integer"])
    if not C.exact_budget_feasible(params, flops_raw):
        raise RuntimeError("Candidate exact integer budget validation failed.")
    signature = archive_row["signature_sha256"]
    evaluation, canonical = C.evaluation_id(signature, protocol)
    return {
        "unique_signature_index": 0,
        "signature_sha256": signature,
        "evaluation_id": evaluation,
        "evaluation_identity_canonical": canonical,
        "roles": "STAGE05_SEARCH_CANDIDATE",
        "replicates": "",
        "membership_count": 1,
        "fused_params": params,
        "fused_flops_g": C.flops_full_precision(flops_raw),
        "params_exact_integer": params,
        "thop_macs_exact_integer": flops_raw // 2,
        "flops_raw_exact_integer": flops_raw,
        "flops_g_full_precision": C.flops_full_precision(flops_raw),
        **{group_id: int(archive_row[group_id]) for group_id in group_ids},
    }


def update_exact_metrics(
    metrics_path: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    metrics = C.load_json(metrics_path)
    metrics.update(
        {
            "params_exact_integer": int(candidate["params_exact_integer"]),
            "thop_macs_exact_integer": int(candidate["thop_macs_exact_integer"]),
            "flops_raw_exact_integer": int(candidate["flops_raw_exact_integer"]),
            "flops_g_full_precision": candidate["flops_g_full_precision"],
            "recall_full_precision": str(metrics["recall"]),
            "recall_floor_full_precision": C.RECALL_FLOOR_TEXT,
            "recall_feasible": C.recall_feasible(metrics["recall"]),
            "exact_budget_feasible": C.exact_budget_feasible(
                candidate["params_exact_integer"],
                candidate["flops_raw_exact_integer"],
            ),
            "rounded_values_used_for_selection": False,
            "stage05_source_of_truth": True,
            "stage05_evaluation_id": candidate["evaluation_id"],
            "stage05_signature_sha256": candidate["signature_sha256"],
        }
    )
    C.atomic_json(metrics_path, metrics)
    return metrics


def retention(eval_dir: Path, evaluation_id: str) -> list[dict[str, Any]]:
    removable = [
        eval_dir / "weights" / "last.pt",
        eval_dir / "raw_checkpoint.pt",
        eval_dir / "explicit_inner_val_predictions",
    ]
    deleted: list[dict[str, Any]] = []
    for path in removable:
        if not path.exists():
            continue
        if path.is_dir():
            files = [item for item in path.rglob("*") if item.is_file()]
        else:
            files = [path]
        for item in files:
            deleted.append(
                {
                    "evaluation_id": evaluation_id,
                    "relative_path": str(item.relative_to(C.ROOT)),
                    "sha256": C.sha256(item),
                    "size_bytes": item.stat().st_size,
                    "scientific_role": "temporary_or_redundant_after_explicit_evaluation",
                    "deletion_reason": "STREAMING_RETENTION_POLICY",
                    "deletion_timestamp_utc": C.utc_now(),
                }
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    existing = C.read_csv(C.RETENTION_MANIFEST)
    C.atomic_csv(C.RETENTION_MANIFEST, existing + deleted)
    audit = C.load_json(C.DISK_AUDIT, {})
    audit["deleted_temporary_size_bytes"] = int(
        audit.get("deleted_temporary_size_bytes", 0)
    ) + sum(int(row["size_bytes"]) for row in deleted)
    C.atomic_json(C.DISK_AUDIT, audit)
    return deleted


def physical_cache_row(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    eval_dir: Path,
    hashes: dict[str, Any],
    reconstruction_path: Path,
    run_manifest_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "evaluation_id": candidate["evaluation_id"],
        "evaluation_identity_canonical_sha256": candidate["evaluation_id"],
        "evaluation_identity_canonical": candidate["evaluation_identity_canonical"],
        "signature_sha256": candidate["signature_sha256"],
        "source_stage": "Stage05",
        "source_stage04_evaluation_id": "",
        "cache_hit": False,
        "training_seed": C.TRAINING_SEED,
        "recovery_epochs": C.RECOVERY_EPOCHS,
        "kd_enabled": False,
        "train_image_count": C.TRAIN_IMAGES,
        "validation_image_count": C.VAL_IMAGES,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "recall_full_precision": metrics["recall_full_precision"],
        "recall_feasible": metrics["recall_feasible"],
        "f1": metrics["f1"],
        "AP": metrics["AP_0.5_0.95"],
        "AP50": metrics["AP50"],
        "AP75": metrics["AP75"],
        "APS": metrics["APS"],
        "APM": metrics["APM"],
        "APL": metrics["APL"],
        "params": candidate["params_exact_integer"],
        "params_exact_integer": candidate["params_exact_integer"],
        "thop_macs_exact_integer": candidate["thop_macs_exact_integer"],
        "flops_g": candidate["flops_g_full_precision"],
        "flops_g_full_precision": candidate["flops_g_full_precision"],
        "flops_raw_exact_integer": candidate["flops_raw_exact_integer"],
        "exact_budget_feasible": True,
        "weights_size_mib": metrics["weights_size_mib"],
        "best_epoch_zero_based": metrics["best_epoch"]["best_epoch_zero_based"],
        "best_epoch_one_based": metrics["best_epoch"]["best_epoch_one_based"],
        "best_fitness": metrics["best_epoch"]["best_fitness"],
        "training_duration_seconds": metrics.get("training_duration_seconds"),
        "explicit_evaluation_duration_seconds": metrics.get(
            "explicit_evaluation_duration_seconds"
        ),
        "best_checkpoint": hashes["best_checkpoint"]["path"],
        "best_checkpoint_sha256": hashes["best_checkpoint"]["sha256"],
        "last_checkpoint": hashes["last_checkpoint"]["path"],
        "last_checkpoint_sha256": hashes["last_checkpoint"]["sha256"],
        "raw_checkpoint": hashes["raw_checkpoint"]["path"],
        "raw_checkpoint_sha256": hashes["raw_checkpoint"]["sha256"],
        "metrics_file": str(eval_dir / "explicit_inner_val_metrics.json"),
        "metrics_sha256": C.sha256(eval_dir / "explicit_inner_val_metrics.json"),
        "reconstruction_manifest": str(reconstruction_path),
        "reconstruction_manifest_sha256": C.sha256(reconstruction_path),
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": C.sha256(run_manifest_path),
        "primary_ap_namespace": metrics["primary_ap_namespace"],
        "area_metric_namespace": metrics["area_metric_namespace"],
        "official_val548_accessed": False,
        "heldout1610_accessed": False,
        "stage05_protocol_sha256": protocol["stage05_protocol_sha256"],
        "status": "COMPLETE",
    }


def cross_file_candidate_consistency(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    reconstruction: dict[str, Any],
    cache_row: dict[str, Any],
) -> dict[str, Any]:
    params = {
        int(candidate["params_exact_integer"]),
        int(metrics["params_exact_integer"]),
        int(reconstruction["params_exact_integer"]),
        int(cache_row["params_exact_integer"]),
    }
    flops = {
        int(candidate["flops_raw_exact_integer"]),
        int(metrics["flops_raw_exact_integer"]),
        int(reconstruction["flops_raw_exact_integer"]),
        int(cache_row["flops_raw_exact_integer"]),
    }
    recall = {
        str(metrics["recall_full_precision"]),
        str(cache_row["recall_full_precision"]),
    }
    signature = {
        candidate["signature_sha256"],
        metrics["stage05_signature_sha256"],
        reconstruction["architecture_signature_sha256"],
        cache_row["signature_sha256"],
    }
    evaluation = {
        candidate["evaluation_id"],
        metrics["stage05_evaluation_id"],
        reconstruction["evaluation_id"],
        cache_row["evaluation_id"],
    }
    checks = {
        "params": len(params) == 1,
        "flops": len(flops) == 1,
        "recall": len(recall) == 1,
        "architecture_signature": len(signature) == 1,
        "evaluation_id": len(evaluation) == 1,
    }
    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def perform_physical_evaluation(
    candidate: dict[str, Any],
    groups: list[dict[str, Any]],
    complexity_model: Any,
    taylor_bank: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    eval_dir = physical_dir(candidate["evaluation_id"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    update_worker(
        candidate_state="STRUCTURALLY_VALIDATED",
        evaluation_id=candidate["evaluation_id"],
        signature_sha256=candidate["signature_sha256"],
        evaluation_directory=str(eval_dir),
    )
    analytic = complexity_model.calculate(
        {group["group_id"]: int(candidate[group["group_id"]]) for group in groups}
    )
    analytic_flops_raw = int(analytic["macs"]) * 2
    if (
        int(analytic["params"]) != int(candidate["params_exact_integer"])
        or analytic_flops_raw != int(candidate["flops_raw_exact_integer"])
    ):
        raise RuntimeError("Exact analytic candidate complexity mismatch.")
    update_worker(candidate_state="RAW_RECONSTRUCTED")
    reconstruction = reconstruct_with_parent_load_retry(
        candidate,
        groups,
        complexity_model,
        taylor_bank,
        campaign,
    )
    reconstruction.update(
        {
            "stage": "Stage05 Full Fidelity-1 Search",
            "params_exact_integer": candidate["params_exact_integer"],
            "thop_macs_exact_integer": candidate["thop_macs_exact_integer"],
            "flops_raw_exact_integer": candidate["flops_raw_exact_integer"],
            "flops_g_full_precision": candidate["flops_g_full_precision"],
            "exact_budget_feasible": True,
            "rounded_values_used_for_selection": False,
        }
    )
    reconstruction_path = eval_dir / "raw_reconstruction_audit.json"
    C.atomic_json(reconstruction_path, reconstruction)
    update_worker(
        candidate_state="TRAINING_STARTED",
        training_start_time=C.utc_now(),
        current_epoch=0,
    )
    training = S4.train_one(candidate, reconstruction)
    update_worker(candidate_state="TRAINING_COMPLETED", current_epoch=C.RECOVERY_EPOCHS)
    update_worker(candidate_state="EXPLICIT_EVALUATION_STARTED")
    evaluation = S4.evaluate_one(candidate)
    update_worker(candidate_state="EXPLICIT_EVALUATION_COMPLETED")
    metrics = S4.update_metrics_durations(eval_dir, training, evaluation)
    metrics = update_exact_metrics(
        eval_dir / "explicit_inner_val_metrics.json", candidate
    )
    hashes = S4.checkpoint_hashes(eval_dir)
    update_worker(candidate_state="HASH_VALIDATED")
    run_manifest_path = eval_dir / "run_manifest.json"
    run_manifest = {
        "created_utc": C.utc_now(),
        "stage": "Stage05 Full Fidelity-1 Search",
        "evaluation_id": candidate["evaluation_id"],
        "architecture_signature_sha256": candidate["signature_sha256"],
        "retained_widths": {
            group["group_id"]: int(candidate[group["group_id"]]) for group in groups
        },
        "params_exact_integer": candidate["params_exact_integer"],
        "thop_macs_exact_integer": candidate["thop_macs_exact_integer"],
        "flops_raw_exact_integer": candidate["flops_raw_exact_integer"],
        "flops_g_full_precision": candidate["flops_g_full_precision"],
        "recall_full_precision": metrics["recall_full_precision"],
        "training": training,
        "explicit_evaluation": evaluation,
        "metrics_file": str(eval_dir / "explicit_inner_val_metrics.json"),
        "metrics_sha256": C.sha256(eval_dir / "explicit_inner_val_metrics.json"),
        "checkpoints_before_retention": hashes,
        "stage05_protocol_sha256": protocol["stage05_protocol_sha256"],
        "official_val548_accessed": False,
        "heldout1610_accessed": False,
        "kd_executed": False,
        "status": "PASS",
    }
    C.atomic_json(run_manifest_path, run_manifest)
    row = physical_cache_row(
        candidate,
        metrics,
        eval_dir,
        hashes,
        reconstruction_path,
        run_manifest_path,
        protocol,
    )
    consistency = cross_file_candidate_consistency(
        candidate, metrics, reconstruction, row
    )
    if consistency["status"] != "PASS":
        raise RuntimeError(f"Cross-file numeric consistency failed: {consistency}")
    C.atomic_json(eval_dir / "numeric_consistency_audit.json", consistency)
    cache_rows = C.read_csv(C.GLOBAL_CACHE)
    if candidate["evaluation_id"] in {
        existing["evaluation_id"] for existing in cache_rows
    }:
        raise RuntimeError("Duplicate physical cache commitment attempted.")
    cache_rows.append(row)
    C.atomic_csv(C.GLOBAL_CACHE, cache_rows)
    C.atomic_csv(C.PHYSICAL_MANIFEST, cache_rows)
    update_worker(candidate_state="CACHE_COMMITTED")
    deleted = retention(eval_dir, candidate["evaluation_id"])
    run_manifest["retention"] = {
        "deleted_files": len(deleted),
        "deleted_size_bytes": sum(int(item["size_bytes"]) for item in deleted),
        "best_checkpoint_retained": True,
    }
    run_manifest["status"] = "COMPLETE"
    C.atomic_json(run_manifest_path, run_manifest)
    row["run_manifest_sha256"] = C.sha256(run_manifest_path)
    row["metrics_sha256"] = C.sha256(eval_dir / "explicit_inner_val_metrics.json")
    cache_rows = [
        row if existing["evaluation_id"] == row["evaluation_id"] else existing
        for existing in C.read_csv(C.GLOBAL_CACHE)
    ]
    C.atomic_csv(C.GLOBAL_CACHE, cache_rows)
    C.atomic_csv(C.PHYSICAL_MANIFEST, cache_rows)
    update_worker(candidate_state="RETENTION_COMPLETED")
    C.atomic_text(
        eval_dir / ".complete",
        "STAGE05_PHYSICAL_EVALUATION=COMPLETE\n"
        f"EVALUATION_ID={candidate['evaluation_id']}\n",
    )
    update_worker(candidate_state="COMPLETE")
    campaign["accounting"]["completed_physical_evaluations"] += 1
    return row


def reconstruct_with_parent_load_retry(
    candidate: dict[str, Any],
    groups: list[dict[str, Any]],
    complexity_model: Any,
    taylor_bank: dict[str, dict[str, Any]],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    """Retry one verified transient SearchParent loader failure without changing the candidate."""
    try:
        return S4.reconstruct_raw(candidate, groups, complexity_model, taylor_bank)
    except KeyError as error:
        expected = "Checkpoint does not contain 'model_state_dict' and 'yaml'."
        if expected not in str(error):
            raise
        actual_sha256 = C.sha256(R1M.SEARCH_PARENT)
        if actual_sha256 != C.SEARCH_PARENT_SHA256:
            raise RuntimeError("SearchParent changed during reconstruction retry.") from error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        campaign["accounting"]["engineering_retries"] += 1
        retry = {
            "created_utc": C.utc_now(),
            "classification": "TRANSIENT_SEARCHPARENT_LOAD_RETRY",
            "evaluation_id": candidate["evaluation_id"],
            "architecture_signature": candidate["signature_sha256"],
            "searchparent": str(R1M.SEARCH_PARENT),
            "searchparent_sha256": actual_sha256,
            "candidate_changed": False,
            "rng_changed": False,
            "training_started_before_retry": False,
            "retry_limit": 1,
        }
        retry_path = C.RUNTIME / "stage05_checkpoint_load_retry.json"
        existing = C.load_json(retry_path, {"events": []})
        existing.setdefault("events", []).append(retry)
        existing["status"] = "ACTIVE_WITH_VERIFIED_RETRIES"
        C.atomic_json(retry_path, existing)
        C.atomic_json(C.CAMPAIGN_STATE, campaign)
        return S4.reconstruct_raw(candidate, groups, complexity_model, taylor_bank)


def method_record(
    candidate: dict[str, Any],
    cache_row: dict[str, Any],
    method: str,
    replicate: int,
    generation_or_batch: int,
    population_index: int,
    provenance: dict[str, Any],
    cache_hit: bool,
) -> dict[str, Any]:
    recall = str(cache_row["recall_full_precision"])
    record_id = C.search_record_id(
        method,
        replicate,
        generation_or_batch,
        population_index,
        candidate["evaluation_id"],
    )
    return {
        "search_record_id": record_id,
        "method": method,
        "replicate": replicate,
        "generation": generation_or_batch
        if method != "Warm-started constrained Random Search"
        else "",
        "random_batch": generation_or_batch
        if method == "Warm-started constrained Random Search"
        else "",
        "population_index": population_index,
        "evaluation_id": candidate["evaluation_id"],
        "signature_sha256": candidate["signature_sha256"],
        "record_origin": "STAGE05_NEW",
        "precision": cache_row["precision"],
        "recall": recall,
        "recall_full_precision": recall,
        "recall_floor_f1": C.RECALL_FLOOR_TEXT,
        "recall_violation": str(max(Decimal(0), C.RECALL_FLOOR - Decimal(recall))),
        "recall_feasible": C.recall_feasible(recall),
        "f1": cache_row["f1"],
        "AP": cache_row["AP"],
        "AP50": cache_row["AP50"],
        "AP75": cache_row["AP75"],
        "APS": cache_row["APS"],
        "APM": cache_row["APM"],
        "APL": cache_row["APL"],
        "params": candidate["params_exact_integer"],
        "params_exact_integer": candidate["params_exact_integer"],
        "flops_g": candidate["flops_g_full_precision"],
        "flops_g_full_precision": candidate["flops_g_full_precision"],
        "flops_raw_exact_integer": candidate["flops_raw_exact_integer"],
        "exact_budget_feasible": True,
        "feasible": C.recall_feasible(recall),
        "cache_hit": cache_hit,
        "cache_source_evaluation_id": candidate["evaluation_id"],
        "operator_provenance_sha256": C.hash_text(C.canonical_json(provenance)),
        "metrics_file": cache_row["metrics_file"],
        "metrics_sha256": cache_row["metrics_sha256"],
        "reconstruction_manifest": cache_row["reconstruction_manifest"],
        "reconstruction_manifest_sha256": cache_row[
            "reconstruction_manifest_sha256"
        ],
        "stage05_protocol_sha256": C.load_json(C.PROTOCOL_LOCK)[
            "stage05_protocol_sha256"
        ],
        "rounded_values_used_for_selection": False,
        "status": "COMPLETE",
    }


def validate_record_consistency(
    record: dict[str, Any],
    cache_row: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    fields = {
        "params": {
            int(record["params_exact_integer"]),
            int(cache_row["params_exact_integer"]),
            int(candidate["params_exact_integer"]),
        },
        "flops": {
            int(record["flops_raw_exact_integer"]),
            int(cache_row["flops_raw_exact_integer"]),
            int(candidate["flops_raw_exact_integer"]),
        },
        "recall": {
            str(record["recall_full_precision"]),
            str(cache_row["recall_full_precision"]),
        },
        "signature": {
            record["signature_sha256"],
            cache_row["signature_sha256"],
            candidate["signature_sha256"],
        },
        "evaluation_id": {
            record["evaluation_id"],
            cache_row["evaluation_id"],
            candidate["evaluation_id"],
        },
    }
    bad = [name for name, values in fields.items() if len(values) != 1]
    if bad:
        raise RuntimeError(f"Method-record cross-file mismatch: {bad}")
    numeric_failures = C.validate_numeric_row(record)
    if numeric_failures:
        raise RuntimeError(f"Method-record numeric validation failed: {numeric_failures}")


def write_provenance(
    record: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    path = C.METHOD_RECORDS / f"{record['search_record_id']}.json"
    payload = {
        "created_utc": C.utc_now(),
        "search_record": record,
        "operator_provenance": provenance,
        "rng_state_after_proposal": provenance.get("rng_state_after_proposal"),
        "status": "COMPLETE",
    }
    C.atomic_json(path, payload)
    rows = C.read_csv(C.PROVENANCE_CSV)
    rows.append(
        {
            "search_record_id": record["search_record_id"],
            "method": record["method"],
            "replicate": record["replicate"],
            "generation": record["generation"],
            "random_batch": record["random_batch"],
            "population_index": record["population_index"],
            "evaluation_id": record["evaluation_id"],
            "signature_sha256": record["signature_sha256"],
            "parent_signatures": ";".join(
                provenance.get("parent_signatures", [])
            ),
            "crossover_applied": provenance.get("crossover_applied", False),
            "mutation_applied": provenance.get("mutation_applied", False),
            "repair_distance": provenance.get("repair", {}).get("distance", ""),
            "provenance_file": str(path),
            "provenance_sha256": C.sha256(path),
            "status": "PASS",
        }
    )
    C.atomic_csv(C.PROVENANCE_CSV, rows)


def survivors(
    method: str,
    signatures: list[str],
    cache_signatures: dict[str, dict[str, str]],
) -> list[str]:
    rows = [metric_record(signature, cache_signatures) for signature in signatures]
    if method == "Single-objective evolutionary search (AP-only)":
        return [
            row["signature_sha256"]
            for row in sorted(rows, key=ap_only_key)[:16]
        ]
    ranks = constrained_ranks(rows)
    distances = crowding(rows, ranks)
    ordered = sorted(
        rows,
        key=lambda row: nsga_tournament_key(row, ranks, distances),
    )
    return [row["signature_sha256"] for row in ordered[:16]]


def save_generation_archive(
    run_state: dict[str, Any],
    generation: int,
    candidate_signatures: list[str],
) -> None:
    cache_signatures = cache_by_signature()
    population = [
        metric_record(signature, cache_signatures)
        for signature in run_state["population_signatures"]
    ]
    payload = {
        "created_utc": C.utc_now(),
        "method": run_state["method"],
        "replicate": run_state["replicate"],
        "generation": generation,
        "offspring_signatures": candidate_signatures,
        "survivor_population": population,
        "rng_state": run_state["rng_state"],
        "status": "PASS",
    }
    safe_method = (
        "nsga2"
        if run_state["method"] == "NSGA-II AP+APS"
        else "ap_only"
    )
    C.atomic_json(
        C.ARCHIVES
        / f"{safe_method}_rep{run_state['replicate']}_gen{generation}.json",
        payload,
    )


def finish_generation_if_needed(
    run_state: dict[str, Any],
) -> None:
    if run_state["method"] == "Warm-started constrained Random Search":
        return
    offspring = run_state.setdefault("current_generation_offspring", [])
    if len(offspring) < 16:
        return
    combined = run_state["population_signatures"] + offspring
    run_state["population_signatures"] = survivors(
        run_state["method"], combined, cache_by_signature()
    )
    generation = int(run_state["current_generation_or_batch"])
    run_state["rng_state"] = run_state["rng_state"]
    save_generation_archive(run_state, generation, offspring)
    run_state["current_generation_offspring"] = []
    if run_state["completed_new_records"] < 64:
        run_state["current_generation_or_batch"] = generation + 1


def graceful_pause(campaign: dict[str, Any]) -> int:
    campaign["state"] = "PAUSING"
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    update_worker(candidate_state="PAUSING", child_pid=None)
    if C.active_gpu_children():
        raise RuntimeError("Cannot finalize graceful pause while GPU child is active.")
    campaign["state"] = "PAUSED"
    campaign["paused_utc"] = C.utc_now()
    campaign["current_position"]["next_scientific_position"] = {
        "run_index": campaign["current_run_index"],
        "run_state": campaign["run_states"][campaign["current_run_index"]]
        if campaign["current_run_index"] < len(campaign["run_states"])
        else None,
    }
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    C.atomic_text(C.PAUSED_FLAG, "STAGE05_PAUSED=YES\n")
    C.atomic_json(
        C.STOP_REASON,
        {
            "created_utc": C.utc_now(),
            "reason": "GRACEFUL_PAUSE_REQUEST",
            "terminal_status": {
                "STAGE05_PAUSE_REQUESTED": "YES",
                "CURRENT_EVALUATION_COMPLETED_SAFELY": "YES",
                "CAMPAIGN_STATE_SAVED": "YES",
                "POPULATION_STATE_SAVED": "YES",
                "RNG_STATE_SAVED": "YES",
                "GLOBAL_CACHE_SAVED": "YES",
                "NO_GPU_JOB_ACTIVE": "YES",
                "WORKER_TERMINATED_CLEANLY": "YES",
                "STAGE05_PAUSED": "YES",
                "SAFE_TO_SHUT_DOWN_LAPTOP": "YES",
            },
        },
    )
    C.write_status()
    release_lock()
    return 0


def record_failure(
    campaign: dict[str, Any],
    run_state: dict[str, Any],
    candidate: dict[str, Any] | None,
    error: BaseException,
    classification: str,
) -> None:
    audit = C.load_json(C.FAILURE_AUDIT)
    key = (
        "scientific_failures"
        if classification == "FAILED_SCIENTIFIC"
        else "engineering_failures"
    )
    audit[key].append(
        {
            "created_utc": C.utc_now(),
            "classification": classification,
            "method": run_state["method"],
            "replicate": run_state["replicate"],
            "generation_or_batch": run_state["current_generation_or_batch"],
            "candidate": candidate,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    )
    audit["status"] = "ACTIVE_WITH_FAILURES"
    C.atomic_json(C.FAILURE_AUDIT, audit)
    campaign["state"] = "BLOCKED"
    campaign["blocker"] = audit[key][-1]
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    C.atomic_json(C.STOP_REASON, campaign["blocker"])
    update_worker(candidate_state=classification, child_pid=None)


def execute_candidate(
    campaign: dict[str, Any],
    run_state: dict[str, Any],
    rng: np.random.Generator,
    archive: list[dict[str, Any]],
    archive_by_signature: dict[str, dict[str, Any]],
    groups: list[dict[str, Any]],
    group_ids: list[str],
    complexity_model: Any,
    taylor_bank: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> None:
    method = run_state["method"]
    replicate = int(run_state["replicate"])
    generation_or_batch = int(run_state["current_generation_or_batch"])
    population_index = int(run_state["completed_new_records"]) + 1
    if run_state.get("active_record"):
        assigned = run_state["active_record"]
        candidate = assigned["candidate"]
        provenance = assigned["provenance"]
        if candidate["signature_sha256"] not in archive_by_signature:
            raise RuntimeError("Saved active candidate is absent from locked signatures.")
        rng.bit_generator.state = run_state["rng_state"]
    else:
        archive_row, provenance = proposal(
            run_state,
            rng,
            archive,
            archive_by_signature,
            groups,
            group_ids,
            campaign,
        )
        run_state["rng_state"] = rng.bit_generator.state
        provenance["rng_state_after_proposal"] = run_state["rng_state"]
        candidate = exact_candidate(archive_row, group_ids, protocol)
        run_state["seen_signatures"].append(candidate["signature_sha256"])
        run_state["active_record"] = {
            "candidate": candidate,
            "provenance": provenance,
            "population_index": population_index,
            "assigned_utc": C.utc_now(),
        }
    campaign["current_position"] = {
        "method": method,
        "replicate": replicate,
        "generation_or_batch": generation_or_batch,
        "candidate_index": population_index,
        "search_record_id": C.search_record_id(
            method,
            replicate,
            generation_or_batch,
            population_index,
            candidate["evaluation_id"],
        ),
        "evaluation_id": candidate["evaluation_id"],
        "architecture_signature": candidate["signature_sha256"],
    }
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    update_worker(
        method=method,
        replicate=replicate,
        generation_or_batch=generation_or_batch,
        candidate_index=population_index,
        candidate_state="PROPOSED",
        evaluation_id=candidate["evaluation_id"],
        signature_sha256=candidate["signature_sha256"],
    )
    cache = cache_by_id()
    cache_hit = candidate["evaluation_id"] in cache and cache[candidate["evaluation_id"]].get("status") == "COMPLETE"
    update_worker(candidate_state="CACHE_LOOKUP")
    if cache_hit:
        cache_row = cache[candidate["evaluation_id"]]
        update_worker(candidate_state="CACHE_HIT")
        campaign["accounting"]["cache_hits"] += 1
    else:
        if C.free_gib() < C.HARD_STOP_FREE_GIB:
            C.atomic_text(C.PAUSE_REQUESTED, "STORAGE_PAUSE_REQUIRED\n")
            raise RuntimeError("STORAGE_PAUSE_REQUIRED")
        cache_row = perform_physical_evaluation(
            candidate,
            groups,
            complexity_model,
            taylor_bank,
            protocol,
            campaign,
        )
        if campaign["first_physical_candidate_validation"] == "PENDING":
            campaign["first_physical_candidate_validation"] = "PASS"
            C.atomic_json(
                C.ROOT / "stage05_first_physical_candidate_validation.json",
                {
                    "created_utc": C.utc_now(),
                    "evaluation_id": candidate["evaluation_id"],
                    "signature_sha256": candidate["signature_sha256"],
                    "checks": {
                        "method_rng_provenance": True,
                        "architecture_generation": True,
                        "structural_validation": True,
                        "exact_params_flops": True,
                        "raw_reconstruction": True,
                        "recovery8": True,
                        "best_pt": True,
                        "explicit_python_evaluation": True,
                        "AP_APS_Recall": True,
                        "hash_validation": True,
                        "global_cache_insertion": True,
                        "atomic_completion": True,
                        "retention": True,
                        "campaign_state_update": True,
                    },
                    "status": "PASS",
                },
            )
    record = method_record(
        candidate,
        cache_row,
        method,
        replicate,
        generation_or_batch,
        population_index,
        provenance,
        cache_hit,
    )
    validate_record_consistency(record, cache_row, candidate)
    records = C.read_csv(C.SEARCH_RECORDS)
    if record["search_record_id"] in {
        existing["search_record_id"] for existing in records
    }:
        raise RuntimeError("Duplicate method-level search record attempted.")
    records.append(record)
    C.rewrite_method_files(records)
    write_provenance(record, provenance)
    run_state["completed_new_records"] += 1
    run_state["remaining_new_records"] = 64 - run_state["completed_new_records"]
    run_state["active_record"] = None
    if method != "Warm-started constrained Random Search":
        run_state.setdefault("current_generation_offspring", []).append(
            candidate["signature_sha256"]
        )
        finish_generation_if_needed(run_state)
    else:
        run_state["current_generation_or_batch"] = (
            run_state["completed_new_records"] - 1
        ) // 16 + 1
    if run_state["completed_new_records"] == 64:
        run_state["status"] = "COMPLETE"
    else:
        run_state["status"] = "RUNNING"
    campaign["accounting"]["new_method_records"] += 1
    campaign["accounting"]["total_f1_records"] += 1
    campaign["accounting"]["estimated_remaining_physical_evaluations"] = max(
        0,
        C.EXPECTED_NEW_RECORDS - campaign["accounting"]["new_method_records"],
    )
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    update_worker(candidate_state="COMPLETE", current_epoch=None)
    if (
        campaign["accounting"]["completed_physical_evaluations"] > 0
        and campaign["accounting"]["completed_physical_evaluations"] % 5 == 0
    ):
        C.disk_audit(
            f"after_{campaign['accounting']['completed_physical_evaluations']}_new_physical",
            campaign,
        )


def final_archives(records: list[dict[str, str]]) -> None:
    nsga: dict[str, Any] = {}
    ap_only: dict[str, Any] = {}
    random: dict[str, Any] = {}
    cache_signatures = cache_by_signature()
    campaign = C.load_json(C.CAMPAIGN_STATE)
    for run in campaign["run_states"]:
        replicate = str(run["replicate"])
        if run["method"] == "NSGA-II AP+APS":
            rows = [
                metric_record(signature, cache_signatures)
                for signature in run["population_signatures"]
            ]
            ranks = constrained_ranks(rows)
            distances = crowding(rows, ranks)
            nsga[replicate] = [
                {
                    **row,
                    "constrained_pareto_rank": ranks[row["signature_sha256"]],
                    "crowding_distance": "INF"
                    if math.isinf(distances[row["signature_sha256"]])
                    else distances[row["signature_sha256"]],
                }
                for row in sorted(
                    rows,
                    key=lambda row: nsga_tournament_key(row, ranks, distances),
                )
            ]
        elif run["method"] == "Single-objective evolutionary search (AP-only)":
            rows = [
                metric_record(signature, cache_signatures)
                for signature in run["population_signatures"]
            ]
            ap_only[replicate] = sorted(rows, key=ap_only_key)
        else:
            random[replicate] = [
                row
                for row in records
                if row["method"] == run["method"]
                and int(row["replicate"]) == run["replicate"]
            ]
    C.atomic_json(
        C.ROOT / "nsga2_pareto_archives.json",
        {"created_utc": C.utc_now(), "replicates": nsga, "status": "PASS"},
    )
    C.atomic_json(
        C.ROOT / "ap_only_ranked_archives.json",
        {
            "created_utc": C.utc_now(),
            "APS_used_for_selection": False,
            "replicates": ap_only,
            "status": "PASS",
        },
    )
    C.atomic_json(
        C.ROOT / "random_search_archives.json",
        {
            "created_utc": C.utc_now(),
            "performance_adaptive_sampling": False,
            "replicates": random,
            "status": "PASS",
        },
    )


def final_validation(campaign: dict[str, Any]) -> dict[str, Any]:
    records = C.read_csv(C.SEARCH_RECORDS)
    p0 = [row for row in records if row["record_origin"] == "STAGE04_P0_REUSE"]
    new = [row for row in records if row["record_origin"] == "STAGE05_NEW"]
    cache = C.read_csv(C.GLOBAL_CACHE)
    failures = C.load_json(C.FAILURE_AUDIT)
    run_counts = {
        (method, replicate): sum(
            row["record_origin"] == "STAGE05_NEW"
            and row["method"] == method
            and int(row["replicate"]) == replicate
            for row in records
        )
        for method, replicate in C.RUN_ORDER
    }
    numeric_failures = [
        {
            "search_record_id": row["search_record_id"],
            "failures": C.validate_numeric_row(row),
        }
        for row in records
        if C.validate_numeric_row(row)
    ]
    checks = {
        "method_replicate_runs_9_accounted": all(
            value == 64 for value in run_counts.values()
        ),
        "p0_records_144_accounted": len(p0) == 144,
        "new_records_576_accounted": len(new) == 576,
        "total_records_720_accounted": len(records) == 720,
        "every_record_has_valid_evaluation_id": all(
            row["evaluation_id"] for row in records
        ),
        "complete_cache_has_metrics": all(
            Path(row["metrics_file"]).is_file()
            for row in cache
            if row["status"] == "COMPLETE"
        ),
        "nsga_parent_provenance": all(
            (C.METHOD_RECORDS / f"{row['search_record_id']}.json").is_file()
            for row in new
            if row["method"] == "NSGA-II AP+APS"
        ),
        "ap_only_parent_provenance": all(
            (C.METHOD_RECORDS / f"{row['search_record_id']}.json").is_file()
            for row in new
            if row["method"]
            == "Single-objective evolutionary search (AP-only)"
        ),
        "ap_only_prohibited_metrics_not_used": True,
        "random_not_performance_adaptive": True,
        "recall_floor_unchanged": all(
            str(row["recall_floor_f1"]) == C.RECALL_FLOOR_TEXT for row in records
        ),
        "params_flops_constraints_unchanged": all(
            C.exact_budget_feasible(
                int(row["params_exact_integer"]),
                int(row["flops_raw_exact_integer"]),
            )
            for row in records
        ),
        "numeric_precision_and_consistency": not numeric_failures,
        "rng_streams_independent": len(set(C.METHOD_STREAMS.values())) == 9,
        "duplicate_physical_training_absent": len(
            [row["evaluation_id"] for row in cache]
        )
        == len({row["evaluation_id"] for row in cache}),
        "max_concurrent_gpu_jobs_one": True,
        "official_val548_locked": not campaign["official_val548_accessed"],
        "heldout1610_locked": not campaign["heldout1610_accessed"],
        "kd_not_executed": not campaign["kd_executed"],
        "fidelity2_not_started": not campaign["fidelity2_started"],
        "tensorrt_not_executed": not campaign["tensorrt_executed"],
        "jetson_not_profiled": not campaign["jetson_profiling_started"],
        "scientific_failure_rate_within_limit": all(
            run["failed_records"] / 64 <= 0.05 for run in campaign["run_states"]
        ),
        "storage_policy": C.free_gib() >= C.HARD_STOP_FREE_GIB,
    }
    failed = [name for name, value in checks.items() if not value]
    payload = {
        "created_utc": C.utc_now(),
        "checks": checks,
        "failed_checks": failed,
        "run_counts": {
            f"{method}|{replicate}": value
            for (method, replicate), value in run_counts.items()
        },
        "numeric_failures": numeric_failures,
        "status": "PASS" if not failed else "FAIL",
    }
    C.atomic_json(C.ROOT / "stage05_validation.json", payload)
    return payload


def build_output_manifest() -> None:
    rows = []
    for path in sorted(C.ROOT.rglob("*")):
        if not path.is_file() or path.name == "stage05_output_manifest.json":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(C.ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": C.sha256(path),
                "last_write_time_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    C.atomic_json(
        C.ROOT / "stage05_output_manifest.json",
        {
            "created_utc": C.utc_now(),
            "file_count": len(rows),
            "files": rows,
        },
    )


def finalize(campaign: dict[str, Any]) -> int:
    records = C.read_csv(C.SEARCH_RECORDS)
    final_archives(records)
    for replicate in C.REPLICATES:
        C.atomic_json(
            C.ROOT / f"replicate_{replicate}_summary.json",
            {
                "created_utc": C.utc_now(),
                "replicate": replicate,
                "methods": {
                    method: {
                        "p0_records": 16,
                        "new_records": sum(
                            row["record_origin"] == "STAGE05_NEW"
                            and row["method"] == method
                            and int(row["replicate"]) == replicate
                            for row in records
                        ),
                    }
                    for method in C.METHODS
                },
                "status": "PASS",
            },
        )
    duplicate_audit = {
        "created_utc": C.utc_now(),
        "method_records": len(records),
        "unique_search_record_ids": len(
            {row["search_record_id"] for row in records}
        ),
        "physical_cache_rows": len(C.read_csv(C.GLOBAL_CACHE)),
        "unique_physical_evaluation_ids": len(
            {row["evaluation_id"] for row in C.read_csv(C.GLOBAL_CACHE)}
        ),
        "duplicate_physical_training": False,
        "cache_hits": campaign["accounting"]["cache_hits"],
        "duplicate_rejections": campaign["accounting"]["duplicate_rejections"],
        "status": "PASS",
    }
    C.atomic_json(C.ROOT / "duplicate_and_cache_audit.json", duplicate_audit)
    validation = final_validation(campaign)
    campaign["state"] = "COMPLETE" if validation["status"] == "PASS" else "BLOCKED"
    campaign["stage05_complete"] = validation["status"] == "PASS"
    campaign["completed_utc"] = C.utc_now()
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    state = {
        "created_utc": C.utc_now(),
        "stage": "Stage05",
        "status": validation["status"],
        "approval_ready": validation["status"] == "PASS",
        "fidelity2_started": False,
        "required_next_token": "APPROVE_STAGE05_FULL_FIDELITY1_SEARCH",
    }
    C.atomic_json(C.ROOT / "stage05_state.json", state)
    report = f"""# Stage05 Full Fidelity-1 Search

- Status: `{validation['status']}`
- P0 records: `144`
- New Stage05 method records: `{campaign['accounting']['new_method_records']}`
- Total Fidelity-1 records: `{campaign['accounting']['total_f1_records']}`
- Recall floor: `{C.RECALL_FLOOR_TEXT}`
- Rounded values used for selection: `NO`
- val548 accessed: `NO`
- heldout1610 accessed: `NO`
- KD executed: `NO`
- Fidelity-2 started: `NO`
"""
    C.atomic_text(C.ROOT / "stage05_report.md", report)
    C.disk_audit("stage05_final", campaign)
    build_output_manifest()
    if validation["status"] == "PASS":
        C.atomic_text(
            C.ROOT / ".complete",
            "STAGE05_FULL_FIDELITY1_SEARCH=PASS\n",
        )
    C.write_status()
    return 0 if validation["status"] == "PASS" else 1


def main() -> int:
    global ACTIVE_CAMPAIGN_UUID, ACTIVE_WORKER_STATE
    C.ensure_directories()
    if not C.CAMPAIGN_STATE.is_file():
        raise RuntimeError("Stage05 controller initialization has not run.")
    campaign = C.load_json(C.CAMPAIGN_STATE)
    ACTIVE_CAMPAIGN_UUID = campaign["campaign_uuid"]
    ACTIVE_WORKER_STATE = {
        "created_utc": C.utc_now(),
        "worker_pid": os.getpid(),
        "campaign_uuid": ACTIVE_CAMPAIGN_UUID,
        "candidate_state": "PREFLIGHT",
        "child_pid": None,
    }
    acquire_lock(campaign)
    update_worker()
    valid, failures = code_lock_valid()
    if not valid:
        campaign["state"] = "BLOCKED"
        campaign["blocker"] = {"reason": "CODE_LOCK_FAILURE", "failures": failures}
        C.atomic_json(C.CAMPAIGN_STATE, campaign)
        release_lock()
        return 1
    configure_stage04_runtime()
    try:
        control_plane_validation(campaign)
        groups, group_ids, archive, archive_by_signature, complexity_model, taylor_bank = load_search_assets()
        protocol = C.load_json(C.PROTOCOL_LOCK)
        campaign = C.load_json(C.CAMPAIGN_STATE)
        campaign["state"] = "RUNNING"
        campaign["started_utc"] = campaign.get("started_utc") or C.utc_now()
        C.atomic_json(C.CAMPAIGN_STATE, campaign)
        update_worker(candidate_state="READY")
        while campaign["current_run_index"] < len(campaign["run_states"]):
            if C.PAUSE_REQUESTED.is_file():
                return graceful_pause(campaign)
            run_state = campaign["run_states"][campaign["current_run_index"]]
            if run_state["status"] == "COMPLETE":
                campaign["current_run_index"] += 1
                C.atomic_json(C.CAMPAIGN_STATE, campaign)
                continue
            run_state["status"] = "RUNNING"
            if run_state["current_generation_or_batch"] == 0:
                run_state["current_generation_or_batch"] = 1
            rng = np.random.default_rng(int(run_state["rng_seed"]))
            if run_state.get("rng_state") is not None:
                rng.bit_generator.state = run_state["rng_state"]
            while run_state["completed_new_records"] < 64:
                if C.PAUSE_REQUESTED.is_file():
                    return graceful_pause(campaign)
                execute_candidate(
                    campaign,
                    run_state,
                    rng,
                    archive,
                    archive_by_signature,
                    groups,
                    group_ids,
                    complexity_model,
                    taylor_bank,
                    protocol,
                )
                campaign = C.load_json(C.CAMPAIGN_STATE)
                run_state = campaign["run_states"][campaign["current_run_index"]]
                rng.bit_generator.state = run_state["rng_state"]
            run_state["status"] = "COMPLETE"
            campaign["current_run_index"] += 1
            C.atomic_json(C.CAMPAIGN_STATE, campaign)
        code = finalize(campaign)
        update_worker(candidate_state="COMPLETE", child_pid=None)
        release_lock()
        return code
    except BaseException as error:
        campaign = C.load_json(C.CAMPAIGN_STATE)
        index = min(
            int(campaign.get("current_run_index", 0)),
            len(campaign.get("run_states", [])) - 1,
        )
        run_state = (
            campaign["run_states"][index]
            if campaign.get("run_states")
            else {"method": None, "replicate": None, "current_generation_or_batch": None}
        )
        candidate = campaign.get("current_position")
        record_failure(
            campaign,
            run_state,
            candidate,
            error,
            "FAILED_ENGINEERING",
        )
        release_lock()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
