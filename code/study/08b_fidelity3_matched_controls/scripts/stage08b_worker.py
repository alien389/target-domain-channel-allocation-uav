from __future__ import annotations

import argparse
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
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import psutil
import torch

import stage08b_common as C


STAGE04_SCRIPT = C.PROJECT / "scripts" / "stage04_shared_p0_fidelity1.py"
STAGE07 = C.PROJECT / "07_fidelity2_confirmation"
STAGE08A = C.PROJECT / "08a_fidelity3_proposed_lock"
A3 = C.PROJECT / "protocol_amendment_A3_defer_kd_final_proposed_ablation"
R1_ROOT = C.PROJECT / "03a_taylor_dependency_gene_space" / "gene_space_R1"
GENE_JSON = R1_ROOT / "gene_space_manifest_R1.json"
TRAJECTORY_CSV = R1_ROOT / "global_taylor_trajectory.csv"
EVALUATOR = C.SCRIPTS / "evaluate_stage08b_inner_val.py"
RESULTS_CSV = C.ROOT / "stage08b_all_results.csv"
RETENTION_CSV = C.ROOT / "checkpoint_retention_manifest.csv"
FAILURE_AUDIT = C.ROOT / "failure_audit.json"
DISK_AUDIT = C.ROOT / "disk_usage_audit.json"

EXPECTED_STAGE08A_MANIFEST = "CC410EF097549381A883C0801148A411BBC7964536806ADD52927B93EC4A0032"
EXPECTED_STAGE08A_SOURCE_HASHES = "D7135BF5BB4E0CAFD1F489B48D085F7565CA7F2F5F3B8FEE3A9A1A466FFEC3EB"
EXPECTED_STAGE07_MANIFEST = "D8E293952CC9E62579AD134EC582A71FB08F57C494BFFE3BEED1AC42320B06DD"
EXPECTED_A3_MANIFEST = "0A77876FF7C4B706EF7A91F1C30572249DF4F253E7A624C1A2528CA81E1541E0"
SEARCH_PARENT_SHA = "15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659"
PROPOSED_SIGNATURE = "EEA3FB45FD3FD1BADD263DE055266F4F6B5EE34EE818F4C1D7026B764E862C80"
PROPOSED_CHECKPOINT_SHA = "593F8C715215266667A4E963C7588B02D198D1F9E15C78C20F062E856A005F58"
P_META = 2_507_047
F_META = 7_847_110_400

METHOD_AP_ONLY = "Single-objective evolutionary search (AP-only)"
METHOD_RANDOM = "Warm-started constrained Random Search"
ACTIVE: dict[str, Any] = {}
CAMPAIGN_UUID = ""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S4 = load_module("stage08b_stage04_locked", STAGE04_SCRIPT)
R1M = S4.R1M


def run_dir(candidate: dict[str, Any]) -> Path:
    return C.ROOT / candidate["directory_name"]


def configure_runtime() -> None:
    S4.OUTPUT = C.ROOT
    S4.PHYSICAL_ROOT = C.ROOT
    S4.RECOVERY_EPOCHS = 50
    S4.TRAINING_SEED = 42
    S4.DISK_STOP_GIB = 8.0
    S4.EVAL_WRAPPER = EVALUATOR
    S4.evaluation_directory = run_dir
    S4.run_logged = run_logged


def update_worker(**values: Any) -> None:
    ACTIVE.update(values)
    ACTIVE["updated_utc"] = C.utc_now()
    C.atomic_json(C.WORKER_STATE, ACTIVE)
    if C.LOCK.is_file():
        lock = C.load_json(C.LOCK, {})
        lock["latest_heartbeat"] = ACTIVE["updated_utc"]
        lock["child_pid"] = ACTIVE.get("child_pid")
        C.atomic_json(C.LOCK, lock)
    C.write_status()


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path, label: str) -> tuple[int, float]:
    lowered = " ".join(command).lower()
    if "yolov5m" in lowered or "teacher" in lowered or "train_stage10_kd" in lowered:
        raise RuntimeError("Teacher/KD token detected in a Recovery-50-only command")
    start = time.time()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
        process = subprocess.Popen(command, cwd=str(C.PROJECT), stdout=stdout, stderr=stderr, env=os.environ.copy())
        update_worker(child_pid=process.pid, child_command=subprocess.list2cmdline(command), child_label=label)
        while process.poll() is None:
            update_worker()
            time.sleep(15)
        code = int(process.returncode)
    update_worker(child_pid=None, child_command=None, child_label=None)
    return code, time.time() - start


def acquire_lock() -> None:
    others = [row for row in C.worker_processes() if row["pid"] != os.getpid()]
    if others:
        raise RuntimeError(f"Another Stage08B worker is active: {others}")
    if C.gpu_children():
        raise RuntimeError(f"A Recovery/evaluation process is active: {C.gpu_children()}")
    if C.LOCK.is_file():
        old = C.load_json(C.LOCK, {})
        if C.process_alive(old.get("worker_pid")):
            raise RuntimeError(f"Live Stage08B lock belongs to PID {old['worker_pid']}")
    C.atomic_json(C.LOCK, {"created_utc": C.utc_now(), "campaign_uuid": CAMPAIGN_UUID, "worker_pid": os.getpid(), "hostname": socket.gethostname(), "max_gpu_jobs": 1})


def release_lock() -> None:
    if C.LOCK.is_file():
        payload = C.load_json(C.LOCK, {})
        if int(payload.get("worker_pid", -1)) == os.getpid():
            C.LOCK.unlink()


def verify_manifest_contents(manifest_path: Path) -> list[str]:
    failures = []
    for row in C.load_json(manifest_path, {}).get("files", []):
        path = Path(row["path"])
        if not path.is_file() or C.sha256(path) != row["sha256"]:
            failures.append(str(path))
    return failures


def verify_upstream() -> None:
    decision = C.load_json(STAGE08A / "proposed_lock_decision.json", {})
    floor = C.load_json(STAGE08A / "recall_floor_f3.json", {})
    checks = {
        "stage08a_validation": C.load_json(STAGE08A / "stage08a_validation.json", {}).get("status") == "PASS",
        "stage08a_manifest_hash": C.sha256(STAGE08A / "stage08a_output_manifest.json") == EXPECTED_STAGE08A_MANIFEST,
        "stage08a_manifest_contents": not verify_manifest_contents(STAGE08A / "stage08a_output_manifest.json"),
        "stage08a_source_hashes": C.sha256(STAGE08A / "stage08a_source_hashes.json") == EXPECTED_STAGE08A_SOURCE_HASHES,
        "stage07_manifest": C.sha256(STAGE07 / "stage07_output_manifest.json") == EXPECTED_STAGE07_MANIFEST,
        "a3_validation": C.load_json(A3 / "amendment_A3_validation.json", {}).get("status") == "PASS",
        "a3_manifest": C.sha256(A3 / "amendment_A3_output_manifest.json") == EXPECTED_A3_MANIFEST,
        "selected_signature": decision.get("selected_architecture_signature") == PROPOSED_SIGNATURE,
        "selected_checkpoint": decision.get("selected_checkpoint_sha256") == PROPOSED_CHECKPOINT_SHA and C.sha256(Path(decision["selected_checkpoint"])) == PROPOSED_CHECKPOINT_SHA,
        "selected_complexity": int(decision.get("P_meta", -1)) == P_META and int(decision.get("F_meta_raw", -1)) == F_META,
        "recall_floor": floor.get("recall_floor_f3_full_precision") == C.RECALL_FLOOR_F3,
        "searchparent": C.sha256(R1M.SEARCH_PARENT) == SEARCH_PARENT_SHA,
        "no_gpu": not C.gpu_children(),
        "no_stage09": not (C.PROJECT / "09_final_architecture_lock").exists(),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"Stage08B upstream lock failed: {failed}")


def exact_text(value: Any) -> str:
    return format(float(value), ".17g")


def full_precision_delta(value: int, reference: int) -> Decimal:
    return abs(Decimal(value) - Decimal(reference)) / Decimal(reference)


def physical_rows() -> dict[str, dict[str, str]]:
    return {row["physical_candidate_id"]: row for row in C.read_csv(STAGE07 / "stage07_physical_evaluation_manifest.csv")}


def method_rows() -> list[dict[str, str]]:
    return C.read_csv(STAGE07 / "stage07_method_result_manifest.csv")


def near_match_rows(method: str) -> tuple[list[dict[str, Any]], Decimal]:
    physical = physical_rows()
    source = [row for row in method_rows() if row["method"] == method]
    if not source:
        raise RuntimeError(f"No Stage07 rows for {method}")
    for limit in (Decimal("0.01"), Decimal("0.015")):
        audited = []
        for row in source:
            prow = physical[row["physical_candidate_id"]]
            dp = full_precision_delta(int(row["params_exact_integer"]), P_META)
            df = full_precision_delta(int(row["flops_raw_exact_integer"]), F_META)
            eligible = row["status"] == "COMPLETE" and row["f2_feasibility"] == "F2_FEASIBLE" and prow["structural_validity"] == "PASS" and dp <= limit and df <= limit
            audited.append({**row, "delta_params": str(dp), "delta_flops": str(df), "near_match_limit": str(limit), "eligible": eligible, "exclusion_reason": "" if eligible else "STATUS_RECALL_STRUCTURE_OR_COMPLEXITY_FILTER"})
        eligible_rows = [row for row in audited if row["eligible"]]
        required = 1 if method == METHOD_AP_ONLY else 2
        if len(eligible_rows) >= required:
            return audited, limit
    raise RuntimeError(f"Insufficient near-matched candidates for {method} within 1.5%")


def select_ap_only() -> dict[str, Any]:
    audited, limit = near_match_rows(METHOD_AP_ONLY)
    eligible = [row for row in audited if row["eligible"]]
    selected = sorted(eligible, key=lambda row: (-Decimal(row["AP"]), -Decimal(row["Recall"]), int(row["flops_raw_exact_integer"]), int(row["params_exact_integer"]), row["physical_candidate_id"]))[0]
    payload = {
        "created_utc": C.utc_now(), "method": METHOD_AP_ONLY, "selection_source": "Stage07 Fidelity-2 AP-only method results only", "near_match_limit": str(limit),
        "fallback_used": limit > Decimal("0.01"), "aps_used_for_selection": False, "selection_rule": "highest AP; ties by Recall, FLOPs, Params, candidate ID",
        "selected_candidate_id": selected["physical_candidate_id"], "selected_architecture_signature": selected["architecture_signature"], "selected_f2_evidence": selected,
        "candidate_audit": audited, "status": "PASS",
    }
    C.atomic_json(C.ROOT / "ap_only_selection.json", payload)
    return payload


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = []
    for row in rows:
        ap, aps = Decimal(row["AP"]), Decimal(row["APS"])
        dominated = any(Decimal(other["AP"]) >= ap and Decimal(other["APS"]) >= aps and (Decimal(other["AP"]) > ap or Decimal(other["APS"]) > aps) for other in rows if other is not row)
        if not dominated:
            front.append(row)
    return front


def knee(front: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not front:
        raise RuntimeError("Empty Random Pareto front")
    ap_values = [Decimal(row["AP"]) for row in front]
    aps_values = [Decimal(row["APS"]) for row in front]
    ap_min, ap_max, aps_min, aps_max = min(ap_values), max(ap_values), min(aps_values), max(aps_values)
    points = []
    for row in front:
        x = (Decimal(row["AP"]) - ap_min) / (ap_max - ap_min) if ap_max != ap_min else Decimal(0)
        y = (Decimal(row["APS"]) - aps_min) / (aps_max - aps_min) if aps_max != aps_min else Decimal(0)
        points.append({"row": row, "x": x, "y": y})
    low_ap = min(points, key=lambda item: (item["x"], -item["y"], item["row"]["physical_candidate_id"]))
    high_ap = max(points, key=lambda item: (item["x"], -item["y"], item["row"]["physical_candidate_id"]))
    dx, dy = high_ap["x"] - low_ap["x"], high_ap["y"] - low_ap["y"]
    denominator = (dx * dx + dy * dy).sqrt() if dx or dy else Decimal(1)
    audit = []
    for item in points:
        distance = abs(dy * item["x"] - dx * item["y"] + high_ap["x"] * low_ap["y"] - high_ap["y"] * low_ap["x"]) / denominator
        audit.append({**item["row"], "normalized_AP": str(item["x"]), "normalized_APS": str(item["y"]), "perpendicular_distance": str(distance)})
    selected = sorted(audit, key=lambda row: (-Decimal(row["perpendicular_distance"]), -Decimal(row["AP"]), -Decimal(row["APS"]), row["physical_candidate_id"]))[0]
    return selected, audit


def select_random() -> dict[str, Any]:
    audited, limit = near_match_rows(METHOD_RANDOM)
    eligible = [row for row in audited if row["eligible"]]
    front = pareto_front(eligible)
    selected_a, knee_audit = knee(front)
    remaining = [row for row in eligible if row["physical_candidate_id"] != selected_a["physical_candidate_id"]]
    selected_b = sorted(remaining, key=lambda row: (-Decimal(row["AP"]), -Decimal(row["Recall"]), int(row["flops_raw_exact_integer"]), int(row["params_exact_integer"]), row["physical_candidate_id"]))[0]
    payload = {
        "created_utc": C.utc_now(), "method": METHOD_RANDOM, "selection_source": "Stage07 Fidelity-2 Random Search results only", "near_match_limit": str(limit),
        "fallback_used": limit > Decimal("0.01"), "candidate_a_rule": "deterministic normalized AP-APS Pareto knee", "candidate_b_rule": "highest-AP eligible candidate distinct from A",
        "candidate_a_id": selected_a["physical_candidate_id"], "candidate_a_signature": selected_a["architecture_signature"],
        "candidate_b_id": selected_b["physical_candidate_id"], "candidate_b_signature": selected_b["architecture_signature"],
        "eligible_audit": audited, "pareto_front_ids": [row["physical_candidate_id"] for row in front], "knee_audit": knee_audit, "status": "PASS",
    }
    C.atomic_json(C.ROOT / "random_selection.json", payload)
    return payload


def removed_score(widths: dict[str, int], groups: list[dict[str, Any]], taylor_bank: dict[str, Any]) -> Decimal:
    total = Decimal(0)
    for group in groups:
        gid = group["group_id"]
        removed = int(group["original_width"]) - int(widths[gid])
        total += sum((Decimal(str(value)) for value in taylor_bank[gid]["normalized_scores_low_to_high"][:removed]), Decimal(0))
    return total


def match_manual(groups: list[dict[str, Any]], complexity: Any, taylor_bank: dict[str, Any]) -> dict[str, Any]:
    ids = [group["group_id"] for group in groups]
    allowed = {group["group_id"]: R1M.parse_allowed(group["allowed_widths"]) for group in groups}
    current = {group["group_id"]: int(group["original_width"]) for group in groups}
    trajectory_states = []
    for row in C.read_csv(TRAJECTORY_CSV):
        if row["step_id"] != "0":
            current[row["selected_dependency_group"]] = int(row["new_width"])
        trajectory_states.append((f"TRAJECTORY_STEP_{row['step_id']}", dict(current), "TRAJECTORY"))
    unique: dict[str, tuple[str, dict[str, int], str]] = {}
    for label, widths, origin in trajectory_states:
        signature = R1M.signature_sha(ids, tuple(widths[gid] for gid in ids))
        unique[signature] = (label, widths, origin)
    for label, widths, _ in trajectory_states:
        for gid in ids:
            values = allowed[gid]
            index = values.index(widths[gid])
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(values):
                    neighbor = dict(widths)
                    neighbor[gid] = int(values[neighbor_index])
                    signature = R1M.signature_sha(ids, tuple(neighbor[key] for key in ids))
                    unique.setdefault(signature, (f"{label}_NEIGHBOR_{gid}_{values[neighbor_index]}", neighbor, "ONE_STEP_ALLOWED_WIDTH_NEIGHBOR"))
    audit = []
    for signature, (label, widths, origin) in unique.items():
        values = complexity.calculate(widths)
        params = int(values["params"])
        flops = int(values["macs"]) * 2
        dp = full_precision_delta(params, P_META)
        df = full_precision_delta(flops, F_META)
        budget_valid = C.PARAMS_LOWER <= params <= C.PARAMS_UPPER and C.FLOPS_LOWER <= flops <= C.FLOPS_UPPER
        audit.append({"state_id": label, "origin": origin, "architecture_signature": signature, "params_exact_integer": params, "flops_raw_exact_integer": flops, "flops_g_full_precision": exact_text(Decimal(flops) / Decimal(1_000_000_000)), "delta_params": str(dp), "delta_flops": str(df), "complexity_distance": str(max(dp, df)), "cumulative_taylor_removal_score": str(removed_score(widths, groups, taylor_bank)), "b6_budget_valid": budget_valid, "widths": widths})
    valid = [row for row in audit if row["b6_budget_valid"]]
    selected = sorted(valid, key=lambda row: (Decimal(row["complexity_distance"]), Decimal(row["delta_flops"]), Decimal(row["delta_params"]), Decimal(row["cumulative_taylor_removal_score"]), row["architecture_signature"]))[0]
    if Decimal(selected["complexity_distance"]) > Decimal("0.015"):
        raise RuntimeError("MatchedManual complexity distance exceeds 1.5%")
    C.atomic_csv(C.ROOT / "complexity_matching.csv", [{key: value for key, value in row.items() if key != "widths"} for row in audit])
    C.atomic_json(C.ROOT / "complexity_matching.json", {"created_utc": C.utc_now(), "states_considered": len(audit), "b6_valid_states": len(valid), "matching_rule": "min max(delta_params, delta_flops); ties by FLOPs, Params, Taylor score, signature", "selected": selected, "status": "PASS"})
    payload = {"created_utc": C.utc_now(), "candidate_id": f"F3CTRL-MATCHEDMANUAL-{selected['architecture_signature'][:16]}", **selected, "match_class": "PREFERRED_NEAR_EXACT" if Decimal(selected["complexity_distance"]) <= Decimal("0.005") else "STANDARD_PASS" if Decimal(selected["complexity_distance"]) <= Decimal("0.01") else "CONDITIONAL_WARNING", "raw_checkpoint_status": "CONSTRUCTION_PENDING", "accuracy_evaluated_before_recovery50": False, "status": "MATCHED_AND_FROZEN"}
    previous = C.load_json(C.ROOT / "matched_manual_generation.json", {}) or {}
    if previous.get("architecture_signature") == selected["architecture_signature"] and previous.get("raw_checkpoint_status") == "FROZEN_BEFORE_ACCURACY_TRAINING":
        for key in ("raw_checkpoint", "raw_checkpoint_sha256", "reconstruction_manifest", "reconstruction_manifest_sha256", "channel_index_map", "channel_index_map_sha256", "raw_checkpoint_status"):
            payload[key] = previous[key]
        payload["status"] = "CONSTRUCTED_AND_FROZEN"
    C.atomic_json(C.ROOT / "matched_manual_generation.json", payload)
    return payload


def create_protocol_lock() -> dict[str, Any]:
    path = C.ROOT / "stage08b_protocol_lock.json"
    existing = C.load_json(path, {}) or {}
    if existing:
        required = existing.get("status") == "LOCKED" and existing.get("P_meta") == P_META and existing.get("F_meta_raw") == F_META and existing.get("recall_floor_f3") == C.RECALL_FLOOR_F3 and existing.get("recovery_epochs") == 50 and existing.get("kd1") is False and existing.get("kd2") is False and existing.get("teacher_loading") is False
        if not required:
            raise RuntimeError("Existing Stage08B protocol lock conflicts with the approved protocol")
        return existing
    payload = {
        "schema": "stage08b_recovery50_only_protocol_v1", "created_utc": C.utc_now(), "approval_token_consumed": "APPROVE_STAGE08A_PROPOSED_LOCK",
        "stage08a_approved": True, "amendment_A3_verified": True, "P_meta": P_META, "F_meta_raw": F_META, "F_meta_g": "7.8471104", "recall_floor_f3": C.RECALL_FLOOR_F3,
        "near_match_primary_limit": "0.01", "near_match_fallback_limit": "0.015", "execution_order": ["MatchedManual-B6", "Meta-AP-only-B6 finalist", "Random candidate A", "Random candidate B"],
        "recovery_epochs": 50, "training_seed": 42, "train": "meta-train5177", "validation": "meta-inner-val1294", "imgsz": 640, "batch_size": 16,
        "best_checkpoint_rule": "meta-inner-val1294 fitness", "explicit_evaluation": {"precision": "FP32", "images": 1294, "conf": 0.001, "iou": 0.60, "max_det": 500, "augment": False, "tta": False},
        "kd1": False, "kd2": False, "teacher_loading": False, "coco_teacher_substitution": False, "official_val548_locked": True, "heldout1610_locked": True, "stage09_started": False, "status": "LOCKED",
    }
    payload["stage08b_protocol_sha256"] = C.hash_text(C.canonical_json(payload))
    C.atomic_json(path, payload)
    return payload


def source_hashes() -> dict[str, Any]:
    paths = [STAGE08A / "stage08a_output_manifest.json", STAGE08A / "stage08a_source_hashes.json", STAGE08A / "proposed_lock_decision.json", STAGE08A / "recall_floor_f3.json", STAGE07 / "stage07_output_manifest.json", STAGE07 / "stage07_method_result_manifest.csv", STAGE07 / "stage07_physical_evaluation_manifest.csv", A3 / "amendment_A3_output_manifest.json", TRAJECTORY_CSV, GENE_JSON, STAGE04_SCRIPT, C.SCRIPTS / "stage08b_common.py", C.SCRIPTS / "stage08b_worker.py", EVALUATOR, S4.TRAIN_WRAPPER, S4.HYP_YAML, S4.DATA_YAML, R1M.SEARCH_PARENT, S4.TAYLOR_RANKING, S4.DEPENDENCY_GRAPH, S4.VISDRONE_PARITY_AUDIT]
    return {"created_utc": C.utc_now(), "files": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": C.sha256(path)} for path in paths], "status": "PASS"}


def candidate_from_stage07(role: str, candidate_id: str, directory_name: str, index: int, protocol_hash: str, groups: list[dict[str, Any]], complexity: Any) -> dict[str, Any]:
    row = physical_rows()[candidate_id]
    reconstruction = Path(row["reconstruction_manifest"])
    channel_map = Path(row["channel_index_map"])
    if C.sha256(reconstruction) != row["reconstruction_manifest_sha256"] or C.sha256(channel_map) != row["channel_index_map_sha256"]:
        raise RuntimeError(f"Stage07 frozen reconstruction evidence changed: {candidate_id}")
    widths = C.load_json(reconstruction, {})["widths"]
    ids = [group["group_id"] for group in groups]
    signature = R1M.signature_sha(ids, tuple(int(widths[gid]) for gid in ids))
    if signature != row["architecture_signature"]:
        raise RuntimeError(f"Stage07 width signature mismatch: {candidate_id}")
    analytic = complexity.calculate(widths)
    flops_raw = int(analytic["macs"]) * 2
    if int(analytic["params"]) != int(row["params_exact_integer"]) or flops_raw != int(row["flops_raw_exact_integer"]):
        raise RuntimeError(f"Stage07 exact complexity mismatch: {candidate_id}")
    identity = C.hash_text(C.canonical_json({"stage": "Stage08B_Recovery50_only", "signature": signature, "protocol_sha256": protocol_hash, "seed": 42}))
    candidate: dict[str, Any] = {"candidate_role": role, "physical_candidate_id": candidate_id, "unique_signature_index": index, "directory_name": directory_name, "evaluation_id": identity, "signature_sha256": signature, "fused_params": int(row["params_exact_integer"]), "fused_flops_g": float(row["flops_g_full_precision"]), "params_exact_integer": int(row["params_exact_integer"]), "flops_raw_exact_integer": flops_raw, "flops_g_full_precision": row["flops_g_full_precision"], "expected_channel_map_sha256": row["channel_index_map_sha256"], "source_stage07_physical_manifest": row}
    candidate.update({gid: int(widths[gid]) for gid in ids})
    return candidate


def build_candidates(protocol: dict[str, Any], groups: list[dict[str, Any]], complexity: Any, matched: dict[str, Any], ap: dict[str, Any], random: dict[str, Any]) -> list[dict[str, Any]]:
    ids = [group["group_id"] for group in groups]
    signature = matched["architecture_signature"]
    identity = C.hash_text(C.canonical_json({"stage": "Stage08B_Recovery50_only", "signature": signature, "protocol_sha256": protocol["stage08b_protocol_sha256"], "seed": 42}))
    manual: dict[str, Any] = {"candidate_role": "MatchedManual-B6", "physical_candidate_id": matched["candidate_id"], "unique_signature_index": 1, "directory_name": "matched_manual_f3", "evaluation_id": identity, "signature_sha256": signature, "fused_params": int(matched["params_exact_integer"]), "fused_flops_g": float(matched["flops_g_full_precision"]), "params_exact_integer": int(matched["params_exact_integer"]), "flops_raw_exact_integer": int(matched["flops_raw_exact_integer"]), "flops_g_full_precision": matched["flops_g_full_precision"], "expected_channel_map_sha256": None, "matching_evidence": matched}
    manual.update({gid: int(matched["widths"][gid]) for gid in ids})
    candidates = [manual]
    candidates.append(candidate_from_stage07("Meta-AP-only-B6 finalist", ap["selected_candidate_id"], "ap_only_f3", 2, protocol["stage08b_protocol_sha256"], groups, complexity))
    candidates.append(candidate_from_stage07("Random candidate A", random["candidate_a_id"], "random_a_f3", 3, protocol["stage08b_protocol_sha256"], groups, complexity))
    candidates.append(candidate_from_stage07("Random candidate B", random["candidate_b_id"], "random_b_f3", 4, protocol["stage08b_protocol_sha256"], groups, complexity))
    if len({row["signature_sha256"] for row in candidates}) != 4:
        raise RuntimeError("Stage08B selected controls are not four distinct architectures")
    return candidates


def preflight() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, Any]]:
    verify_upstream()
    protocol = create_protocol_lock()
    groups = C.load_json(GENE_JSON, {})["groups"]
    parent = R1M.load_source_model(R1M.SEARCH_PARENT, "stage08b_parent")
    complexity = R1M.ExactComplexityModel(parent.model, groups)
    taylor_bank, taylor_manifest = R1M.taylor_scope_bank(groups)
    if taylor_manifest["checkpoint_sha256"] != SEARCH_PARENT_SHA:
        raise RuntimeError("Taylor ranking SearchParent binding failed")
    ap = select_ap_only()
    random = select_random()
    matched = match_manual(groups, complexity, taylor_bank)
    candidates = build_candidates(protocol, groups, complexity, matched, ap, random)
    C.atomic_json(C.ROOT / "stage08b_source_hashes.json", source_hashes())
    C.atomic_json(C.ROOT / "stage08b_preflight.json", {"created_utc": C.utc_now(), "checks": {"stage08a": True, "A3": True, "four_distinct_candidates": len(candidates) == 4, "matchedmanual_first": candidates[0]["candidate_role"] == "MatchedManual-B6", "ap_only_aps_not_used": ap["aps_used_for_selection"] is False, "random_source_isolated": True, "kd_disabled": True, "teacher_loading_disabled": True, "no_active_gpu": not C.gpu_children()}, "status": "PASS"})
    return candidates, groups, complexity, taylor_bank


def disk_sample(event: str) -> None:
    payload = C.load_json(DISK_AUDIT, {"created_utc": C.utc_now(), "samples": []})
    payload["samples"].append({"timestamp_utc": C.utc_now(), "event": event, "free_gib": C.free_gib()})
    C.atomic_json(DISK_AUDIT, payload)


def train_one(candidate: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    directory = run_dir(candidate)
    command = S4.training_command(directory, Path(raw["raw_checkpoint"]))
    if any(token in " ".join(command).lower() for token in ("teacher", "yolov5m", "kd")):
        raise RuntimeError("Recovery command contains a forbidden teacher/KD token")
    S4.text_write(directory / "train_command.txt", S4.command_text(command) + "\n")
    if S4.training_complete(directory):
        return S4.recovered_training_execution(directory)
    if (directory / "results.csv").exists() or (directory / "weights").exists():
        S4.clear_partial_training(directory)
    attempts = []
    for attempt in (1, 2):
        stdout = directory / f"train_attempt_{attempt}_stdout.log"
        stderr = directory / f"train_attempt_{attempt}_stderr.log"
        code, duration = run_logged(command, stdout, stderr, f"{candidate['candidate_role']} Recovery-50 attempt {attempt}")
        transient = S4.transient_failure(stdout, stderr) if code else False
        attempts.append({"attempt": attempt, "return_code": code, "duration_seconds": duration, "stdout": str(stdout), "stderr": str(stderr), "transient_failure": transient})
        if code == 0 and S4.training_complete(directory):
            shutil.copy2(stdout, directory / "stdout.log")
            shutil.copy2(stderr, directory / "stderr.log")
            result = {"created_utc": C.utc_now(), "training_reused": False, "training_duration_seconds": duration, "attempts": attempts}
            C.atomic_json(directory / "training_execution.json", result)
            return result
        if attempt == 1 and transient:
            S4.clear_partial_training(directory)
            continue
        raise RuntimeError(f"Recovery-50 failed: {candidate['candidate_role']}")
    raise AssertionError("Unreachable")


def checkpoint_hashes(directory: Path) -> dict[str, Any]:
    paths = {"raw_checkpoint": directory / "raw_checkpoint.pt", "best_checkpoint": directory / "weights" / "best.pt", "last_checkpoint_before_deletion": directory / "weights" / "last.pt"}
    payload = {"created_utc": C.utc_now()}
    for key, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"Missing checkpoint: {path}")
        payload[key] = {"path": str(path), "sha256": C.sha256(path), "size_bytes": path.stat().st_size}
    C.atomic_json(directory / "checkpoint_hashes_before_retention.json", payload)
    return payload


def metric_texts(metrics: dict[str, Any]) -> dict[str, str]:
    return {"AP": exact_text(metrics["AP_0.5_0.95"]), "AP50": exact_text(metrics["AP50"]), "AP75": exact_text(metrics["AP75"]), "APS": exact_text(metrics["APS"]), "APM": exact_text(metrics["APM"]), "APL": exact_text(metrics["APL"]), "Precision": exact_text(metrics["precision"]), "Recall": exact_text(metrics["recall"]), "F1": exact_text(metrics["f1"])}


def classify_row(row: dict[str, Any]) -> str:
    finite = all(math.isfinite(float(row[key])) for key in ("AP", "APS", "Recall"))
    feasible = Decimal(row["Recall"]) >= Decimal(C.RECALL_FLOOR_F3) and C.PARAMS_LOWER <= int(row["params_exact_integer"]) <= C.PARAMS_UPPER and C.FLOPS_LOWER <= int(row["flops_raw_exact_integer"]) <= C.FLOPS_UPPER and row["structural_validity"] == "PASS" and row["numerical_validity"] == "PASS" and finite
    return "F3_FEASIBLE" if feasible else "F3_INFEASIBLE"


def commit_candidate(candidate: dict[str, Any], groups: list[dict[str, Any]], complexity: Any, taylor_bank: dict[str, Any]) -> None:
    directory = run_dir(candidate)
    if (directory / ".complete").is_file():
        return
    if C.free_gib() < C.HARD_STOP_FREE_GIB:
        raise RuntimeError("Free disk below 8 GiB")
    update_worker(candidate_role=candidate["candidate_role"], candidate_id=candidate["physical_candidate_id"], run_number=candidate["unique_signature_index"], run_state="FRESH_RECONSTRUCTION", run_directory=str(directory))
    raw = S4.reconstruct_raw(candidate, groups, complexity, taylor_bank)
    raw.update({"stage": "Stage08B Recovery-50-only", "candidate_role": candidate["candidate_role"], "fresh_reconstruction_from_searchparent": True, "f1_f2_checkpoint_initialization": False, "kd_executed": False, "teacher_loaded": False, "params_exact_integer": candidate["params_exact_integer"], "flops_raw_exact_integer": candidate["flops_raw_exact_integer"], "flops_g_full_precision": candidate["flops_g_full_precision"]})
    reconstruction = directory / "raw_reconstruction_audit.json"
    C.atomic_json(reconstruction, raw)
    channel_map_sha = C.sha256(directory / "channel_index_map.csv")
    if candidate.get("expected_channel_map_sha256") and channel_map_sha != candidate["expected_channel_map_sha256"]:
        raise RuntimeError("Fresh channel-index map does not match Stage07 frozen source")
    if candidate["candidate_role"] == "MatchedManual-B6":
        generation = C.load_json(C.ROOT / "matched_manual_generation.json", {})
        generation.update({"raw_checkpoint": raw["raw_checkpoint"], "raw_checkpoint_sha256": C.sha256(Path(raw["raw_checkpoint"])), "reconstruction_manifest": str(reconstruction), "reconstruction_manifest_sha256": C.sha256(reconstruction), "channel_index_map": str(directory / "channel_index_map.csv"), "channel_index_map_sha256": channel_map_sha, "raw_checkpoint_status": "FROZEN_BEFORE_ACCURACY_TRAINING", "status": "CONSTRUCTED_AND_FROZEN"})
        C.atomic_json(C.ROOT / "matched_manual_generation.json", generation)
    update_worker(run_state="TRAINING_RECOVERY50", training_start_utc=C.utc_now())
    training = train_one(candidate, raw)
    update_worker(run_state="EXPLICIT_FP32_EVALUATION")
    evaluation = S4.evaluate_one(candidate)
    metrics_path = directory / "explicit_inner_val_metrics.json"
    metrics = C.load_json(metrics_path, {})
    metrics.update({"candidate_role": candidate["candidate_role"], "physical_candidate_id": candidate["physical_candidate_id"], "params_exact_integer": candidate["params_exact_integer"], "flops_raw_exact_integer": candidate["flops_raw_exact_integer"], "flops_g_full_precision": candidate["flops_g_full_precision"], "training_duration_seconds": training["training_duration_seconds"], "explicit_evaluation_duration_seconds": evaluation["evaluation_duration_seconds"], "fresh_reconstruction_from_searchparent": True, "f1_f2_checkpoint_initialization": False, "kd_enabled": False, "teacher_loaded": False, "official_val548_accessed": False, "heldout1610_accessed": False})
    metrics.update({key.lower() + "_full_precision": value for key, value in metric_texts(metrics).items()})
    C.atomic_json(metrics_path, metrics)
    if int(metrics["image_count"]) != 1294 or metrics["precision_mode"] != "FP32":
        raise RuntimeError("Explicit evaluation contract failed")
    hashes = checkpoint_hashes(directory)
    protocol = C.load_json(C.ROOT / "stage08b_protocol_lock.json", {})
    manifest_path = directory / "run_manifest.json"
    run_manifest = {"created_utc": C.utc_now(), "stage": "Stage08B Recovery-50-only", "candidate_role": candidate["candidate_role"], "physical_candidate_id": candidate["physical_candidate_id"], "evaluation_id": candidate["evaluation_id"], "architecture_signature": candidate["signature_sha256"], "fresh_reconstruction": True, "f1_f2_checkpoint_initialization": False, "recovery_epochs": 50, "training_seed": 42, "kd_executed": False, "teacher_loaded": False, "training": training, "explicit_evaluation": evaluation, "metrics_sha256": C.sha256(metrics_path), "checkpoints_before_retention": hashes, "protocol_sha256": protocol["stage08b_protocol_sha256"], "official_val548_accessed": False, "heldout1610_accessed": False, "status": "COMPLETE"}
    C.atomic_json(manifest_path, run_manifest)
    values = metric_texts(metrics)
    row: dict[str, Any] = {"candidate_role": candidate["candidate_role"], "physical_candidate_id": candidate["physical_candidate_id"], "evaluation_id": candidate["evaluation_id"], "architecture_signature": candidate["signature_sha256"], **values, "params_exact_integer": candidate["params_exact_integer"], "flops_raw_exact_integer": candidate["flops_raw_exact_integer"], "flops_g_full_precision": candidate["flops_g_full_precision"], "recall_floor_f3": C.RECALL_FLOOR_F3, "best_epoch_zero_based": metrics["best_epoch"]["best_epoch_zero_based"], "best_epoch_one_based": metrics["best_epoch"]["best_epoch_one_based"], "best_fitness": exact_text(metrics["best_epoch"]["best_fitness"]), "training_duration_seconds": training["training_duration_seconds"], "evaluation_duration_seconds": evaluation["evaluation_duration_seconds"], "best_checkpoint": str(directory / "weights" / "best.pt"), "best_checkpoint_sha256": hashes["best_checkpoint"]["sha256"], "raw_checkpoint": str(directory / "raw_checkpoint.pt"), "raw_checkpoint_sha256": hashes["raw_checkpoint"]["sha256"], "last_checkpoint_sha256_before_deletion": hashes["last_checkpoint_before_deletion"]["sha256"], "metrics_path": str(metrics_path), "metrics_sha256": C.sha256(metrics_path), "reconstruction_manifest": str(reconstruction), "reconstruction_manifest_sha256": C.sha256(reconstruction), "channel_index_map": str(directory / "channel_index_map.csv"), "channel_index_map_sha256": channel_map_sha, "run_manifest": str(manifest_path), "run_manifest_sha256": C.sha256(manifest_path), "structural_validity": "PASS", "numerical_validity": "PASS", "kd_executed": False, "teacher_loaded": False, "status": "COMPLETE"}
    row["f3_feasibility"] = classify_row(row)
    rows = [old for old in C.read_csv(RESULTS_CSV) if old["physical_candidate_id"] != candidate["physical_candidate_id"]] + [row]
    rows.sort(key=lambda old: next(i for i, role in enumerate(("MatchedManual-B6", "Meta-AP-only-B6 finalist", "Random candidate A", "Random candidate B")) if role == old["candidate_role"]))
    C.atomic_csv(RESULTS_CSV, rows)
    C.atomic_json(C.ROOT / "stage08b_all_results.json", {"created_utc": C.utc_now(), "protocol": "Recovery-50-only", "recall_floor_f3": C.RECALL_FLOOR_F3, "results": rows})
    retention = C.read_csv(RETENTION_CSV)
    retention.append({"candidate_id": candidate["physical_candidate_id"], "artifact_role": "raw_checkpoint", "path": hashes["raw_checkpoint"]["path"], "sha256": hashes["raw_checkpoint"]["sha256"], "size_bytes": hashes["raw_checkpoint"]["size_bytes"], "decision": "RETAINED"})
    retention.append({"candidate_id": candidate["physical_candidate_id"], "artifact_role": "best_checkpoint", "path": hashes["best_checkpoint"]["path"], "sha256": hashes["best_checkpoint"]["sha256"], "size_bytes": hashes["best_checkpoint"]["size_bytes"], "decision": "RETAINED"})
    last = directory / "weights" / "last.pt"
    retention.append({"candidate_id": candidate["physical_candidate_id"], "artifact_role": "last_checkpoint", "path": str(last), "sha256": hashes["last_checkpoint_before_deletion"]["sha256"], "size_bytes": hashes["last_checkpoint_before_deletion"]["size_bytes"], "decision": "DELETED_AFTER_HASH_AND_ATOMIC_COMMIT", "timestamp_utc": C.utc_now()})
    if last.exists():
        last.unlink()
    predictions = directory / "explicit_inner_val_predictions"
    if predictions.exists():
        shutil.rmtree(predictions)
    C.atomic_csv(RETENTION_CSV, retention)
    C.atomic_text(directory / ".complete", f"STAGE08B_F3_RECOVERY50=COMPLETE\nCANDIDATE={candidate['physical_candidate_id']}\n")
    update_worker(run_state="ATOMICALLY_COMMITTED")
    disk_sample(f"completed_{candidate['unique_signature_index']}_of_4")
    if candidate["candidate_role"] in ("MatchedManual-B6", "Meta-AP-only-B6 finalist") and row["f3_feasibility"] != "F3_FEASIBLE":
        raise RuntimeError("BLOCKED_REQUIRED_CONTROL_INFEASIBLE")


def select_random_lock(rows: list[dict[str, str]]) -> dict[str, Any]:
    random_rows = [row for row in rows if row["candidate_role"].startswith("Random candidate")]
    feasible = [row for row in random_rows if row["f3_feasibility"] == "F3_FEASIBLE"]
    if not feasible:
        return {"created_utc": C.utc_now(), "status": "BLOCKED_NO_FEASIBLE_RANDOM_CONTROL", "selected": None}
    if len(feasible) == 1:
        selected = feasible[0]
        clause = "ONLY_ONE_F3_FEASIBLE"
    else:
        a, b = random_rows
        ap_a, ap_b, aps_a, aps_b = Decimal(a["AP"]), Decimal(b["AP"]), Decimal(a["APS"]), Decimal(b["APS"])
        a_dom = ap_a >= ap_b and aps_a >= aps_b and (ap_a > ap_b or aps_a > aps_b)
        b_dom = ap_b >= ap_a and aps_b >= aps_a and (ap_b > ap_a or aps_b > aps_a)
        if a_dom or b_dom:
            selected = a if a_dom else b
            clause = "DOMINANCE_AP_AND_APS"
        elif abs(ap_a - ap_b) >= Decimal("0.001"):
            selected = a if ap_a > ap_b else b
            clause = "NONDOMINATED_ABSOLUTE_AP_DIFFERENCE_GE_0.001_HIGHER_AP"
        elif aps_a != aps_b:
            selected = a if aps_a > aps_b else b
            clause = "NONDOMINATED_AP_DIFFERENCE_LT_0.001_HIGHER_APS"
        elif Decimal(a["Recall"]) != Decimal(b["Recall"]):
            selected = a if Decimal(a["Recall"]) > Decimal(b["Recall"]) else b
            clause = "TIE_HIGHER_RECALL"
        elif int(a["flops_raw_exact_integer"]) != int(b["flops_raw_exact_integer"]):
            selected = a if int(a["flops_raw_exact_integer"]) < int(b["flops_raw_exact_integer"]) else b
            clause = "TIE_LOWER_FLOPS"
        elif int(a["params_exact_integer"]) != int(b["params_exact_integer"]):
            selected = a if int(a["params_exact_integer"]) < int(b["params_exact_integer"]) else b
            clause = "TIE_LOWER_PARAMS"
        else:
            selected = min((a, b), key=lambda row: row["physical_candidate_id"])
            clause = "TIE_DETERMINISTIC_CANDIDATE_ID"
    rejected = next((row for row in random_rows if row["physical_candidate_id"] != selected["physical_candidate_id"]), None)
    return {"created_utc": C.utc_now(), "status": "PASS", "label": "RandomSearch-B6 provisional locked architecture", "selected_candidate_id": selected["physical_candidate_id"], "selected_architecture_signature": selected["architecture_signature"], "selected_checkpoint": selected["best_checkpoint"], "selected_checkpoint_sha256": selected["best_checkpoint_sha256"], "rejected_candidate_id": rejected["physical_candidate_id"] if rejected else None, "rule_clause": clause, "manual_visual_selection": False, "kd_executed": False, "teacher_loaded": False}


def build_output_manifest() -> None:
    rows = []
    for path in sorted(C.ROOT.rglob("*")):
        if not path.is_file() or path == C.ROOT / "stage08b_output_manifest.json" or C.RUNTIME in path.parents:
            continue
        rows.append({"path": str(path), "relative_path": str(path.relative_to(C.ROOT)), "size_bytes": path.stat().st_size, "sha256": C.sha256(path)})
    C.atomic_json(C.ROOT / "stage08b_output_manifest.json", {"created_utc": C.utc_now(), "files": rows, "status": "PASS"})


def finalize() -> None:
    rows = C.read_csv(RESULTS_CSV)
    if len(rows) != 4:
        raise RuntimeError("Expected four Stage08B F3 results")
    random_decision = select_random_lock(rows)
    C.atomic_json(C.ROOT / "random_lock_decision.json", random_decision)
    if random_decision["status"] != "PASS":
        raise RuntimeError(random_decision["status"])
    by_role = {row["candidate_role"]: row for row in rows}
    ap_selection = C.load_json(C.ROOT / "ap_only_selection.json", {})
    matched = C.load_json(C.ROOT / "matched_manual_generation.json", {})
    locks = {
        "created_utc": C.utc_now(),
        "matched_manual": {"label": "MatchedManual-B6 provisional locked architecture", "candidate_id": by_role["MatchedManual-B6"]["physical_candidate_id"], "architecture_signature": by_role["MatchedManual-B6"]["architecture_signature"], "checkpoint": by_role["MatchedManual-B6"]["best_checkpoint"], "checkpoint_sha256": by_role["MatchedManual-B6"]["best_checkpoint_sha256"], "f3_feasibility": by_role["MatchedManual-B6"]["f3_feasibility"]},
        "ap_only": {"label": "Meta-AP-only-B6 provisional locked architecture", "candidate_id": by_role["Meta-AP-only-B6 finalist"]["physical_candidate_id"], "architecture_signature": by_role["Meta-AP-only-B6 finalist"]["architecture_signature"], "checkpoint": by_role["Meta-AP-only-B6 finalist"]["best_checkpoint"], "checkpoint_sha256": by_role["Meta-AP-only-B6 finalist"]["best_checkpoint_sha256"], "f3_feasibility": by_role["Meta-AP-only-B6 finalist"]["f3_feasibility"], "aps_used_for_f2_selection": ap_selection["aps_used_for_selection"]},
        "random": random_decision,
        "recall_floor_f3": C.RECALL_FLOOR_F3,
        "matchedmanual_complexity_distance": matched["complexity_distance"],
        "status": "PASS",
    }
    C.atomic_json(C.ROOT / "provisional_control_locks.json", locks)
    checks = {
        "stage08a_approval_verified": C.load_json(STAGE08A / "stage08a_validation.json", {}).get("status") == "PASS",
        "amendment_A3_verified": C.load_json(A3 / "amendment_A3_validation.json", {}).get("status") == "PASS",
        "four_runs_complete": len(rows) == 4 and all(row["status"] == "COMPLETE" for row in rows),
        "execution_order_preserved": [row["candidate_role"] for row in rows] == ["MatchedManual-B6", "Meta-AP-only-B6 finalist", "Random candidate A", "Random candidate B"],
        "all_f3_feasible": all(row["f3_feasibility"] == "F3_FEASIBLE" for row in rows),
        "all_fresh_reconstructions": all(C.load_json(Path(row["reconstruction_manifest"]), {}).get("fresh_reconstruction_from_searchparent") is True for row in rows),
        "no_f1_f2_initialization": all(C.load_json(Path(row["reconstruction_manifest"]), {}).get("f1_f2_checkpoint_initialization") is False for row in rows),
        "recovery50_only": all(int(row["best_epoch_one_based"]) <= 50 for row in rows),
        "ap_only_aps_not_used": ap_selection.get("aps_used_for_selection") is False,
        "random_lock_automatic": random_decision["manual_visual_selection"] is False,
        "matchedmanual_standard_pass": Decimal(matched["complexity_distance"]) <= Decimal("0.01"),
        "kd_not_executed": all(str(row["kd_executed"]).lower() == "false" for row in rows),
        "teacher_not_loaded": all(str(row["teacher_loaded"]).lower() == "false" for row in rows),
        "official_val548_locked": True, "heldout1610_locked": True, "stage09_not_started": not (C.PROJECT / "09_final_architecture_lock").exists(),
    }
    failed = [key for key, value in checks.items() if not value]
    validation = {"created_utc": C.utc_now(), "checks": checks, "failed_checks": failed, "status": "PASS" if not failed else "FAIL"}
    C.atomic_json(C.ROOT / "stage08b_validation.json", validation)
    if failed:
        raise RuntimeError(f"Stage08B validation failed: {failed}")
    C.atomic_text(C.ROOT / "stage08b_report.md", "# Stage08B Near-Matched Controls, Recovery-50-only\n\n" f"- Status: `PASS`\n- P_meta: `{P_META}`\n- F_meta_raw: `{F_META}`\n" f"- MatchedManual: `{by_role['MatchedManual-B6']['physical_candidate_id']}`\n- MatchedManual distance: `{matched['complexity_distance']}`\n" f"- AP-only finalist: `{by_role['Meta-AP-only-B6 finalist']['physical_candidate_id']}`\n- APS used for AP-only selection: `NO`\n" f"- Random candidate A: `{by_role['Random candidate A']['physical_candidate_id']}`\n- Random candidate B: `{by_role['Random candidate B']['physical_candidate_id']}`\n" f"- Random provisional lock: `{random_decision['selected_candidate_id']}`\n- Random lock rule: `{random_decision['rule_clause']}`\n" "- F3 runs: `4/4`\n- KD executed: `NO`\n- Teacher loaded: `NO`\n- val548 accessed: `NO`\n- heldout1610 accessed: `NO`\n- Stage09 started: `NO`\n")
    C.atomic_json(FAILURE_AUDIT, C.load_json(FAILURE_AUDIT, {"created_utc": C.utc_now(), "failures": [], "status": "PASS"}))
    C.atomic_text(C.ROOT / ".complete", "STAGE08B_STATUS=PASS\nAPPROVAL_READY=YES\nSTAGE09_STARTED=NO\n")
    campaign = C.load_json(C.CAMPAIGN_STATE, {})
    campaign.update({"state": "COMPLETE", "completed_utc": C.utc_now(), "approval_ready": True, "stage09_started": False})
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    update_worker(run_state="COMPLETE", child_pid=None)
    build_output_manifest()


def pause() -> int:
    state = C.load_json(C.CAMPAIGN_STATE, {})
    state.update({"state": "PAUSED_AFTER_ATOMIC_CANDIDATE_COMMIT", "paused_utc": C.utc_now(), "safe_to_shut_down_laptop": True, "approval_ready": False})
    C.atomic_json(C.CAMPAIGN_STATE, state)
    update_worker(run_state="PAUSED_SAFE_TO_SHUT_DOWN", child_pid=None)
    if C.PAUSE_REQUEST.exists():
        C.PAUSE_REQUEST.unlink()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    C.ensure_directories()
    configure_runtime()
    global CAMPAIGN_UUID
    existing = C.load_json(C.CAMPAIGN_STATE, {}) or {}
    CAMPAIGN_UUID = existing.get("campaign_uuid") or str(uuid.uuid4())
    acquire_lock()
    try:
        candidates, groups, complexity, taylor_bank = preflight()
        if args.preflight_only:
            return 0
        state = {"campaign_uuid": CAMPAIGN_UUID, "created_utc": existing.get("created_utc", C.utc_now()), "started_utc": existing.get("started_utc", C.utc_now()), "state": "RUNNING", "approval_token_consumed": "APPROVE_STAGE08A_PROPOSED_LOCK", "protocol": "RECOVERY50_ONLY", "candidate_selection_status": "FROZEN", "f3_runs_expected": 4, "kd_executed": False, "teacher_loaded": False, "official_val548_accessed": False, "heldout1610_accessed": False, "stage09_started": False, "approval_ready": False}
        C.atomic_json(C.CAMPAIGN_STATE, state)
        ACTIVE.update({"worker_pid": os.getpid(), "campaign_uuid": CAMPAIGN_UUID})
        update_worker(run_state="PREFLIGHT_PASS")
        disk_sample("before_stage08b_or_resume")
        for candidate in candidates:
            if (run_dir(candidate) / ".complete").is_file():
                continue
            if C.PAUSE_REQUEST.is_file():
                return pause()
            commit_candidate(candidate, groups, complexity, taylor_bank)
            if C.PAUSE_REQUEST.is_file():
                return pause()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        finalize()
        return 0
    except Exception as error:
        audit = C.load_json(FAILURE_AUDIT, {"created_utc": C.utc_now(), "failures": []})
        audit["failures"].append({"timestamp_utc": C.utc_now(), "error": repr(error), "traceback": traceback.format_exc(), "worker_state": dict(ACTIVE)})
        audit["status"] = "BLOCKED"
        C.atomic_json(FAILURE_AUDIT, audit)
        state = C.load_json(C.CAMPAIGN_STATE, {}) or {"campaign_uuid": CAMPAIGN_UUID}
        state.update({"state": "BLOCKED", "blocked_utc": C.utc_now(), "approval_ready": False})
        C.atomic_json(C.CAMPAIGN_STATE, state)
        update_worker(run_state="BLOCKED", error=repr(error), child_pid=None)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    finally:
        release_lock()
        C.write_status()


if __name__ == "__main__":
    raise SystemExit(main())
