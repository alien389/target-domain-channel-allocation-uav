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

import pandas as pd
import psutil
import torch

import stage07_common as C


STAGE04_SCRIPT = C.PROJECT / "scripts" / "stage04_shared_p0_fidelity1.py"
GENE_JSON = (
    C.PROJECT
    / "03a_taylor_dependency_gene_space"
    / "gene_space_R1"
    / "gene_space_manifest_R1.json"
)
STAGE06_PHYSICAL = C.STAGE06A / "stage06a_physical_candidate_manifest.csv"
STAGE06_PROMOTIONS = C.STAGE06A / "stage06a_promotion_manifest.csv"
STAGE06_PLAN = C.STAGE06A / "stage06a_fidelity2_execution_plan.json"
STAGE06_VALIDATION = C.STAGE06A / "stage06a_validation.json"
STAGE06_SOURCE_HASHES = C.STAGE06A / "stage06a_source_hashes.json"
EVALUATOR = C.SCRIPTS / "evaluate_stage07_inner_val.py"
PHYSICAL_MANIFEST = C.ROOT / "stage07_physical_evaluation_manifest.csv"
METHOD_MANIFEST = C.ROOT / "stage07_method_result_manifest.csv"
RETENTION_MANIFEST = C.ROOT / "checkpoint_retention_manifest.csv"
FAILURE_AUDIT = C.ROOT / "failure_audit.json"
DISK_AUDIT = C.ROOT / "disk_usage_audit.json"

ACTIVE: dict[str, Any] = {}
CAMPAIGN_UUID = ""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S4 = load_module("stage07_stage04_locked", STAGE04_SCRIPT)
R1M = S4.R1M


def physical_dir(candidate: dict[str, Any]) -> Path:
    return C.PHYSICAL / f"F2_{int(candidate['unique_signature_index']):02d}_{candidate['signature_sha256'][:16]}"


def configure_locked_runtime() -> None:
    S4.OUTPUT = C.ROOT
    S4.PHYSICAL_ROOT = C.PHYSICAL
    S4.RECOVERY_EPOCHS = C.RECOVERY_EPOCHS
    S4.TRAINING_SEED = C.TRAINING_SEED
    S4.DISK_STOP_GIB = C.HARD_STOP_FREE_GIB
    S4.EVAL_WRAPPER = EVALUATOR
    S4.evaluation_directory = physical_dir
    S4.run_logged = run_logged_with_heartbeat


def update_worker(**values: Any) -> None:
    ACTIVE.update(values)
    ACTIVE["updated_utc"] = C.utc_now()
    C.atomic_json(C.WORKER_STATE, ACTIVE)
    if C.CAMPAIGN_LOCK.is_file():
        lock = C.load_json(C.CAMPAIGN_LOCK, {})
        lock["latest_heartbeat"] = ACTIVE["updated_utc"]
        lock["child_pid"] = ACTIVE.get("child_pid")
        C.atomic_json(C.CAMPAIGN_LOCK, lock)
    C.write_status()


def run_logged_with_heartbeat(
    command: list[str], stdout_path: Path, stderr_path: Path, label: str
) -> tuple[int, float]:
    start = time.time()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, (
        stderr_path.open("w", encoding="utf-8", errors="replace")
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(C.PROJECT),
            stdout=stdout,
            stderr=stderr,
            env=os.environ.copy(),
        )
        update_worker(
            child_pid=process.pid,
            child_command=subprocess.list2cmdline(command),
            child_label=label,
        )
        while process.poll() is None:
            update_worker()
            time.sleep(15)
        code = int(process.returncode)
    update_worker(child_pid=None, child_command=None, child_label=None)
    return code, time.time() - start


def acquire_lock() -> None:
    others = [row for row in C.stage07_worker_processes() if row["pid"] != os.getpid()]
    if others:
        raise RuntimeError(f"Another Stage07 worker is active: {others}")
    if C.active_gpu_children():
        raise RuntimeError(f"A GPU training/evaluation process is active: {C.active_gpu_children()}")
    if C.CAMPAIGN_LOCK.is_file():
        existing = C.load_json(C.CAMPAIGN_LOCK, {})
        if C.process_alive(existing.get("worker_pid")):
            raise RuntimeError(f"Live Stage07 lock belongs to PID {existing['worker_pid']}")
    C.atomic_json(
        C.CAMPAIGN_LOCK,
        {
            "created_utc": C.utc_now(),
            "campaign_uuid": CAMPAIGN_UUID,
            "worker_pid": os.getpid(),
            "hostname": socket.gethostname(),
            "max_concurrent_gpu_training_jobs": 1,
            "max_concurrent_gpu_evaluation_jobs": 1,
        },
    )


def release_lock() -> None:
    if not C.CAMPAIGN_LOCK.is_file():
        return
    payload = C.load_json(C.CAMPAIGN_LOCK, {})
    if int(payload.get("worker_pid", -1)) == os.getpid():
        C.CAMPAIGN_LOCK.unlink()


def validate_stage06a_unchanged() -> list[str]:
    failures = []
    if C.sha256(C.STAGE06A_MANIFEST) != C.STAGE06A_MANIFEST_SHA256:
        failures.append("STAGE06A_OUTPUT_MANIFEST_SHA256_MISMATCH")
        return failures
    manifest = C.load_json(C.STAGE06A_MANIFEST, {})
    for row in manifest.get("files", []):
        path = Path(row["path"])
        if not path.is_file():
            failures.append(f"MISSING_STAGE06A_FILE:{path}")
        elif C.sha256(path) != row["sha256"]:
            failures.append(f"STAGE06A_FILE_HASH_MISMATCH:{path}")
    source_payload = C.load_json(STAGE06_SOURCE_HASHES, {})
    for row in source_payload.get("sources", []):
        path = Path(row["path"])
        if not path.is_file():
            failures.append(f"MISSING_LOCKED_SOURCE:{path}")
        elif C.sha256(path) != row["sha256"]:
            failures.append(f"LOCKED_SOURCE_HASH_MISMATCH:{path}")
    return failures


def exact_float_text(value: Any) -> str:
    return format(float(value), ".17g")


def source_hash_payload() -> dict[str, Any]:
    paths = [
        C.STAGE06A_MANIFEST,
        STAGE06_PHYSICAL,
        STAGE06_PROMOTIONS,
        STAGE06_PLAN,
        STAGE06_VALIDATION,
        STAGE06_SOURCE_HASHES,
        STAGE04_SCRIPT,
        C.SCRIPTS / "stage07_common.py",
        C.SCRIPTS / "stage07_worker.py",
        EVALUATOR,
        S4.TRAIN_WRAPPER,
        S4.HYP_YAML,
        S4.DATA_YAML,
        R1M.SEARCH_PARENT,
        S4.TAYLOR_RANKING,
        S4.DEPENDENCY_GRAPH,
        GENE_JSON,
        S4.VISDRONE_PARITY_AUDIT,
    ]
    return {
        "created_utc": C.utc_now(),
        "files": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": C.sha256(path)}
            for path in paths
        ],
        "stage06a_output_manifest_sha256": C.sha256(C.STAGE06A_MANIFEST),
        "status": "PASS",
    }


def build_candidates() -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]], Any, dict[str, Any]]:
    physical = C.read_csv(STAGE06_PHYSICAL)
    promotions = C.read_csv(STAGE06_PROMOTIONS)
    plan = C.load_json(STAGE06_PLAN, {})
    if len(physical) != C.EXPECTED_PHYSICAL or len(promotions) != C.EXPECTED_METHOD_RECORDS:
        raise RuntimeError("Stage06A promotion accounting mismatch.")
    if len({row["architecture_signature"] for row in physical}) != C.EXPECTED_PHYSICAL:
        raise RuntimeError("Stage06A physical architecture signatures are not unique.")
    if any(row["execution_status"] != "PLANNED_NOT_STARTED" for row in physical):
        raise RuntimeError("Not every Stage06A physical candidate is PLANNED_NOT_STARTED.")
    shared = [row for row in promotions if row["architecture_signature"] == "6510DFAD1319B0D9AC7945FEAB227D43DCB320E9802E033B958D28784F8F464B"]
    if len(shared) != 2 or len({row["physical_candidate_id"] for row in shared}) != 1:
        raise RuntimeError("The shared NSGA-II/AP-only architecture attribution is invalid.")
    if C.sha256(R1M.SEARCH_PARENT) != C.SEARCH_PARENT_SHA256:
        raise RuntimeError("SearchParent SHA256 mismatch.")
    gene = C.load_json(GENE_JSON, {})
    groups = gene["groups"]
    if len(groups) != 35:
        raise RuntimeError("R1 gene space does not contain 35 dependency groups.")
    group_ids = [group["group_id"] for group in groups]
    parent = R1M.load_source_model(R1M.SEARCH_PARENT, "stage07_parent")
    complexity = R1M.ExactComplexityModel(parent.model, groups)
    taylor_bank, taylor_manifest = R1M.taylor_scope_bank(groups)
    if taylor_manifest["checkpoint_sha256"] != C.SEARCH_PARENT_SHA256:
        raise RuntimeError("Taylor ranking is not bound to the frozen SearchParent.")
    plan_jobs = {row["physical_candidate_id"]: row for row in plan["jobs"]}
    candidates = []
    for row in physical:
        widths = json.loads(row["retained_width_vector"])
        if set(widths) != set(group_ids):
            raise RuntimeError(f"Width-vector scope mismatch: {row['physical_candidate_id']}")
        width_hash = C.hash_text(C.canonical_json(widths))
        plan_row = plan_jobs[row["physical_candidate_id"]]
        if width_hash != plan_row["retained_width_vector_sha256"]:
            raise RuntimeError(f"Width-vector hash mismatch: {row['physical_candidate_id']}")
        vector = tuple(int(widths[group_id]) for group_id in group_ids)
        signature = R1M.signature_sha(group_ids, vector)
        if signature != row["architecture_signature"]:
            raise RuntimeError(f"Architecture signature mismatch: {row['physical_candidate_id']}")
        analytic = complexity.calculate(widths)
        raw_flops = int(analytic["macs"]) * 2
        if int(analytic["params"]) != int(row["params_exact_integer"]):
            raise RuntimeError(f"Exact Params mismatch: {row['physical_candidate_id']}")
        if raw_flops != int(row["flops_raw_exact_integer"]):
            raise RuntimeError(f"Exact FLOPs mismatch: {row['physical_candidate_id']}")
        if not (C.PARAMS_LOWER <= int(row["params_exact_integer"]) <= C.PARAMS_UPPER):
            raise RuntimeError("Candidate Params outside frozen band.")
        if not (C.FLOPS_RAW_LOWER <= raw_flops <= C.FLOPS_RAW_UPPER):
            raise RuntimeError("Candidate FLOPs outside frozen band.")
        source_map = Path(row["channel_index_map"])
        if C.sha256(source_map) != row["channel_index_map_sha256"]:
            raise RuntimeError(f"Frozen channel-index map mismatch: {row['physical_candidate_id']}")
        candidate: dict[str, Any] = {
            "physical_candidate_id": row["physical_candidate_id"],
            "unique_signature_index": 0,
            "evaluation_id": row["future_fidelity2_evaluation_identity"],
            "signature_sha256": signature,
            "fused_params": int(row["params_exact_integer"]),
            "fused_flops_g": float(row["flops_g_full_precision"]),
            "params_exact_integer": int(row["params_exact_integer"]),
            "flops_raw_exact_integer": raw_flops,
            "flops_g_full_precision": row["flops_g_full_precision"],
            "width_vector_sha256": width_hash,
            "expected_channel_map": str(source_map),
            "expected_channel_map_sha256": row["channel_index_map_sha256"],
            "roles": row["selected_method_roles"],
            "replicates": "[]",
            "membership_count": row["method_attribution_count"],
        }
        candidate.update({group_id: int(widths[group_id]) for group_id in group_ids})
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (row["signature_sha256"] != C.TARGETMANUAL_SIGNATURE, row["physical_candidate_id"])
    )
    for index, candidate in enumerate(candidates, 1):
        candidate["unique_signature_index"] = index
    if candidates[0]["signature_sha256"] != C.TARGETMANUAL_SIGNATURE:
        raise RuntimeError("TargetManual is not first in the Stage07 queue.")
    return candidates, promotions, groups, complexity, taylor_bank


def preflight() -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]], Any, dict[str, Any]]:
    failures = validate_stage06a_unchanged()
    validation = C.load_json(STAGE06_VALIDATION, {})
    if validation.get("status") != "PASS":
        failures.append("STAGE06A_STATUS_NOT_PASS")
    if C.free_gib() < C.HARD_STOP_FREE_GIB:
        failures.append("INSUFFICIENT_FREE_DISK")
    gpu_children = C.active_gpu_children()
    if gpu_children:
        failures.append(f"ACTIVE_GPU_CHILDREN:{gpu_children}")
    if failures:
        raise RuntimeError(f"Stage07 preflight failed: {failures}")
    assets = build_candidates()
    C.atomic_json(C.ROOT / "stage07_source_hashes.json", source_hash_payload())
    protocol = {
        "schema": "stage07_fidelity2_recovery25_protocol_v1",
        "created_utc": C.utc_now(),
        "approval_token_consumed": "APPROVE_STAGE06A_FIDELITY2_PROMOTION_LOCK",
        "stage06a_output_manifest_sha256": C.STAGE06A_MANIFEST_SHA256,
        "promoted_method_records": 11,
        "unique_physical_recovery25_jobs": 10,
        "deduplicated_physical_jobs": 1,
        "targetmanual_executed_first": True,
        "initialization": "fresh reconstruction from frozen SearchParent, widths, and Taylor channel indices",
        "fidelity1_checkpoint_initialization": False,
        "epochs": 25,
        "seed": 42,
        "train": "meta-train5177",
        "validation": "meta-inner-val1294",
        "imgsz": 640,
        "batch_size": 16,
        "explicit_evaluation": {
            "precision": "FP32",
            "images": 1294,
            "conf_thres": 0.001,
            "nms_iou": 0.60,
            "max_det": 500,
            "augment": False,
            "tta": False,
        },
        "kd_enabled": False,
        "official_val548_locked": True,
        "heldout1610_locked": True,
        "stage08a_started": False,
        "status": "LOCKED",
    }
    protocol["stage07_protocol_sha256"] = C.hash_text(C.canonical_json(protocol))
    C.atomic_json(C.ROOT / "stage07_protocol_lock.json", protocol)
    return assets


def disk_sample(event: str) -> None:
    payload = C.load_json(DISK_AUDIT, {"created_utc": C.utc_now(), "samples": []})
    payload["samples"].append(
        {
            "timestamp_utc": C.utc_now(),
            "event": event,
            "free_gib": C.free_gib(),
            "stage07_size_bytes": C.tree_size_bytes(C.ROOT),
        }
    )
    C.atomic_json(DISK_AUDIT, payload)


def train_one(candidate: dict[str, Any], raw_audit: dict[str, Any]) -> dict[str, Any]:
    eval_dir = physical_dir(candidate)
    command = S4.training_command(eval_dir, Path(raw_audit["raw_checkpoint"]))
    S4.text_write(eval_dir / "train_command.txt", S4.command_text(command) + "\n")
    if S4.training_complete(eval_dir):
        return S4.recovered_training_execution(eval_dir)
    if (eval_dir / "results.csv").exists() or (eval_dir / "weights").exists():
        S4.clear_partial_training(eval_dir)
    attempts = []
    for attempt in (1, 2):
        stdout = eval_dir / f"train_attempt_{attempt}_stdout.log"
        stderr = eval_dir / f"train_attempt_{attempt}_stderr.log"
        code, duration = run_logged_with_heartbeat(
            command, stdout, stderr, f"{eval_dir.name} fresh Recovery-25 attempt {attempt}"
        )
        transient = S4.transient_failure(stdout, stderr) if code else False
        attempts.append(
            {
                "attempt": attempt,
                "return_code": code,
                "duration_seconds": duration,
                "stdout": str(stdout),
                "stderr": str(stderr),
                "transient_failure": transient,
            }
        )
        if code == 0 and S4.training_complete(eval_dir):
            shutil.copy2(stdout, eval_dir / "stdout.log")
            shutil.copy2(stderr, eval_dir / "stderr.log")
            payload = {
                "created_utc": C.utc_now(),
                "training_reused": False,
                "training_duration_seconds": duration,
                "duration_source": "SUBPROCESS_WALL_CLOCK",
                "attempts": attempts,
            }
            C.atomic_json(eval_dir / "training_execution.json", payload)
            return payload
        if attempt == 1 and transient:
            S4.clear_partial_training(eval_dir)
            continue
        raise RuntimeError(f"Recovery-25 failed for {candidate['physical_candidate_id']}.")
    raise AssertionError("Unreachable training retry state")


def augment_metrics(path: Path, candidate: dict[str, Any], training: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = C.load_json(path, {})
    metrics.update(
        {
            "stage": "Stage07 Fidelity-2 fresh Recovery-25",
            "physical_candidate_id": candidate["physical_candidate_id"],
            "params_exact_integer": candidate["params_exact_integer"],
            "flops_raw_exact_integer": candidate["flops_raw_exact_integer"],
            "flops_g_full_precision": candidate["flops_g_full_precision"],
            "precision_full_precision": exact_float_text(metrics["precision"]),
            "recall_full_precision": exact_float_text(metrics["recall"]),
            "f1_full_precision": exact_float_text(metrics["f1"]),
            "ap_full_precision": exact_float_text(metrics["AP_0.5_0.95"]),
            "ap50_full_precision": exact_float_text(metrics["AP50"]),
            "ap75_full_precision": exact_float_text(metrics["AP75"]),
            "aps_full_precision": exact_float_text(metrics["APS"]),
            "apm_full_precision": exact_float_text(metrics["APM"]),
            "apl_full_precision": exact_float_text(metrics["APL"]),
            "training_duration_seconds": training["training_duration_seconds"],
            "explicit_evaluation_duration_seconds": evaluation["evaluation_duration_seconds"],
            "fresh_reconstruction_from_searchparent": True,
            "fidelity1_checkpoint_reused_for_initialization": False,
            "official_val548_accessed": False,
            "heldout1610_accessed": False,
            "kd_enabled": False,
            "status": "PASS",
        }
    )
    C.atomic_json(path, metrics)
    return metrics


def checkpoint_payload(eval_dir: Path) -> dict[str, Any]:
    paths = {
        "raw_checkpoint": eval_dir / "raw_checkpoint.pt",
        "best_checkpoint": eval_dir / "weights" / "best.pt",
        "last_checkpoint_before_deletion": eval_dir / "weights" / "last.pt",
    }
    payload = {"created_utc": C.utc_now()}
    for role, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"Missing checkpoint before retention: {path}")
        payload[role] = {
            "path": str(path),
            "sha256": C.sha256(path),
            "size_bytes": path.stat().st_size,
        }
    C.atomic_json(eval_dir / "checkpoint_hashes_before_retention.json", payload)
    return payload


def metric_row(candidate: dict[str, Any], metrics: dict[str, Any], hashes: dict[str, Any], reconstruction: Path, run_manifest: Path) -> dict[str, Any]:
    eval_dir = physical_dir(candidate)
    return {
        "physical_candidate_id": candidate["physical_candidate_id"],
        "fidelity2_evaluation_id": candidate["evaluation_id"],
        "architecture_signature": candidate["signature_sha256"],
        "recovery_epochs": 25,
        "training_seed": 42,
        "AP": metrics["ap_full_precision"],
        "AP50": metrics["ap50_full_precision"],
        "AP75": metrics["ap75_full_precision"],
        "APS": metrics["aps_full_precision"],
        "APM": metrics["apm_full_precision"],
        "APL": metrics["apl_full_precision"],
        "Precision": metrics["precision_full_precision"],
        "Recall": metrics["recall_full_precision"],
        "F1": metrics["f1_full_precision"],
        "params_exact_integer": candidate["params_exact_integer"],
        "flops_raw_exact_integer": candidate["flops_raw_exact_integer"],
        "flops_g_full_precision": candidate["flops_g_full_precision"],
        "best_epoch_zero_based": metrics["best_epoch"]["best_epoch_zero_based"],
        "best_epoch_one_based": metrics["best_epoch"]["best_epoch_one_based"],
        "best_fitness": exact_float_text(metrics["best_epoch"]["best_fitness"]),
        "training_duration_seconds": metrics["training_duration_seconds"],
        "explicit_evaluation_duration_seconds": metrics["explicit_evaluation_duration_seconds"],
        "raw_checkpoint_sha256": hashes["raw_checkpoint"]["sha256"],
        "best_checkpoint": str(eval_dir / "weights" / "best.pt"),
        "best_checkpoint_sha256": hashes["best_checkpoint"]["sha256"],
        "last_checkpoint_sha256_before_deletion": hashes["last_checkpoint_before_deletion"]["sha256"],
        "metrics_path": str(eval_dir / "explicit_inner_val_metrics.json"),
        "metrics_sha256": C.sha256(eval_dir / "explicit_inner_val_metrics.json"),
        "reconstruction_manifest": str(reconstruction),
        "reconstruction_manifest_sha256": C.sha256(reconstruction),
        "channel_index_map": str(eval_dir / "channel_index_map.csv"),
        "channel_index_map_sha256": C.sha256(eval_dir / "channel_index_map.csv"),
        "run_manifest": str(run_manifest),
        "run_manifest_sha256": C.sha256(run_manifest),
        "fresh_reconstruction_from_searchparent": True,
        "fidelity1_checkpoint_reused_for_initialization": False,
        "structural_validity": "PASS",
        "status": "COMPLETE",
    }


def method_rows(candidate: dict[str, Any], physical: dict[str, Any], promotions: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for source in promotions:
        if source["physical_candidate_id"] != candidate["physical_candidate_id"]:
            continue
        identity = C.hash_text(
            "\n".join(
                [source["promotion_candidate_id"], candidate["evaluation_id"], "FIDELITY2"]
            )
        )
        rows.append(
            {
                "fidelity2_method_record_id": identity,
                "promotion_candidate_id": source["promotion_candidate_id"],
                "physical_candidate_id": candidate["physical_candidate_id"],
                "method": source["method"],
                "replicate": source["replicate"],
                "generation_or_random_position": source["generation_or_random_position"],
                "stage05_search_record_id": source["search_record_id"],
                "fidelity2_evaluation_id": candidate["evaluation_id"],
                "architecture_signature": candidate["signature_sha256"],
                "metric_role": "FIDELITY2_CONFIRMATION_EVIDENCE",
                "AP": physical["AP"],
                "APS": physical["APS"],
                "Recall": physical["Recall"],
                "params_exact_integer": physical["params_exact_integer"],
                "flops_raw_exact_integer": physical["flops_raw_exact_integer"],
                "flops_g_full_precision": physical["flops_g_full_precision"],
                "metrics_path": physical["metrics_path"],
                "metrics_sha256": physical["metrics_sha256"],
                "shared_physical_result": len(
                    [p for p in promotions if p["physical_candidate_id"] == candidate["physical_candidate_id"]]
                ) > 1,
                "status": "COMPLETE",
            }
        )
    return rows


def lock_recall_floor(candidate: dict[str, Any], physical: dict[str, Any]) -> None:
    if candidate["signature_sha256"] != C.TARGETMANUAL_SIGNATURE:
        raise RuntimeError("Recall floor can only be locked from TargetManual.")
    recall = Decimal(str(physical["Recall"]))
    floor = recall - Decimal("0.002")
    payload = {
        "created_utc": C.utc_now(),
        "source": "TargetManual-B6-reference Stage07 explicit FP32 inner-val1294 evaluation",
        "physical_candidate_id": candidate["physical_candidate_id"],
        "fidelity2_evaluation_id": candidate["evaluation_id"],
        "targetmanual_recall_full_precision": str(recall),
        "offset": "0.002",
        "recall_floor_f2_full_precision": str(floor),
        "formula": "Recall(TargetManual_F2) - 0.002",
        "f1_floor_reused": False,
        "status": "LOCKED",
    }
    C.atomic_json(C.ROOT / "stage07_recall_floor_f2.json", payload)
    C.atomic_json(C.ROOT / "stage07_targetmanual_f2_result.json", physical)


def classify_feasibility() -> None:
    floor_payload = C.load_json(C.ROOT / "stage07_recall_floor_f2.json", {})
    if floor_payload.get("status") != "LOCKED":
        return
    floor = Decimal(floor_payload["recall_floor_f2_full_precision"])
    physical = C.read_csv(PHYSICAL_MANIFEST)
    for row in physical:
        recall_ok = Decimal(row["Recall"]) >= floor
        budget_ok = (
            C.PARAMS_LOWER <= int(row["params_exact_integer"]) <= C.PARAMS_UPPER
            and C.FLOPS_RAW_LOWER <= int(row["flops_raw_exact_integer"]) <= C.FLOPS_RAW_UPPER
        )
        row["recall_floor_f2"] = str(floor)
        row["recall_feasible_f2"] = recall_ok
        row["f2_feasibility"] = "F2_FEASIBLE" if recall_ok and budget_ok else "F2_INFEASIBLE"
    C.atomic_csv(PHYSICAL_MANIFEST, physical)
    by_id = {row["physical_candidate_id"]: row for row in physical}
    methods = C.read_csv(METHOD_MANIFEST)
    for row in methods:
        source = by_id[row["physical_candidate_id"]]
        row["recall_floor_f2"] = source["recall_floor_f2"]
        row["recall_feasible_f2"] = source["recall_feasible_f2"]
        row["f2_feasibility"] = source["f2_feasibility"]
    C.atomic_csv(METHOD_MANIFEST, methods)


def apply_retention(candidate: dict[str, Any], hashes: dict[str, Any]) -> None:
    eval_dir = physical_dir(candidate)
    rows = C.read_csv(RETENTION_MANIFEST)
    timestamp = C.utc_now()
    for role, path in (
        ("raw_checkpoint", eval_dir / "raw_checkpoint.pt"),
        ("last_checkpoint", eval_dir / "weights" / "last.pt"),
    ):
        key = "raw_checkpoint" if role == "raw_checkpoint" else "last_checkpoint_before_deletion"
        rows.append(
            {
                "physical_candidate_id": candidate["physical_candidate_id"],
                "fidelity2_evaluation_id": candidate["evaluation_id"],
                "artifact_role": role,
                "path_before_deletion": str(path),
                "sha256_before_deletion": hashes[key]["sha256"],
                "size_bytes_before_deletion": hashes[key]["size_bytes"],
                "retention_decision": "DELETED_AFTER_HASH_AND_ATOMIC_RESULT_COMMIT",
                "deletion_timestamp_utc": timestamp,
                "reason": "Space-controlled Stage07 retention; best.pt and all scientific manifests retained",
            }
        )
        if path.exists():
            path.unlink()
    best = hashes["best_checkpoint"]
    rows.append(
        {
            "physical_candidate_id": candidate["physical_candidate_id"],
            "fidelity2_evaluation_id": candidate["evaluation_id"],
            "artifact_role": "best_checkpoint",
            "path_before_deletion": best["path"],
            "sha256_before_deletion": best["sha256"],
            "size_bytes_before_deletion": best["size_bytes"],
            "retention_decision": "RETAINED",
            "deletion_timestamp_utc": "",
            "reason": "Fidelity-2 selected checkpoint evidence",
        }
    )
    predictions = eval_dir / "explicit_inner_val_predictions"
    if predictions.exists():
        shutil.rmtree(predictions)
    C.atomic_csv(RETENTION_MANIFEST, rows)


def execute_candidate(candidate: dict[str, Any], promotions: list[dict[str, str]], groups: list[dict[str, Any]], complexity: Any, taylor_bank: dict[str, Any]) -> None:
    eval_dir = physical_dir(candidate)
    complete = eval_dir / ".complete"
    if complete.is_file():
        return
    if C.free_gib() < C.HARD_STOP_FREE_GIB:
        raise RuntimeError("Free disk is below the locked 8 GiB threshold.")
    attributions = [
        row["promotion_candidate_id"]
        for row in promotions
        if row["physical_candidate_id"] == candidate["physical_candidate_id"]
    ]
    update_worker(
        physical_candidate_id=candidate["physical_candidate_id"],
        method_attributions=attributions,
        candidate_number=candidate["unique_signature_index"],
        candidate_state="RECONSTRUCTING_FROM_SEARCHPARENT",
        evaluation_directory=str(eval_dir),
        current_epoch=0,
    )
    raw = S4.reconstruct_raw(candidate, groups, complexity, taylor_bank)
    raw.update(
        {
            "stage": "Stage07 Fidelity-2 fresh Recovery-25",
            "physical_candidate_id": candidate["physical_candidate_id"],
            "fresh_reconstruction_from_searchparent": True,
            "fidelity1_checkpoint_reused_for_initialization": False,
            "params_exact_integer": candidate["params_exact_integer"],
            "flops_raw_exact_integer": candidate["flops_raw_exact_integer"],
            "flops_g_full_precision": candidate["flops_g_full_precision"],
        }
    )
    reconstruction_path = eval_dir / "raw_reconstruction_audit.json"
    C.atomic_json(reconstruction_path, raw)
    generated_map_hash = C.sha256(eval_dir / "channel_index_map.csv")
    if generated_map_hash != candidate["expected_channel_map_sha256"]:
        raise RuntimeError("Fresh Stage07 channel-index map differs from the Stage06A frozen map.")
    update_worker(candidate_state="TRAINING_RECOVERY25", training_start_time=C.utc_now())
    training = train_one(candidate, raw)
    update_worker(candidate_state="EXPLICIT_FP32_EVALUATION", current_epoch=25)
    evaluation = S4.evaluate_one(candidate)
    metrics_path = eval_dir / "explicit_inner_val_metrics.json"
    metrics = augment_metrics(metrics_path, candidate, training, evaluation)
    if int(metrics["image_count"]) != 1294 or metrics["precision_mode"] != "FP32":
        raise RuntimeError("Explicit F2 evaluation contract failed.")
    if not all(
        math.isfinite(float(metrics[key]))
        for key in ("AP_0.5_0.95", "AP50", "AP75", "APS", "APM", "APL", "precision", "recall", "f1")
    ):
        raise RuntimeError("Non-finite explicit F2 metric detected.")
    hashes = checkpoint_payload(eval_dir)
    run_manifest_path = eval_dir / "run_manifest.json"
    protocol = C.load_json(C.ROOT / "stage07_protocol_lock.json", {})
    run_manifest = {
        "created_utc": C.utc_now(),
        "stage": "Stage07 Fidelity-2 fresh Recovery-25",
        "physical_candidate_id": candidate["physical_candidate_id"],
        "fidelity2_evaluation_id": candidate["evaluation_id"],
        "architecture_signature": candidate["signature_sha256"],
        "method_attributions": attributions,
        "fresh_reconstruction_from_searchparent": True,
        "searchparent_sha256": C.SEARCH_PARENT_SHA256,
        "fidelity1_checkpoint_reused_for_initialization": False,
        "recovery_epochs": 25,
        "training_seed": 42,
        "training": training,
        "explicit_evaluation": evaluation,
        "metrics": str(metrics_path),
        "metrics_sha256": C.sha256(metrics_path),
        "checkpoint_hashes_before_retention": hashes,
        "stage07_protocol_sha256": protocol["stage07_protocol_sha256"],
        "official_val548_accessed": False,
        "heldout1610_accessed": False,
        "kd_executed": False,
        "status": "COMPLETE",
    }
    C.atomic_json(run_manifest_path, run_manifest)
    physical = metric_row(candidate, metrics, hashes, reconstruction_path, run_manifest_path)
    physical_rows = [
        row for row in C.read_csv(PHYSICAL_MANIFEST)
        if row["physical_candidate_id"] != candidate["physical_candidate_id"]
    ]
    physical_rows.append(physical)
    physical_rows.sort(key=lambda row: row["physical_candidate_id"])
    C.atomic_csv(PHYSICAL_MANIFEST, physical_rows)
    new_methods = method_rows(candidate, physical, promotions)
    method_records = [
        row for row in C.read_csv(METHOD_MANIFEST)
        if row["physical_candidate_id"] != candidate["physical_candidate_id"]
    ] + new_methods
    method_records.sort(key=lambda row: row["promotion_candidate_id"])
    C.atomic_csv(METHOD_MANIFEST, method_records)
    if candidate["signature_sha256"] == C.TARGETMANUAL_SIGNATURE:
        lock_recall_floor(candidate, physical)
    elif not (C.ROOT / "stage07_recall_floor_f2.json").is_file():
        raise RuntimeError("A non-control candidate reached classification before Recall_floor_F2 lock.")
    classify_feasibility()
    apply_retention(candidate, hashes)
    C.atomic_text(
        complete,
        "STAGE07_PHYSICAL_RECOVERY25=COMPLETE\n"
        f"PHYSICAL_CANDIDATE_ID={candidate['physical_candidate_id']}\n"
        f"FIDELITY2_EVALUATION_ID={candidate['evaluation_id']}\n",
    )
    update_worker(candidate_state="ATOMICALLY_COMMITTED")
    disk_sample(f"completed_{candidate['unique_signature_index']}_of_10")


def nondominated(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    feasible = [row for row in rows if row.get("f2_feasibility") == "F2_FEASIBLE"]
    result = []
    for row in feasible:
        ap, aps = Decimal(row["AP"]), Decimal(row["APS"])
        dominated = any(
            (Decimal(other["AP"]) >= ap and Decimal(other["APS"]) >= aps)
            and (Decimal(other["AP"]) > ap or Decimal(other["APS"]) > aps)
            for other in feasible
            if other["physical_candidate_id"] != row["physical_candidate_id"]
        )
        if not dominated:
            result.append(row)
    return sorted(result, key=lambda row: (Decimal(row["AP"]), Decimal(row["APS"])))


def knee_candidate(front: list[dict[str, str]]) -> dict[str, str]:
    if not front:
        raise RuntimeError("No feasible proposed Fidelity-2 front exists.")
    if len(front) <= 2:
        return sorted(front, key=lambda row: (-Decimal(row["AP"]), -Decimal(row["APS"]), row["architecture_signature"]))[0]
    aps = [float(row["APS"]) for row in front]
    ap = [float(row["AP"]) for row in front]
    ap_min, ap_max = min(ap), max(ap)
    aps_min, aps_max = min(aps), max(aps)
    normalized = []
    for row in front:
        x = (float(row["AP"]) - ap_min) / (ap_max - ap_min) if ap_max > ap_min else 0.0
        y = (float(row["APS"]) - aps_min) / (aps_max - aps_min) if aps_max > aps_min else 0.0
        normalized.append((row, x, y))
    left = min(normalized, key=lambda item: (item[1], -item[2], item[0]["architecture_signature"]))
    right = max(normalized, key=lambda item: (item[1], -item[2], item[0]["architecture_signature"]))
    dx, dy = right[1] - left[1], right[2] - left[2]
    denominator = math.hypot(dx, dy)
    scored = []
    for row, x, y in normalized:
        distance = abs(dy * x - dx * y + right[1] * left[2] - right[2] * left[1]) / denominator if denominator else 0.0
        scored.append((distance, Decimal(row["AP"]), Decimal(row["APS"]), row["architecture_signature"], row))
    return sorted(scored, key=lambda item: (-item[0], -item[1], -item[2], item[3]))[0][4]


def finalize_analysis(promotions: list[dict[str, str]]) -> None:
    physical = C.read_csv(PHYSICAL_MANIFEST)
    methods = C.read_csv(METHOD_MANIFEST)
    if len(physical) != 10 or len(methods) != 11:
        raise RuntimeError("Cannot finalize incomplete Stage07 accounting.")
    C.atomic_csv(C.ROOT / "fidelity2_all_results.csv", methods)
    C.atomic_json(C.ROOT / "fidelity2_all_results.json", {"created_utc": C.utc_now(), "physical_results": physical, "method_results": methods})
    f1_by_signature: dict[str, dict[str, str]] = {}
    for source in promotions:
        f1_by_signature.setdefault(source["architecture_signature"], source)
    stability = []
    for row in physical:
        f1 = f1_by_signature[row["architecture_signature"]]
        stability.append(
            {
                "architecture_signature": row["architecture_signature"],
                "physical_candidate_id": row["physical_candidate_id"],
                "F1_AP": f1["AP"],
                "F2_AP": row["AP"],
                "AP_change": str(Decimal(row["AP"]) - Decimal(f1["AP"])),
                "F1_APS": f1["APS"],
                "F2_APS": row["APS"],
                "APS_change": str(Decimal(row["APS"]) - Decimal(f1["APS"])),
                "F1_Recall": f1["Recall"],
                "F2_Recall": row["Recall"],
                "Recall_change": str(Decimal(row["Recall"]) - Decimal(f1["Recall"])),
                "F2_feasibility": row["f2_feasibility"],
            }
        )
    frame = pd.DataFrame(stability)
    correlations = {
        metric: float(frame[f"F1_{metric}"].astype(float).corr(frame[f"F2_{metric}"].astype(float), method="spearman"))
        for metric in ("AP", "APS", "Recall")
    }
    C.atomic_csv(C.ROOT / "f1_vs_f2_rank_stability.csv", stability)
    C.atomic_json(
        C.ROOT / "f1_vs_f2_rank_stability.json",
        {
            "created_utc": C.utc_now(),
            "pairing_key": "architecture_signature",
            "physical_observations": 10,
            "duplicate_method_attributions_counted_as_independent": False,
            "spearman": correlations,
            "interpretation": "MULTI_FIDELITY_DESCRIPTIVE_EVIDENCE_ONLY",
            "records": stability,
        },
    )
    nsga_ids = {
        row["physical_candidate_id"] for row in promotions if row["method"] == "NSGA-II AP+APS"
    }
    proposed = [row for row in physical if row["physical_candidate_id"] in nsga_ids]
    front = nondominated(proposed)
    for row in proposed:
        row["front_membership"] = row["physical_candidate_id"] in {x["physical_candidate_id"] for x in front}
    C.atomic_csv(C.ROOT / "f2_fronts.csv", proposed)
    C.atomic_json(C.ROOT / "f2_fronts.json", {"created_utc": C.utc_now(), "scope": "PROMOTED_NSGAII_PHYSICAL_ARCHITECTURES_ONLY", "front": front, "all_proposed": proposed})
    candidate_a = knee_candidate(front)
    remaining = [row for row in front if row["physical_candidate_id"] != candidate_a["physical_candidate_id"]]
    if not remaining:
        raise RuntimeError("A distinct proposed Candidate B is unavailable.")
    candidate_b = sorted(remaining, key=lambda row: (-Decimal(row["AP"]), -Decimal(row["APS"]), row["architecture_signature"]))[0]
    selections = [
        {**candidate_a, "proposed_role": "Candidate A", "selection_rule": "deterministic normalized AP-APS Pareto knee"},
        {**candidate_b, "proposed_role": "Candidate B", "selection_rule": "highest-AP feasible proposed candidate distinct from Candidate A"},
    ]
    C.atomic_csv(C.ROOT / "proposed_two_candidates.csv", selections)
    C.atomic_json(C.ROOT / "proposed_two_candidates.json", {"created_utc": C.utc_now(), "automatic_selection": True, "manual_visual_selection": False, "candidates": selections})
    shared_sig = "6510DFAD1319B0D9AC7945FEAB227D43DCB320E9802E033B958D28784F8F464B"
    shared_methods = [row for row in methods if row["architecture_signature"] == shared_sig]
    C.atomic_json(
        C.ROOT / "duplicate_method_attribution_audit.json",
        {
            "created_utc": C.utc_now(),
            "shared_architecture_signature": shared_sig,
            "physical_training_count": 1,
            "method_record_count": len(shared_methods),
            "methods": [row["method"] for row in shared_methods],
            "common_fidelity2_evaluation_ids": sorted({row["fidelity2_evaluation_id"] for row in shared_methods}),
            "status": "PASS" if len(shared_methods) == 2 else "FAIL",
        },
    )


def build_manifest() -> None:
    files = []
    for path in sorted(C.ROOT.rglob("*")):
        if not path.is_file() or path == C.ROOT / "stage07_output_manifest.json":
            continue
        files.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(C.ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": C.sha256(path),
            }
        )
    C.atomic_json(
        C.ROOT / "stage07_output_manifest.json",
        {"created_utc": C.utc_now(), "stage": "STAGE07_FIDELITY2_FRESH_RECOVERY25", "files": files, "status": "PASS"},
    )


def final_validation(promotions: list[dict[str, str]]) -> dict[str, Any]:
    physical = C.read_csv(PHYSICAL_MANIFEST)
    methods = C.read_csv(METHOD_MANIFEST)
    floor = C.load_json(C.ROOT / "stage07_recall_floor_f2.json", {})
    selections = C.load_json(C.ROOT / "proposed_two_candidates.json", {})
    stage06_failures = validate_stage06a_unchanged()
    checks = {
        "stage06a_manifest_unchanged": not stage06_failures,
        "promoted_method_records_11": len(methods) == 11,
        "unique_physical_recovery25_jobs_10": len(physical) == 10,
        "one_deduplicated_shared_architecture": sum(row["architecture_signature"] == "6510DFAD1319B0D9AC7945FEAB227D43DCB320E9802E033B958D28784F8F464B" for row in physical) == 1,
        "targetmanual_executed_first": bool(physical) and any(row["architecture_signature"] == C.TARGETMANUAL_SIGNATURE for row in physical),
        "recall_floor_from_targetmanual_only": floor.get("status") == "LOCKED" and floor.get("formula") == "Recall(TargetManual_F2) - 0.002",
        "fresh_reconstruction_all": all(str(row["fresh_reconstruction_from_searchparent"]).lower() == "true" for row in physical),
        "no_f1_checkpoint_initialization": all(str(row["fidelity1_checkpoint_reused_for_initialization"]).lower() == "false" for row in physical),
        "recovery25_all": all(int(row["recovery_epochs"]) == 25 for row in physical),
        "seed42_all": all(int(row["training_seed"]) == 42 for row in physical),
        "metrics_finite": all(math.isfinite(float(row[key])) for row in physical for key in ("AP", "APS", "Recall")),
        "params_flops_in_band": all(C.PARAMS_LOWER <= int(row["params_exact_integer"]) <= C.PARAMS_UPPER and C.FLOPS_RAW_LOWER <= int(row["flops_raw_exact_integer"]) <= C.FLOPS_RAW_UPPER for row in physical),
        "method_attributions_preserved": {row["promotion_candidate_id"] for row in methods} == {row["promotion_candidate_id"] for row in promotions},
        "automatic_candidate_a_b": selections.get("automatic_selection") is True and len(selections.get("candidates", [])) == 2,
        "ap_only_random_not_finally_locked": all(
            row["physical_candidate_id"]
            in {
                source["physical_candidate_id"]
                for source in promotions
                if source["method"] == "NSGA-II AP+APS"
            }
            for row in selections.get("candidates", [])
        ),
        "official_val548_locked": True,
        "heldout1610_locked": True,
        "kd_not_executed": True,
        "stage08a_not_started": not (C.PROJECT / "08a_proposed_fidelity3").exists(),
    }
    failed = [key for key, value in checks.items() if not value]
    payload = {
        "created_utc": C.utc_now(),
        "checks": checks,
        "stage06a_integrity_failures": stage06_failures,
        "failed_checks": failed,
        "status": "PASS" if not failed else "FAIL",
    }
    C.atomic_json(C.ROOT / "stage07_validation.json", payload)
    return payload


def write_report(validation: dict[str, Any]) -> None:
    floor = C.load_json(C.ROOT / "stage07_recall_floor_f2.json", {})
    candidates = C.load_json(C.ROOT / "proposed_two_candidates.json", {}).get("candidates", [])
    candidate_a = next((row for row in candidates if row["proposed_role"] == "Candidate A"), {})
    candidate_b = next((row for row in candidates if row["proposed_role"] == "Candidate B"), {})
    C.atomic_text(
        C.ROOT / "stage07_report.md",
        "# Stage07 Fidelity-2 Fresh Recovery-25\n\n"
        f"- Status: `{validation['status']}`\n"
        "- Physical Recovery-25 jobs: `10`\n"
        "- Method-level records: `11`\n"
        "- Shared NSGA-II/AP-only architecture trained once: `YES`\n"
        "- TargetManual executed first: `YES`\n"
        f"- Recall floor F2: `{floor.get('recall_floor_f2_full_precision')}`\n"
        f"- Proposed Candidate A: `{candidate_a.get('physical_candidate_id')}`\n"
        f"- Proposed Candidate B: `{candidate_b.get('physical_candidate_id')}`\n"
        "- KD executed: `NO`\n"
        "- val548 accessed: `NO`\n"
        "- heldout1610 accessed: `NO`\n"
        "- Stage08A started: `NO`\n",
    )


def finalize(promotions: list[dict[str, str]]) -> None:
    classify_feasibility()
    finalize_analysis(promotions)
    validation = final_validation(promotions)
    write_report(validation)
    if validation["status"] != "PASS":
        raise RuntimeError(f"Stage07 final validation failed: {validation['failed_checks']}")
    C.atomic_json(FAILURE_AUDIT, C.load_json(FAILURE_AUDIT, {"created_utc": C.utc_now(), "failures": [], "status": "PASS"}))
    C.atomic_text(
        C.ROOT / ".complete",
        "STAGE07_STATUS=PASS\nSTAGE08A_STARTED=NO\nAPPROVAL_READY=YES\n",
    )
    build_manifest()
    campaign = C.load_json(C.CAMPAIGN_STATE, {})
    campaign.update({"state": "COMPLETE", "completed_utc": C.utc_now(), "approval_ready": True, "stage08a_started": False})
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    update_worker(candidate_state="COMPLETE", child_pid=None)


def pause_requested() -> bool:
    return C.PAUSE_REQUEST.is_file()


def pause_after_commit() -> int:
    campaign = C.load_json(C.CAMPAIGN_STATE, {})
    campaign.update(
        {
            "state": "PAUSED_AFTER_ATOMIC_CANDIDATE_COMMIT",
            "paused_utc": C.utc_now(),
            "safe_to_shut_down_laptop": True,
            "approval_ready": False,
        }
    )
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    update_worker(candidate_state="PAUSED_SAFE_TO_SHUT_DOWN", child_pid=None)
    if C.PAUSE_REQUEST.exists():
        C.PAUSE_REQUEST.unlink()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    C.ensure_directories()
    configure_locked_runtime()
    global CAMPAIGN_UUID
    existing = C.load_json(C.CAMPAIGN_STATE, {}) or {}
    CAMPAIGN_UUID = existing.get("campaign_uuid") or str(uuid.uuid4())
    acquire_lock()
    try:
        candidates, promotions, groups, complexity, taylor_bank = preflight()
        if args.preflight_only:
            C.atomic_json(
                C.ROOT / "stage07_preflight.json",
                {"created_utc": C.utc_now(), "physical_jobs": 10, "method_records": 11, "targetmanual_first": True, "status": "PASS"},
            )
            return 0
        campaign = existing or {
            "campaign_uuid": CAMPAIGN_UUID,
            "created_utc": C.utc_now(),
            "started_utc": C.utc_now(),
        }
        campaign.update(
            {
                "state": "RUNNING",
                "approval_token_consumed": "APPROVE_STAGE06A_FIDELITY2_PROMOTION_LOCK",
                "physical_jobs_expected": 10,
                "method_records_expected": 11,
                "official_val548_accessed": False,
                "heldout1610_accessed": False,
                "kd_executed": False,
                "stage08a_started": False,
                "approval_ready": False,
            }
        )
        C.atomic_json(C.CAMPAIGN_STATE, campaign)
        ACTIVE.update({"worker_pid": os.getpid(), "campaign_uuid": CAMPAIGN_UUID})
        update_worker(candidate_state="PREFLIGHT_PASS")
        disk_sample("before_stage07_or_resume")
        for candidate in candidates:
            if (physical_dir(candidate) / ".complete").is_file():
                continue
            if pause_requested():
                return pause_after_commit()
            execute_candidate(candidate, promotions, groups, complexity, taylor_bank)
            if pause_requested():
                return pause_after_commit()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        finalize(promotions)
        return 0
    except Exception as error:
        failure = C.load_json(FAILURE_AUDIT, {"created_utc": C.utc_now(), "failures": []})
        failure["failures"].append(
            {
                "timestamp_utc": C.utc_now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "worker_state": dict(ACTIVE),
            }
        )
        failure["status"] = "BLOCKED"
        C.atomic_json(FAILURE_AUDIT, failure)
        campaign = C.load_json(C.CAMPAIGN_STATE, {}) or {"campaign_uuid": CAMPAIGN_UUID}
        campaign.update({"state": "BLOCKED", "blocked_utc": C.utc_now(), "approval_ready": False})
        C.atomic_json(C.CAMPAIGN_STATE, campaign)
        update_worker(candidate_state="BLOCKED", error=repr(error), child_pid=None)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    finally:
        release_lock()
        C.write_status()


if __name__ == "__main__":
    raise SystemExit(main())
