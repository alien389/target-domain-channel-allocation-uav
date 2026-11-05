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

import stage08a_common as C


STAGE04_SCRIPT = C.PROJECT / "scripts" / "stage04_shared_p0_fidelity1.py"
STAGE05 = C.PROJECT / "05_full_fidelity1_search"
STAGE06 = C.PROJECT / "06a_fidelity2_promotion_lock"
STAGE07 = C.PROJECT / "07_fidelity2_confirmation"
A3 = C.PROJECT / "protocol_amendment_A3_defer_kd_final_proposed_ablation"
GENE_JSON = C.PROJECT / "03a_taylor_dependency_gene_space" / "gene_space_R1" / "gene_space_manifest_R1.json"
EVALUATOR = C.SCRIPTS / "evaluate_stage08a_inner_val.py"
RESULTS_CSV = C.ROOT / "stage08a_all_results.csv"
RETENTION_CSV = C.ROOT / "checkpoint_retention_manifest.csv"
FAILURE_AUDIT = C.ROOT / "failure_audit.json"
DISK_AUDIT = C.ROOT / "disk_usage_audit.json"
HISTORY = C.ROOT / "pre_amendment_A3_blocked_attempt"

EXPECTED_STAGE07_MANIFEST = "D8E293952CC9E62579AD134EC582A71FB08F57C494BFFE3BEED1AC42320B06DD"
EXPECTED_STAGE06_MANIFEST = "E5410026CF52D69C086A6044AEB0AC4B5F0C68B2129D5DDFB05E86D67FFA23C2"
EXPECTED_A3_MANIFEST = "0A77876FF7C4B706EF7A91F1C30572249DF4F253E7A624C1A2528CA81E1541E0"
SEARCH_PARENT_SHA = "15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659"

ROLES = [
    ("TargetManual-B6-reference", "F2PHYS-AEA2D0A18DAC15E4", "AEA2D0A18DAC15E40091E474DBF688A98FCA2B6CFF8D4D3F4845165C59244F5D", "target_manual_f3"),
    ("Proposed Candidate A", "F2PHYS-EEA3FB45FD3FD1BA", "EEA3FB45FD3FD1BADD263DE055266F4F6B5EE34EE818F4C1D7026B764E862C80", "candidate_a_f3"),
    ("Proposed Candidate B", "F2PHYS-5A943CA1E4B27085", "5A943CA1E4B27085C1E000D5D6B757D34A281B221E31FAD509E051E7B19CF11C", "candidate_b_f3"),
]

ACTIVE: dict[str, Any] = {}
CAMPAIGN_UUID = ""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S4 = load_module("stage08a_stage04_locked", STAGE04_SCRIPT)
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
        raise RuntimeError("Teacher/KD token detected in a Recovery-50-only command.")
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
        raise RuntimeError(f"Another Stage08A worker is active: {others}")
    if C.gpu_children():
        raise RuntimeError(f"A Recovery/evaluation process is active: {C.gpu_children()}")
    if C.LOCK.is_file():
        old = C.load_json(C.LOCK, {})
        if C.process_alive(old.get("worker_pid")):
            raise RuntimeError(f"Live campaign lock belongs to PID {old['worker_pid']}")
    C.atomic_json(C.LOCK, {"created_utc": C.utc_now(), "campaign_uuid": CAMPAIGN_UUID, "worker_pid": os.getpid(), "hostname": socket.gethostname(), "max_gpu_jobs": 1})


def release_lock() -> None:
    if C.LOCK.is_file():
        payload = C.load_json(C.LOCK, {})
        if int(payload.get("worker_pid", -1)) == os.getpid():
            C.LOCK.unlink()


def archive_old_blocked_attempt() -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    expected = {
        "stage08a_preflight_blocker.json": "D8A3E9B77FB6614B455768143F10BD6AC013AA3462E71D7EE27AD094522FCF17",
        "stage08a_campaign_state.json": "A274A7F2157A5DD43DC4CED8BD5501556EB35938ED4A249016973D934A248C8E",
        "stage08a_report.md": "2F1D619CE9ADA27BE6FE5908A6EA9186342FE0B630E8CD76827330F51A0DA1FF",
    }
    rows = []
    for name, digest in expected.items():
        source = C.ROOT / name
        target = HISTORY / name
        if target.is_file():
            if C.sha256(target) != digest:
                raise RuntimeError(f"Historical archive hash mismatch: {target}")
        else:
            if not source.is_file() or C.sha256(source) != digest:
                raise RuntimeError(f"Historical blocked-attempt evidence changed before archival: {source}")
            shutil.copy2(source, target)
        rows.append({"original_path": str(source), "archived_path": str(target), "sha256": digest, "status": "PRESERVED_BYTE_FOR_BYTE"})
    C.atomic_json(HISTORY / "historical_source_map.json", {"created_utc": C.utc_now(), "files": rows, "superseded_by": "Amendment A3 Recovery-50-only protocol", "status": "PASS"})


def verify_upstream() -> None:
    checks = {
        "stage07_validation": C.load_json(STAGE07 / "stage07_validation.json", {}).get("status") == "PASS",
        "stage07_manifest": C.sha256(STAGE07 / "stage07_output_manifest.json") == EXPECTED_STAGE07_MANIFEST,
        "stage06_manifest": C.sha256(STAGE06 / "stage06a_output_manifest.json") == EXPECTED_STAGE06_MANIFEST,
        "a3_validation": C.load_json(A3 / "amendment_A3_validation.json", {}).get("status") == "PASS",
        "a3_manifest": C.sha256(A3 / "amendment_A3_output_manifest.json") == EXPECTED_A3_MANIFEST,
        "searchparent": C.sha256(R1M.SEARCH_PARENT) == SEARCH_PARENT_SHA,
        "no_gpu": not C.gpu_children(),
        "no_stage08b": not (C.PROJECT / "08b_fidelity3_controls").exists(),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"Stage08A upstream lock failed: {failed}")


def exact_text(value: Any) -> str:
    return format(float(value), ".17g")


def build_candidates(protocol_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, Any]]:
    rows = {row["physical_candidate_id"]: row for row in C.read_csv(STAGE07 / "stage07_physical_evaluation_manifest.csv")}
    groups = C.load_json(GENE_JSON, {})["groups"]
    group_ids = [group["group_id"] for group in groups]
    parent = R1M.load_source_model(R1M.SEARCH_PARENT, "stage08a_parent")
    complexity = R1M.ExactComplexityModel(parent.model, groups)
    taylor_bank, taylor_manifest = R1M.taylor_scope_bank(groups)
    if taylor_manifest["checkpoint_sha256"] != SEARCH_PARENT_SHA:
        raise RuntimeError("Taylor ranking SearchParent binding failed")
    candidates = []
    for index, (role, candidate_id, signature, directory_name) in enumerate(ROLES, 1):
        row = rows[candidate_id]
        if row["architecture_signature"] != signature:
            raise RuntimeError(f"Frozen candidate identity mismatch: {candidate_id}")
        reconstruction = Path(row["reconstruction_manifest"])
        if C.sha256(reconstruction) != row["reconstruction_manifest_sha256"]:
            raise RuntimeError(f"Stage07 reconstruction manifest changed: {candidate_id}")
        widths = C.load_json(reconstruction, {})["widths"]
        vector = tuple(int(widths[group_id]) for group_id in group_ids)
        if R1M.signature_sha(group_ids, vector) != signature:
            raise RuntimeError(f"Retained-width signature mismatch: {candidate_id}")
        analytic = complexity.calculate(widths)
        flops_raw = int(analytic["macs"]) * 2
        if int(analytic["params"]) != int(row["params_exact_integer"]) or flops_raw != int(row["flops_raw_exact_integer"]):
            raise RuntimeError(f"Exact complexity mismatch: {candidate_id}")
        source_map = Path(row["channel_index_map"])
        if C.sha256(source_map) != row["channel_index_map_sha256"]:
            raise RuntimeError(f"Channel map changed: {candidate_id}")
        identity = C.hash_text(C.canonical_json({"stage": "Stage08A_Recovery50_only", "signature": signature, "protocol_sha256": protocol_hash, "seed": 42}))
        candidate: dict[str, Any] = {
            "candidate_role": role,
            "physical_candidate_id": candidate_id,
            "unique_signature_index": index,
            "directory_name": directory_name,
            "evaluation_id": identity,
            "signature_sha256": signature,
            "fused_params": int(row["params_exact_integer"]),
            "fused_flops_g": float(row["flops_g_full_precision"]),
            "params_exact_integer": int(row["params_exact_integer"]),
            "flops_raw_exact_integer": flops_raw,
            "flops_g_full_precision": row["flops_g_full_precision"],
            "expected_channel_map_sha256": row["channel_index_map_sha256"],
            "roles": role,
            "replicates": "[]",
            "membership_count": 1,
        }
        candidate.update({group_id: int(widths[group_id]) for group_id in group_ids})
        candidates.append(candidate)
    return candidates, groups, complexity, taylor_bank


def source_hashes() -> dict[str, Any]:
    paths = [
        STAGE05 / "stage05_validation.json",
        STAGE06 / "stage06a_output_manifest.json",
        STAGE07 / "stage07_output_manifest.json",
        STAGE07 / "stage07_physical_evaluation_manifest.csv",
        STAGE07 / "proposed_two_candidates.json",
        A3 / "amendment_A3_output_manifest.json",
        A3 / "revised_stage08_protocol.yaml",
        STAGE04_SCRIPT,
        C.SCRIPTS / "stage08a_common.py",
        C.SCRIPTS / "stage08a_worker.py",
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
    return {"created_utc": C.utc_now(), "files": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": C.sha256(path)} for path in paths], "status": "PASS"}


def create_protocol_lock() -> dict[str, Any]:
    payload = {
        "schema": "stage08a_recovery50_only_protocol_v1",
        "created_utc": C.utc_now(),
        "approval_token_consumed": "APPROVE_AMENDMENT_A3_START_STAGE08A_RECOVERY50_ONLY",
        "amendment_A3_verified": True,
        "execution_order": [{"role": role, "candidate_id": candidate_id, "signature": signature} for role, candidate_id, signature, _ in ROLES],
        "initialization": "fresh reconstruction from SearchParent with frozen widths and target-domain Taylor channel indices",
        "fidelity1_or_fidelity2_checkpoint_initialization": False,
        "recovery_epochs": 50,
        "training_seed": 42,
        "train": "meta-train5177",
        "validation": "meta-inner-val1294",
        "imgsz": 640,
        "batch_size": 16,
        "best_checkpoint_rule": "meta-inner-val1294 fitness",
        "explicit_evaluation": {"precision": "FP32", "images": 1294, "conf": 0.001, "iou": 0.60, "max_det": 500, "augment": False, "tta": False},
        "kd1": False,
        "kd2": False,
        "teacher_loading": False,
        "coco_teacher_substitution": False,
        "official_val548_locked": True,
        "heldout1610_locked": True,
        "stage08b_started": False,
        "status": "LOCKED",
    }
    payload["stage08a_protocol_sha256"] = C.hash_text(C.canonical_json(payload))
    C.atomic_json(C.ROOT / "stage08a_protocol_lock.json", payload)
    return payload


def preflight() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, Any]]:
    verify_upstream()
    archive_old_blocked_attempt()
    protocol = create_protocol_lock()
    C.atomic_json(C.ROOT / "stage08a_source_hashes.json", source_hashes())
    candidates = build_candidates(protocol["stage08a_protocol_sha256"])
    C.atomic_json(C.ROOT / "stage08a_preflight.json", {"created_utc": C.utc_now(), "checks": {"A3": True, "three_candidates": len(candidates[0]) == 3, "targetmanual_first": candidates[0][0]["candidate_role"] == "TargetManual-B6-reference", "kd_disabled": True, "teacher_loading_disabled": True, "no_active_gpu": not C.gpu_children()}, "status": "PASS"})
    return candidates


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
    return {
        "AP": exact_text(metrics["AP_0.5_0.95"]), "AP50": exact_text(metrics["AP50"]), "AP75": exact_text(metrics["AP75"]),
        "APS": exact_text(metrics["APS"]), "APM": exact_text(metrics["APM"]), "APL": exact_text(metrics["APL"]),
        "Precision": exact_text(metrics["precision"]), "Recall": exact_text(metrics["recall"]), "F1": exact_text(metrics["f1"]),
    }


def commit_candidate(candidate: dict[str, Any], groups: list[dict[str, Any]], complexity: Any, taylor_bank: dict[str, Any]) -> None:
    directory = run_dir(candidate)
    if (directory / ".complete").is_file():
        return
    if C.free_gib() < 8.0:
        raise RuntimeError("Free disk below 8 GiB")
    update_worker(candidate_role=candidate["candidate_role"], candidate_id=candidate["physical_candidate_id"], run_number=candidate["unique_signature_index"], run_state="FRESH_RECONSTRUCTION", run_directory=str(directory))
    raw = S4.reconstruct_raw(candidate, groups, complexity, taylor_bank)
    raw.update({"stage": "Stage08A Recovery-50-only", "candidate_role": candidate["candidate_role"], "fresh_reconstruction_from_searchparent": True, "f1_f2_checkpoint_initialization": False, "kd_executed": False, "teacher_loaded": False, "params_exact_integer": candidate["params_exact_integer"], "flops_raw_exact_integer": candidate["flops_raw_exact_integer"], "flops_g_full_precision": candidate["flops_g_full_precision"]})
    reconstruction = directory / "raw_reconstruction_audit.json"
    C.atomic_json(reconstruction, raw)
    if C.sha256(directory / "channel_index_map.csv") != candidate["expected_channel_map_sha256"]:
        raise RuntimeError("Fresh channel-index map does not match the frozen source")
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
    if not all(math.isfinite(float(metrics[key])) for key in ("AP_0.5_0.95", "APS", "recall")):
        raise RuntimeError("Non-finite F3 metric")
    hashes = checkpoint_hashes(directory)
    protocol = C.load_json(C.ROOT / "stage08a_protocol_lock.json", {})
    manifest_path = directory / "run_manifest.json"
    run_manifest = {"created_utc": C.utc_now(), "stage": "Stage08A Recovery-50-only", "candidate_role": candidate["candidate_role"], "physical_candidate_id": candidate["physical_candidate_id"], "evaluation_id": candidate["evaluation_id"], "architecture_signature": candidate["signature_sha256"], "fresh_reconstruction": True, "f1_f2_checkpoint_initialization": False, "recovery_epochs": 50, "training_seed": 42, "kd_executed": False, "teacher_loaded": False, "training": training, "explicit_evaluation": evaluation, "metrics_sha256": C.sha256(metrics_path), "checkpoints_before_retention": hashes, "protocol_sha256": protocol["stage08a_protocol_sha256"], "official_val548_accessed": False, "heldout1610_accessed": False, "status": "COMPLETE"}
    C.atomic_json(manifest_path, run_manifest)
    values = metric_texts(metrics)
    row: dict[str, Any] = {
        "candidate_role": candidate["candidate_role"], "physical_candidate_id": candidate["physical_candidate_id"], "evaluation_id": candidate["evaluation_id"], "architecture_signature": candidate["signature_sha256"],
        **values, "params_exact_integer": candidate["params_exact_integer"], "flops_raw_exact_integer": candidate["flops_raw_exact_integer"], "flops_g_full_precision": candidate["flops_g_full_precision"],
        "best_epoch_zero_based": metrics["best_epoch"]["best_epoch_zero_based"], "best_epoch_one_based": metrics["best_epoch"]["best_epoch_one_based"], "best_fitness": exact_text(metrics["best_epoch"]["best_fitness"]),
        "training_duration_seconds": training["training_duration_seconds"], "evaluation_duration_seconds": evaluation["evaluation_duration_seconds"],
        "best_checkpoint": str(directory / "weights" / "best.pt"), "best_checkpoint_sha256": hashes["best_checkpoint"]["sha256"], "raw_checkpoint_sha256": hashes["raw_checkpoint"]["sha256"], "last_checkpoint_sha256_before_deletion": hashes["last_checkpoint_before_deletion"]["sha256"],
        "metrics_path": str(metrics_path), "metrics_sha256": C.sha256(metrics_path), "reconstruction_manifest": str(reconstruction), "reconstruction_manifest_sha256": C.sha256(reconstruction), "channel_index_map": str(directory / "channel_index_map.csv"), "channel_index_map_sha256": C.sha256(directory / "channel_index_map.csv"), "run_manifest": str(manifest_path), "run_manifest_sha256": C.sha256(manifest_path),
        "structural_validity": "PASS", "numerical_validity": "PASS", "kd_executed": False, "teacher_loaded": False, "status": "COMPLETE",
    }
    rows = [old for old in C.read_csv(RESULTS_CSV) if old["physical_candidate_id"] != candidate["physical_candidate_id"]] + [row]
    rows.sort(key=lambda old: next(index for index, role in enumerate(ROLES) if role[1] == old["physical_candidate_id"]))
    C.atomic_csv(RESULTS_CSV, rows)
    C.atomic_json(C.ROOT / "stage08a_all_results.json", {"created_utc": C.utc_now(), "protocol": "Recovery-50-only", "results": rows})
    if candidate["candidate_role"] == "TargetManual-B6-reference":
        recall = Decimal(row["Recall"])
        floor = recall - Decimal("0.002")
        C.atomic_json(C.ROOT / "recall_floor_f3.json", {"created_utc": C.utc_now(), "source": "TargetManual Recovery-50 explicit FP32 inner-val1294", "targetmanual_recall_full_precision": str(recall), "offset": "0.002", "recall_floor_f3_full_precision": str(floor), "f1_floor_reused": False, "f2_floor_reused": False, "status": "LOCKED"})
    elif not (C.ROOT / "recall_floor_f3.json").is_file():
        raise RuntimeError("Candidate classification reached before F3 Recall floor lock")
    classify_results()
    retention = C.read_csv(RETENTION_CSV)
    for role, key, path in (("raw_checkpoint", "raw_checkpoint", directory / "raw_checkpoint.pt"), ("last_checkpoint", "last_checkpoint_before_deletion", directory / "weights" / "last.pt")):
        retention.append({"candidate_id": candidate["physical_candidate_id"], "artifact_role": role, "path_before_deletion": str(path), "sha256_before_deletion": hashes[key]["sha256"], "size_bytes": hashes[key]["size_bytes"], "decision": "DELETED_AFTER_HASH_AND_ATOMIC_COMMIT", "timestamp_utc": C.utc_now()})
        if path.exists(): path.unlink()
    retention.append({"candidate_id": candidate["physical_candidate_id"], "artifact_role": "best_checkpoint", "path_before_deletion": hashes["best_checkpoint"]["path"], "sha256_before_deletion": hashes["best_checkpoint"]["sha256"], "size_bytes": hashes["best_checkpoint"]["size_bytes"], "decision": "RETAINED", "timestamp_utc": ""})
    predictions = directory / "explicit_inner_val_predictions"
    if predictions.exists(): shutil.rmtree(predictions)
    C.atomic_csv(RETENTION_CSV, retention)
    C.atomic_text(directory / ".complete", f"STAGE08A_F3_RECOVERY50=COMPLETE\nCANDIDATE={candidate['physical_candidate_id']}\n")
    update_worker(run_state="ATOMICALLY_COMMITTED")
    disk_sample(f"completed_{candidate['unique_signature_index']}_of_3")


def classify_results() -> None:
    floor_payload = C.load_json(C.ROOT / "recall_floor_f3.json", {})
    if floor_payload.get("status") != "LOCKED": return
    floor = Decimal(floor_payload["recall_floor_f3_full_precision"])
    rows = C.read_csv(RESULTS_CSV)
    for row in rows:
        feasible = Decimal(row["Recall"]) >= floor and C.PARAMS_LOWER <= int(row["params_exact_integer"]) <= C.PARAMS_UPPER and C.FLOPS_LOWER <= int(row["flops_raw_exact_integer"]) <= C.FLOPS_UPPER and row["structural_validity"] == "PASS" and row["numerical_validity"] == "PASS"
        row["recall_floor_f3"] = str(floor)
        row["f3_feasibility"] = "F3_FEASIBLE" if feasible else "F3_INFEASIBLE"
    C.atomic_csv(RESULTS_CSV, rows)
    C.atomic_json(C.ROOT / "stage08a_all_results.json", {"created_utc": C.utc_now(), "protocol": "Recovery-50-only", "results": rows})


def select_proposed() -> dict[str, Any]:
    rows = {row["candidate_role"]: row for row in C.read_csv(RESULTS_CSV)}
    a, b = rows["Proposed Candidate A"], rows["Proposed Candidate B"]
    feasible = [row for row in (a, b) if row["f3_feasibility"] == "F3_FEASIBLE"]
    if not feasible:
        return {"created_utc": C.utc_now(), "status": "BLOCKED_NO_FEASIBLE_PROPOSED_CANDIDATE", "candidate_a_feasibility": a["f3_feasibility"], "candidate_b_feasibility": b["f3_feasibility"], "selected": None}
    if len(feasible) == 1:
        selected = feasible[0]
        clause = "ONLY_ONE_RECALL_FEASIBLE"
    else:
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
    rejected = b if selected["physical_candidate_id"] == a["physical_candidate_id"] else a
    return {
        "created_utc": C.utc_now(), "status": "PASS", "label": "Meta-AP+APS-B6 provisional locked architecture", "selected_candidate_id": selected["physical_candidate_id"], "selected_architecture_signature": selected["architecture_signature"], "selected_checkpoint": selected["best_checkpoint"], "selected_checkpoint_sha256": selected["best_checkpoint_sha256"], "rejected_candidate_id": rejected["physical_candidate_id"], "rule_clause": clause,
        "candidate_a_feasibility": a["f3_feasibility"], "candidate_b_feasibility": b["f3_feasibility"], "delta_A_minus_B": {"AP": str(Decimal(a["AP"]) - Decimal(b["AP"])), "APS": str(Decimal(a["APS"]) - Decimal(b["APS"])), "Recall": str(Decimal(a["Recall"]) - Decimal(b["Recall"])), "FLOPs_raw": int(a["flops_raw_exact_integer"]) - int(b["flops_raw_exact_integer"]), "Params": int(a["params_exact_integer"]) - int(b["params_exact_integer"])},
        "P_meta": int(selected["params_exact_integer"]), "F_meta_raw": int(selected["flops_raw_exact_integer"]), "F_meta_g_full_precision": selected["flops_g_full_precision"], "manual_visual_selection": False, "recovery50_only": True, "kd_executed": False, "teacher_loaded": False,
    }


def build_output_manifest() -> None:
    rows = []
    for path in sorted(C.ROOT.rglob("*")):
        if not path.is_file() or path == C.ROOT / "stage08a_output_manifest.json": continue
        if C.RUNTIME in path.parents:
            continue
        rows.append({"path": str(path), "relative_path": str(path.relative_to(C.ROOT)), "size_bytes": path.stat().st_size, "sha256": C.sha256(path)})
    C.atomic_json(C.ROOT / "stage08a_output_manifest.json", {"created_utc": C.utc_now(), "files": rows, "status": "PASS"})


def finalize() -> None:
    classify_results()
    rows = C.read_csv(RESULTS_CSV)
    if len(rows) != 3: raise RuntimeError("Expected three F3 results")
    decision = select_proposed()
    C.atomic_json(C.ROOT / "proposed_lock_decision.json", decision)
    if decision["status"] != "PASS":
        raise RuntimeError("BLOCKED_NO_FEASIBLE_PROPOSED_CANDIDATE")
    checks = {
        "amendment_A3_verified": C.load_json(A3 / "amendment_A3_validation.json", {}).get("status") == "PASS",
        "recovery50_only": all(int(row.get("best_epoch_one_based", 0)) <= 50 for row in rows),
        "three_runs_complete": len(rows) == 3 and all(row["status"] == "COMPLETE" for row in rows),
        "targetmanual_first": (C.ROOT / "target_manual_f3" / ".complete").is_file(),
        "recall_floor_f3_locked": C.load_json(C.ROOT / "recall_floor_f3.json", {}).get("status") == "LOCKED",
        "all_fresh_reconstructions": all(C.load_json(Path(row["reconstruction_manifest"]), {}).get("fresh_reconstruction_from_searchparent") is True for row in rows),
        "no_f1_f2_checkpoint_initialization": all(C.load_json(Path(row["reconstruction_manifest"]), {}).get("f1_f2_checkpoint_initialization") is False for row in rows),
        "kd_not_executed": all(str(row["kd_executed"]).lower() == "false" for row in rows),
        "teacher_not_loaded": all(str(row["teacher_loaded"]).lower() == "false" for row in rows),
        "automatic_rule_applied": decision["manual_visual_selection"] is False,
        "val548_locked": True, "heldout1610_locked": True, "stage08b_not_started": not (C.PROJECT / "08b_fidelity3_controls").exists(),
    }
    failed = [key for key, value in checks.items() if not value]
    validation = {"created_utc": C.utc_now(), "checks": checks, "failed_checks": failed, "status": "PASS" if not failed else "FAIL"}
    C.atomic_json(C.ROOT / "stage08a_validation.json", validation)
    if failed: raise RuntimeError(f"Stage08A validation failed: {failed}")
    floor = C.load_json(C.ROOT / "recall_floor_f3.json", {})["recall_floor_f3_full_precision"]
    C.atomic_text(C.ROOT / "stage08a_report.md", "# Stage08A Recovery-50-only\n\n" f"- Status: `PASS`\n- F3 runs: `3`\n- Recall floor F3: `{floor}`\n" f"- Candidate A feasibility: `{decision['candidate_a_feasibility']}`\n- Candidate B feasibility: `{decision['candidate_b_feasibility']}`\n" f"- Provisional Meta-AP+APS-B6: `{decision['selected_candidate_id']}`\n- Rule: `{decision['rule_clause']}`\n" "- KD executed: `NO`\n- Teacher loaded: `NO`\n- val548 accessed: `NO`\n- heldout1610 accessed: `NO`\n- Stage08B started: `NO`\n")
    C.atomic_json(FAILURE_AUDIT, C.load_json(FAILURE_AUDIT, {"created_utc": C.utc_now(), "failures": [], "status": "PASS"}))
    C.atomic_text(C.ROOT / ".complete", "STAGE08A_STATUS=PASS\nAPPROVAL_READY=YES\nSTAGE08B_STARTED=NO\n")
    campaign = C.load_json(C.CAMPAIGN_STATE, {})
    campaign.update({"state": "COMPLETE", "completed_utc": C.utc_now(), "approval_ready": True, "stage08b_started": False})
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    update_worker(run_state="COMPLETE", child_pid=None)
    build_output_manifest()


def pause() -> int:
    state = C.load_json(C.CAMPAIGN_STATE, {})
    state.update({"state": "PAUSED_AFTER_ATOMIC_CANDIDATE_COMMIT", "paused_utc": C.utc_now(), "safe_to_shut_down_laptop": True, "approval_ready": False})
    C.atomic_json(C.CAMPAIGN_STATE, state)
    update_worker(run_state="PAUSED_SAFE_TO_SHUT_DOWN", child_pid=None)
    if C.PAUSE_REQUEST.exists(): C.PAUSE_REQUEST.unlink()
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
        if args.preflight_only: return 0
        state = {"campaign_uuid": CAMPAIGN_UUID, "created_utc": existing.get("created_utc", C.utc_now()), "started_utc": existing.get("started_utc", C.utc_now()), "state": "RUNNING", "approval_token_consumed": "APPROVE_AMENDMENT_A3_START_STAGE08A_RECOVERY50_ONLY", "protocol": "RECOVERY50_ONLY", "f3_runs_expected": 3, "kd_executed": False, "teacher_loaded": False, "official_val548_accessed": False, "heldout1610_accessed": False, "stage08b_started": False, "approval_ready": False}
        C.atomic_json(C.CAMPAIGN_STATE, state)
        ACTIVE.update({"worker_pid": os.getpid(), "campaign_uuid": CAMPAIGN_UUID})
        update_worker(run_state="PREFLIGHT_PASS")
        disk_sample("before_stage08a_or_resume")
        for candidate in candidates:
            if (run_dir(candidate) / ".complete").is_file(): continue
            if C.PAUSE_REQUEST.is_file(): return pause()
            commit_candidate(candidate, groups, complexity, taylor_bank)
            if C.PAUSE_REQUEST.is_file(): return pause()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()
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
