from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE01 = PROJECT / "01_data_split"
STAGE01A = PROJECT / "01A_protocol_amendment"
STAGE03A = PROJECT / "03a_taylor_dependency_gene_space"
R1_ROOT = STAGE03A / "gene_space_R1"
STAGE03B = PROJECT / "03b_search_initialization"
OUTPUT = PROJECT / "04_shared_p0_fidelity1"
PHYSICAL_ROOT = OUTPUT / "physical_evaluations"
SCRIPT_PATH = PROJECT / "scripts" / "stage04_shared_p0_fidelity1.py"
TRAIN_WRAPPER = PROJECT / "scripts" / "train_structural_recovery_drone.py"
EVAL_WRAPPER = PROJECT / "scripts" / "evaluate_stage04_inner_val.py"
VISDRONE_PYTHON_PORT = PROJECT / "scripts" / "visdrone_toolkit_python_port.py"
VISDRONE_PARITY_AUDIT = OUTPUT / "visdrone_evaluator_parity_audit.json"
R1_SCRIPT = PROJECT / "scripts" / "stage03a_r1_gene_space_repair.py"

PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))
DATA_YAML = (
    STAGE01
    / "data_yamls"
    / "visdrone_meta_train5177_inner1294.yaml"
)
META_TRAIN = STAGE01 / "meta_train5177_seed42.txt"
META_INNER_VAL = STAGE01 / "meta_inner_val1294_seed42.txt"
OFFICIAL_VAL_LOCK = STAGE01 / "official_val548_locked.txt"
HYP_YAML = Path(_release_paths.expand('${SOURCE_PARENT_HYP}'))
P0_UNIQUE = STAGE03B / "p0" / "p0_unique_signatures.csv"
P0_MEMBERSHIP = STAGE03B / "p0" / "p0_membership_manifest.csv"
GENE_JSON = R1_ROOT / "gene_space_manifest_R1.json"
TAYLOR_RANKING = STAGE03A / "taylor_ranking_manifest.json"
DEPENDENCY_GRAPH = STAGE03A / "dependency_graph.json"
EVALUATION_SCHEMA = STAGE03B / "evaluation_id_schema.json"
SEARCH_RECORD_SCHEMA = STAGE03B / "search_record_id_schema.json"
STAGE03B_VALIDATION = STAGE03B / "stage03b_validation.json"
STAGE03B_STATE = STAGE03B / "stage03b_state.json"
STAGE03B_MANIFEST = STAGE03B / "stage03b_output_manifest.json"
TRAINING_PROTOCOL = STAGE01A / "revised_frozen_hyperparameters.yaml"

SEARCH_PARENT_SHA = (
    "15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659"
)
METHODS = (
    "NSGA-II AP+APS",
    "Evolutionary AP-only",
    "Warm-started constrained Random Search",
)
EXPECTED_UNIQUE = 35
EXPECTED_MEMBERSHIPS = 48
EXPECTED_METHOD_RECORDS = 144
EXPECTED_TRAIN_IMAGES = 5177
EXPECTED_VAL_IMAGES = 1294
RECOVERY_EPOCHS = 8
TRAINING_SEED = 42
DISK_STOP_GIB = 8.0
IMGSZ = 640
BATCH_SIZE = 16
FLOPS_TOLERANCE_G = 1e-9


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1M = load_module("stage04_r1_locked", R1_SCRIPT)


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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def list_count(path: Path) -> int:
    return len(
        [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    )


def disk_free_gib() -> float:
    return shutil.disk_usage(OUTPUT.anchor).free / (1024**3)


def disk_audit(event: str, completed_jobs: int) -> None:
    path = OUTPUT / "disk_usage_audit.json"
    payload = (
        json_load(path)
        if path.exists()
        else {
            "created_utc": utc_now(),
            "drive": OUTPUT.anchor,
            "minimum_free_gib_before_new_job": DISK_STOP_GIB,
            "samples": [],
        }
    )
    usage = shutil.disk_usage(OUTPUT.anchor)
    payload["samples"].append(
        {
            "timestamp_utc": utc_now(),
            "event": event,
            "completed_jobs": completed_jobs,
            "total_gib": usage.total / (1024**3),
            "used_gib": usage.used / (1024**3),
            "free_gib": usage.free / (1024**3),
        }
    )
    payload["latest_free_gib"] = usage.free / (1024**3)
    json_write(path, payload)


def source_paths() -> list[Path]:
    return [
        DATA_YAML,
        META_TRAIN,
        META_INNER_VAL,
        OFFICIAL_VAL_LOCK,
        HYP_YAML,
        P0_UNIQUE,
        P0_MEMBERSHIP,
        GENE_JSON,
        TAYLOR_RANKING,
        DEPENDENCY_GRAPH,
        EVALUATION_SCHEMA,
        SEARCH_RECORD_SCHEMA,
        STAGE03B_VALIDATION,
        STAGE03B_STATE,
        STAGE03B_MANIFEST,
        TRAINING_PROTOCOL,
        R1M.SEARCH_PARENT,
        TRAIN_WRAPPER,
        EVAL_WRAPPER,
        VISDRONE_PYTHON_PORT,
        VISDRONE_PARITY_AUDIT,
    ]


def source_snapshot() -> dict[str, str]:
    return {str(path): sha256(path) for path in source_paths()}


def preflight() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
    list[dict[str, Any]],
    Any,
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    missing = [str(path) for path in source_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing locked Stage04 inputs: {missing}")
    stage03b_validation = json_load(STAGE03B_VALIDATION)
    if stage03b_validation.get("status") != "PASS":
        raise RuntimeError("Stage03B validation is not PASS.")
    if json_load(STAGE03B_STATE).get("approval_ready") is not True:
        raise RuntimeError("Stage03B is not approval-ready.")
    if sha256(R1M.SEARCH_PARENT) != SEARCH_PARENT_SHA:
        raise RuntimeError("SearchParent SHA256 does not match the approved lock.")
    unique_rows = csv_read(P0_UNIQUE)
    membership_rows = csv_read(P0_MEMBERSHIP)
    if len(unique_rows) != EXPECTED_UNIQUE:
        raise RuntimeError(f"Expected 35 unique P0 signatures, found {len(unique_rows)}.")
    if len(membership_rows) != EXPECTED_MEMBERSHIPS:
        raise RuntimeError(f"Expected 48 P0 memberships, found {len(membership_rows)}.")
    if len({row["evaluation_id"] for row in unique_rows}) != EXPECTED_UNIQUE:
        raise RuntimeError("P0 evaluation IDs are not unique.")
    if len({row["signature_sha256"] for row in unique_rows}) != EXPECTED_UNIQUE:
        raise RuntimeError("P0 architecture signatures are not unique.")
    if list_count(META_TRAIN) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError("Meta-train5177 count mismatch.")
    if list_count(META_INNER_VAL) != EXPECTED_VAL_IMAGES:
        raise RuntimeError("Meta-inner-val1294 count mismatch.")
    if list_count(OFFICIAL_VAL_LOCK) != 548:
        raise RuntimeError("Official val548 lock count mismatch.")
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    if int(data.get("nc", -1)) != 10 or data.get("official_val_used") is not False:
        raise RuntimeError("Stage01 data YAML lock failed.")
    gene_payload = json_load(GENE_JSON)
    groups = gene_payload["groups"]
    if len(groups) != 35:
        raise RuntimeError("Locked R1 gene space does not contain 35 groups.")
    group_ids = [row["group_id"] for row in groups]
    source_artifact = R1M.load_source_model(R1M.SEARCH_PARENT, "stage04_parent")
    complexity_model = R1M.ExactComplexityModel(source_artifact.model, groups)
    taylor_bank, taylor_manifest = R1M.taylor_scope_bank(groups)
    if taylor_manifest["checkpoint_sha256"] != SEARCH_PARENT_SHA:
        raise RuntimeError("Taylor ranking is not linked to the approved SearchParent.")
    evaluation_schema = json_load(EVALUATION_SCHEMA)
    if evaluation_schema.get("status") != "PASS":
        raise RuntimeError("Global evaluation cache schema is not PASS.")
    for unique in unique_rows:
        widths = tuple(int(unique[key]) for key in group_ids)
        if R1M.signature_sha(group_ids, widths) != unique["signature_sha256"]:
            raise RuntimeError(
                f"Signature mismatch for unique row {unique['unique_signature_index']}."
            )
        metrics = complexity_model.calculate(dict(zip(group_ids, widths)))
        if int(metrics["params"]) != int(unique["fused_params"]) or not math.isclose(
            float(metrics["flops_g"]),
            float(unique["fused_flops_g"]),
            rel_tol=0.0,
            abs_tol=FLOPS_TOLERANCE_G,
        ):
            raise RuntimeError(
                f"Locked complexity mismatch for {unique['signature_sha256']}."
            )
    before = source_snapshot()
    preflight_payload = {
        "created_utc": utc_now(),
        "approval_token_consumed": "APPROVE_STAGE03B_SEARCH_INITIALIZATION",
        "stage03b_validation": "PASS",
        "searchparent_sha256": sha256(R1M.SEARCH_PARENT),
        "unique_physical_evaluations": len(unique_rows),
        "replicate_memberships": len(membership_rows),
        "meta_train_count": list_count(META_TRAIN),
        "meta_inner_val_count": list_count(META_INNER_VAL),
        "official_val_locked_count": list_count(OFFICIAL_VAL_LOCK),
        "nc": data["nc"],
        "official_val_used": data["official_val_used"],
        "free_disk_gib": disk_free_gib(),
        "kd_enabled": False,
        "search_offspring_enabled": False,
        "status": "PASS",
    }
    json_write(OUTPUT / "stage04_preflight.json", preflight_payload)
    return (
        unique_rows,
        membership_rows,
        gene_payload,
        groups,
        complexity_model,
        taylor_bank,
        before,
    )


def evaluation_directory(unique: dict[str, str]) -> Path:
    index = int(unique["unique_signature_index"])
    return PHYSICAL_ROOT / f"E{index:03d}_{unique['evaluation_id'][:10]}"


def role_priority(unique: dict[str, str]) -> tuple[int, int]:
    roles = unique["roles"].lower()
    if "exact_targetmanual" in roles:
        return (0, int(unique["unique_signature_index"]))
    if "exact_stage6" in roles:
        return (1, int(unique["unique_signature_index"]))
    return (2, int(unique["unique_signature_index"]))


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
                "taylor_ranking_sha256": sha256(TAYLOR_RANKING),
                "status": "PASS",
            }
        )
    csv_write(path, rows)


def safe_remove(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    expected_parent = parent.resolve()
    if resolved.parent != expected_parent:
        raise RuntimeError(f"Refusing unsafe removal outside {expected_parent}: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def reconstruct_raw(
    unique: dict[str, str],
    groups: list[dict[str, Any]],
    complexity_model,
    taylor_bank: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from pruning_research.pruning.structured_pruner import apply_structured_pruning

    eval_dir = evaluation_directory(unique)
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_checkpoint = eval_dir / "raw_checkpoint.pt"
    group_ids = [row["group_id"] for row in groups]
    widths = tuple(int(unique[key]) for key in group_ids)
    expected_signature = R1M.signature_sha(group_ids, widths)
    if expected_signature != unique["signature_sha256"]:
        raise RuntimeError("Raw reconstruction signature mismatch.")
    for group, width in zip(groups, widths):
        if width not in R1M.parse_allowed(group["allowed_widths"]):
            raise RuntimeError(f"Disallowed width {width} for {group['group_id']}.")
    analytic = complexity_model.calculate(dict(zip(group_ids, widths)))
    if raw_checkpoint.exists():
        physical = R1M.physical_complexity(raw_checkpoint)
        reuse = True
        result_validation = {
            "in_memory_forward": {"forward_ok": True, "reused_checkpoint": True},
            "reloaded_forward": {"forward_ok": physical["forward_pass"]},
        }
    else:
        scratch_root = OUTPUT / "_r"
        scratch = scratch_root / f"E{int(unique['unique_signature_index']):03d}"
        if scratch.exists():
            safe_remove(scratch, scratch_root)
        plan_path = scratch / "p.json"
        R1M.json_write(
            plan_path,
            R1M.build_pruning_plan(
                f"Stage04-{unique['evaluation_id']}",
                groups,
                widths,
                taylor_bank,
            ),
        )
        result = apply_structured_pruning(
            weights=str(R1M.SEARCH_PARENT),
            yolov5_root=str(R1M.YOLOV5_ROOT),
            plan_json=str(plan_path),
            profile_dir=str(R1M.PROFILE_DIR),
            output_dir=str(scratch / "r"),
            weights_output_dir=str(scratch / "w"),
            model_id="s4p0",
            pruned_model_id=f"s4_{int(unique['unique_signature_index']):03d}",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            validate=True,
            validate_imgsz=IMGSZ,
        )
        shutil.copy2(Path(result["pruned_weights"]), raw_checkpoint)
        shutil.copy2(plan_path, eval_dir / "pruning_plan.json")
        result_validation = result["validation"]
        physical = R1M.physical_complexity(raw_checkpoint)
        safe_remove(scratch, scratch_root)
        reuse = False
    write_channel_map(eval_dir / "channel_index_map.csv", groups, widths, taylor_bank)
    audit = {
        "created_utc": utc_now(),
        "evaluation_id": unique["evaluation_id"],
        "architecture_signature_sha256": unique["signature_sha256"],
        "raw_checkpoint": str(raw_checkpoint),
        "raw_checkpoint_sha256": sha256(raw_checkpoint),
        "raw_checkpoint_reused": reuse,
        "searchparent": str(R1M.SEARCH_PARENT),
        "searchparent_sha256": sha256(R1M.SEARCH_PARENT),
        "approved_taylor_ranking": str(TAYLOR_RANKING),
        "approved_taylor_ranking_sha256": sha256(TAYLOR_RANKING),
        "dependency_graph": str(DEPENDENCY_GRAPH),
        "dependency_graph_sha256": sha256(DEPENDENCY_GRAPH),
        "fused_params_expected": int(unique["fused_params"]),
        "fused_params_actual": int(physical["fused_params"]),
        "fused_flops_g_expected": float(unique["fused_flops_g"]),
        "fused_flops_g_actual": float(physical["flops_g"]),
        "nc": physical["nc"],
        "detect_scales": physical["detect_scales"],
        "nonfinite_values": physical["nonfinite_values"],
        "in_memory_forward": result_validation["in_memory_forward"],
        "reloaded_forward": result_validation["reloaded_forward"],
        "widths": dict(zip(group_ids, widths)),
    }
    audit["params_match"] = (
        audit["fused_params_expected"] == audit["fused_params_actual"]
        == int(analytic["params"])
    )
    audit["flops_match"] = math.isclose(
        audit["fused_flops_g_expected"],
        audit["fused_flops_g_actual"],
        rel_tol=0.0,
        abs_tol=FLOPS_TOLERANCE_G,
    ) and math.isclose(
        audit["fused_flops_g_expected"],
        float(analytic["flops_g"]),
        rel_tol=0.0,
        abs_tol=FLOPS_TOLERANCE_G,
    )
    audit["status"] = (
        "PASS"
        if audit["params_match"]
        and audit["flops_match"]
        and audit["nc"] == 10
        and audit["detect_scales"] == 3
        and audit["nonfinite_values"] == 0
        and bool(audit["reloaded_forward"]["forward_ok"])
        else "FAIL"
    )
    json_write(eval_dir / "raw_reconstruction_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Raw reconstruction failed for {unique['evaluation_id']}.")
    return audit


def training_command(eval_dir: Path, raw_checkpoint: Path) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN_WRAPPER),
        "--img",
        str(IMGSZ),
        "--batch-size",
        str(BATCH_SIZE),
        "--epochs",
        str(RECOVERY_EPOCHS),
        "--data",
        str(DATA_YAML),
        "--weights",
        str(raw_checkpoint),
        "--hyp",
        str(HYP_YAML),
        "--optimizer",
        "SGD",
        "--seed",
        str(TRAINING_SEED),
        "--max-det",
        "500",
        "--val-iou-thres",
        "0.60",
        "--device",
        "0",
        "--workers",
        "4",
        "--patience",
        "300",
        "--save-period",
        "-1",
        "--noplots",
        "--project",
        str(PHYSICAL_ROOT),
        "--name",
        eval_dir.name,
        "--exist-ok",
    ]


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_logged(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    label: str,
) -> tuple[int, float]:
    start = time.time()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle, (
        stderr_path.open("w", encoding="utf-8", errors="replace")
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=os.environ.copy(),
        )
        while process.poll() is None:
            elapsed = time.time() - start
            print(
                f"[{utc_now()}] {label} running; elapsed={elapsed / 60.0:.1f} min",
                flush=True,
            )
            time.sleep(60)
    return int(process.returncode), time.time() - start


def results_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return len(pd.read_csv(path))


def training_complete(eval_dir: Path) -> bool:
    return (
        (eval_dir / "weights" / "best.pt").is_file()
        and (eval_dir / "weights" / "last.pt").is_file()
        and results_rows(eval_dir / "results.csv") == RECOVERY_EPOCHS
    )


def clear_partial_training(eval_dir: Path) -> None:
    removable_dirs = ["weights"]
    removable_files = [
        "results.csv",
        "opt.yaml",
        "hyp.yaml",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]
    for name in removable_dirs:
        path = eval_dir / name
        if path.exists():
            safe_remove(path, eval_dir)
    for name in removable_files:
        path = eval_dir / name
        if path.exists():
            safe_remove(path, eval_dir)
    for path in eval_dir.glob("events.out.tfevents.*"):
        safe_remove(path, eval_dir)


def transient_failure(stdout_path: Path, stderr_path: Path) -> bool:
    text = ""
    for path in (stdout_path, stderr_path):
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="replace").lower()
    tokens = (
        "cuda out of memory",
        "cuda error",
        "cublas",
        "cudnn",
        "device-side assert",
        "system interruption",
        "connection reset",
        "input/output error",
        "permissionerror",
        "temporarily unavailable",
    )
    return any(token in text for token in tokens)


def recovered_training_execution(eval_dir: Path) -> dict[str, Any]:
    path = eval_dir / "training_execution.json"
    if path.is_file():
        return json_load(path)
    logs = sorted(eval_dir.glob("train_attempt_*_*.log"))
    if not logs:
        return {
            "training_reused": True,
            "training_duration_seconds": None,
            "duration_source": "NOT_RECORDED",
            "attempts": [],
        }
    start = min(log.stat().st_ctime for log in logs)
    end = max(log.stat().st_mtime for log in logs)
    payload = {
        "created_utc": utc_now(),
        "training_reused": True,
        "training_duration_seconds": max(0.0, end - start),
        "duration_source": "RECOVERED_FROM_TRAIN_LOG_FILE_TIMESTAMPS",
        "attempts": [],
    }
    json_write(path, payload)
    return payload


def train_one(unique: dict[str, str], raw_audit: dict[str, Any]) -> dict[str, Any]:
    eval_dir = evaluation_directory(unique)
    raw_checkpoint = Path(raw_audit["raw_checkpoint"])
    command = training_command(eval_dir, raw_checkpoint)
    text_write(eval_dir / "train_command.txt", command_text(command) + "\n")
    attempts: list[dict[str, Any]] = []
    if training_complete(eval_dir):
        return recovered_training_execution(eval_dir)
    if (eval_dir / "results.csv").exists() or (eval_dir / "weights").exists():
        clear_partial_training(eval_dir)
    for attempt in (1, 2):
        stdout_path = eval_dir / f"train_attempt_{attempt}_stdout.log"
        stderr_path = eval_dir / f"train_attempt_{attempt}_stderr.log"
        code, duration = run_logged(
            command,
            stdout_path,
            stderr_path,
            f"{eval_dir.name} Recovery-8 attempt {attempt}",
        )
        attempts.append(
            {
                "attempt": attempt,
                "return_code": code,
                "duration_seconds": duration,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "transient_failure": transient_failure(stdout_path, stderr_path)
                if code
                else False,
            }
        )
        if code == 0 and training_complete(eval_dir):
            shutil.copy2(stdout_path, eval_dir / "stdout.log")
            shutil.copy2(stderr_path, eval_dir / "stderr.log")
            payload = {
                "created_utc": utc_now(),
                "training_reused": False,
                "training_duration_seconds": duration,
                "duration_source": "SUBPROCESS_WALL_CLOCK",
                "attempts": attempts,
            }
            json_write(eval_dir / "training_execution.json", payload)
            return payload
        if attempt == 1 and attempts[-1]["transient_failure"]:
            clear_partial_training(eval_dir)
            continue
        raise RuntimeError(
            f"Recovery-8 failed for {unique['evaluation_id']} on attempt {attempt}."
        )
    raise AssertionError("Unreachable training retry state.")


def explicit_evaluation_complete(eval_dir: Path) -> bool:
    metrics_path = eval_dir / "explicit_inner_val_metrics.json"
    if not metrics_path.is_file():
        return False
    metrics = json_load(metrics_path)
    return (
        metrics.get("status") == "PASS"
        and int(metrics.get("image_count", -1)) == EXPECTED_VAL_IMAGES
        and metrics.get("official_val548_accessed") is False
        and metrics.get("kd_enabled") is False
    )


def evaluate_one(unique: dict[str, str]) -> dict[str, Any]:
    eval_dir = evaluation_directory(unique)
    command = [
        str(PYTHON),
        str(EVAL_WRAPPER),
        "--run-dir",
        str(eval_dir),
        "--evaluation-id",
        unique["evaluation_id"],
        "--signature-sha256",
        unique["signature_sha256"],
    ]
    text_write(eval_dir / "explicit_eval_command.txt", command_text(command) + "\n")
    if explicit_evaluation_complete(eval_dir):
        return {"evaluation_reused": True, "evaluation_duration_seconds": None}
    validation_root = eval_dir / "explicit_inner_val_predictions"
    if validation_root.exists():
        safe_remove(validation_root, eval_dir)
    metrics_path = eval_dir / "explicit_inner_val_metrics.json"
    if metrics_path.exists():
        safe_remove(metrics_path, eval_dir)
    code, duration = run_logged(
        command,
        eval_dir / "explicit_eval_stdout.log",
        eval_dir / "explicit_eval_stderr.log",
        f"{eval_dir.name} explicit FP32 inner-val1294",
    )
    if code != 0 or not explicit_evaluation_complete(eval_dir):
        raise RuntimeError(f"Explicit evaluation failed for {unique['evaluation_id']}.")
    return {"evaluation_reused": False, "evaluation_duration_seconds": duration}


def update_metrics_durations(
    eval_dir: Path,
    training_result: dict[str, Any],
    evaluation_result: dict[str, Any],
) -> dict[str, Any]:
    path = eval_dir / "explicit_inner_val_metrics.json"
    metrics = json_load(path)
    metrics["training_duration_seconds"] = training_result["training_duration_seconds"]
    metrics["explicit_evaluation_duration_seconds"] = evaluation_result[
        "evaluation_duration_seconds"
    ]
    metrics["training_reused"] = training_result["training_reused"]
    metrics["explicit_evaluation_reused"] = evaluation_result["evaluation_reused"]
    json_write(path, metrics)
    return metrics


def checkpoint_hashes(eval_dir: Path) -> dict[str, Any]:
    paths = {
        "raw_checkpoint": eval_dir / "raw_checkpoint.pt",
        "best_checkpoint": eval_dir / "weights" / "best.pt",
        "last_checkpoint": eval_dir / "weights" / "last.pt",
    }
    payload: dict[str, Any] = {
        "created_utc": utc_now(),
    }
    payload.update(
        {
            key: {
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
            for key, path in paths.items()
        }
    )
    json_write(eval_dir / "checkpoint_hashes.json", payload)
    return payload


def complete_job(
    unique: dict[str, str],
    raw_audit: dict[str, Any],
    training_result: dict[str, Any],
    evaluation_result: dict[str, Any],
    metrics: dict[str, Any],
    hashes: dict[str, Any],
) -> None:
    eval_dir = evaluation_directory(unique)
    manifest = {
        "created_utc": utc_now(),
        "stage": "Stage04 Shared P0 Fidelity-1",
        "unique_signature_index": int(unique["unique_signature_index"]),
        "evaluation_id": unique["evaluation_id"],
        "signature_sha256": unique["signature_sha256"],
        "roles": unique["roles"],
        "replicates": unique["replicates"],
        "membership_count": int(unique["membership_count"]),
        "protocol": {
            "source": "fresh raw pruned checkpoint from SearchParent",
            "recovery_epochs": RECOVERY_EPOCHS,
            "kd": False,
            "training_seed": TRAINING_SEED,
            "train_images": EXPECTED_TRAIN_IMAGES,
            "validation_images": EXPECTED_VAL_IMAGES,
            "imgsz": IMGSZ,
            "batch_size": BATCH_SIZE,
            "optimizer": "SGD",
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "fresh_ema": True,
            "amp": True,
            "ema": True,
            "multi_scale": False,
            "tta": False,
            "best_pt_selection": "meta-inner-val1294 YOLOv5 fitness",
            "explicit_evaluation": "FP32 on all 1294 inner-validation images",
        },
        "raw_reconstruction": raw_audit,
        "training": training_result,
        "explicit_evaluation": evaluation_result,
        "metrics_file": str(eval_dir / "explicit_inner_val_metrics.json"),
        "metrics_sha256": sha256(eval_dir / "explicit_inner_val_metrics.json"),
        "checkpoints": hashes,
        "training_rows": results_rows(eval_dir / "results.csv"),
        "official_val548_accessed": metrics["official_val548_accessed"],
        "search_offspring_generated": False,
        "status": "PASS",
    }
    json_write(eval_dir / "run_manifest.json", manifest)
    text_write(
        eval_dir / ".complete",
        "STAGE04_P0_PHYSICAL_EVALUATION=PASS\n"
        f"EVALUATION_ID={unique['evaluation_id']}\n",
    )


def repair_completed_metadata(eval_dir: Path) -> None:
    metrics_path = eval_dir / "explicit_inner_val_metrics.json"
    manifest_path = eval_dir / "run_manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Completed job is missing metrics or manifest: {eval_dir}")
    metrics = json_load(metrics_path)
    if metrics.get("training_duration_seconds") is not None:
        return
    training = recovered_training_execution(eval_dir)
    metrics["training_duration_seconds"] = training["training_duration_seconds"]
    metrics["training_duration_source"] = training["duration_source"]
    json_write(metrics_path, metrics)
    manifest = json_load(manifest_path)
    manifest["training"] = training
    manifest["metrics_sha256"] = sha256(metrics_path)
    manifest["metadata_reconciled_utc"] = utc_now()
    json_write(manifest_path, manifest)


def execute_jobs(
    unique_rows: list[dict[str, str]],
    groups: list[dict[str, Any]],
    complexity_model,
    taylor_bank: dict[str, dict[str, Any]],
    max_jobs: int | None,
) -> int:
    PHYSICAL_ROOT.mkdir(parents=True, exist_ok=True)
    completed_before = sum(
        1 for row in unique_rows if (evaluation_directory(row) / ".complete").is_file()
    )
    disk_audit("before_stage04_or_resume", completed_before)
    executed = 0
    for unique in sorted(unique_rows, key=role_priority):
        eval_dir = evaluation_directory(unique)
        if (eval_dir / ".complete").is_file():
            repair_completed_metadata(eval_dir)
            print(
                f"REUSE_COMPLETE {eval_dir.name} {unique['evaluation_id']}",
                flush=True,
            )
            continue
        if max_jobs is not None and executed >= max_jobs:
            break
        free_gib = disk_free_gib()
        if free_gib < DISK_STOP_GIB:
            disk_audit("disk_stop_before_new_job", completed_before + executed)
            raise RuntimeError(
                f"Free disk {free_gib:.3f} GiB is below the locked 8 GiB stop threshold."
            )
        print(
            f"START_JOB {eval_dir.name} roles={unique['roles']} "
            f"free_disk_gib={free_gib:.3f}",
            flush=True,
        )
        raw_audit = reconstruct_raw(unique, groups, complexity_model, taylor_bank)
        training_result = train_one(unique, raw_audit)
        evaluation_result = evaluate_one(unique)
        metrics = update_metrics_durations(eval_dir, training_result, evaluation_result)
        hashes = checkpoint_hashes(eval_dir)
        complete_job(
            unique,
            raw_audit,
            training_result,
            evaluation_result,
            metrics,
            hashes,
        )
        executed += 1
        completed = completed_before + executed
        print(
            f"COMPLETE_JOB {eval_dir.name} AP={metrics['AP_0.5_0.95']:.8f} "
            f"APS={metrics['APS']:.8f} Recall={metrics['recall']:.8f}",
            flush=True,
        )
        if completed % 5 == 0:
            disk_audit(f"after_{completed}_completed_jobs", completed)
    total_completed = sum(
        1 for row in unique_rows if (evaluation_directory(row) / ".complete").is_file()
    )
    disk_audit("after_current_execution_batch", total_completed)
    return total_completed


def metric_row(unique: dict[str, str]) -> dict[str, Any]:
    eval_dir = evaluation_directory(unique)
    metrics_path = eval_dir / "explicit_inner_val_metrics.json"
    metrics = json_load(metrics_path)
    hashes = json_load(eval_dir / "checkpoint_hashes.json")
    best_epoch = metrics["best_epoch"]
    return {
        "evaluation_id": unique["evaluation_id"],
        "unique_signature_index": int(unique["unique_signature_index"]),
        "signature_sha256": unique["signature_sha256"],
        "roles": unique["roles"],
        "replicates": unique["replicates"],
        "membership_count": int(unique["membership_count"]),
        "training_seed": TRAINING_SEED,
        "recovery_epochs": RECOVERY_EPOCHS,
        "kd_enabled": False,
        "train_image_count": EXPECTED_TRAIN_IMAGES,
        "validation_image_count": EXPECTED_VAL_IMAGES,
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "AP": float(metrics["AP_0.5_0.95"]),
        "AP50": float(metrics["AP50"]),
        "AP75": float(metrics["AP75"]),
        "APS": float(metrics["APS"]),
        "APM": float(metrics["APM"]),
        "APL": float(metrics["APL"]),
        "params": int(metrics["params"]),
        "flops_g": float(metrics["flops_g"]),
        "weights_size_mib": float(metrics["weights_size_mib"]),
        "best_epoch_zero_based": int(best_epoch["best_epoch_zero_based"]),
        "best_epoch_one_based": int(best_epoch["best_epoch_one_based"]),
        "best_fitness": float(best_epoch["best_fitness"]),
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
        "metrics_file": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "primary_ap_namespace": metrics["primary_ap_namespace"],
        "area_metric_namespace": metrics["area_metric_namespace"],
        "official_val548_accessed": False,
        "status": "PASS",
    }


def target_and_stage6(
    physical_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = [row for row in physical_rows if "exact_targetmanual" in row["roles"].lower()]
    stages = [row for row in physical_rows if "exact_stage6" in row["roles"].lower()]
    if len(targets) != 1 or len(stages) != 1:
        raise RuntimeError(
            f"Expected one TargetManual and one Stage6 anchor; found {len(targets)}/{len(stages)}."
        )
    return targets[0], stages[0]


def nondominated_rank(rows: list[dict[str, Any]]) -> dict[str, int]:
    ids = [row["evaluation_id"] for row in rows]
    by_id = {row["evaluation_id"]: row for row in rows}
    dominates: dict[str, set[str]] = {key: set() for key in ids}
    dominated_count = {key: 0 for key in ids}

    def constrained_dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        if a["feasible"] != b["feasible"]:
            return bool(a["feasible"])
        if not a["feasible"]:
            return float(a["recall_violation"]) < float(b["recall_violation"])
        return (
            float(a["AP"]) >= float(b["AP"])
            and float(a["APS"]) >= float(b["APS"])
            and (
                float(a["AP"]) > float(b["AP"])
                or float(a["APS"]) > float(b["APS"])
            )
        )

    for first in ids:
        for second in ids:
            if first == second:
                continue
            if constrained_dominates(by_id[first], by_id[second]):
                dominates[first].add(second)
            elif constrained_dominates(by_id[second], by_id[first]):
                dominated_count[first] += 1
    ranks: dict[str, int] = {}
    front = sorted(key for key in ids if dominated_count[key] == 0)
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
    if len(ranks) != len(ids):
        raise RuntimeError("Constrained nondominated sorting did not rank every row.")
    return ranks


def crowding_distance(rows: list[dict[str, Any]], ranks: dict[str, int]) -> dict[str, float]:
    output = {row["evaluation_id"]: 0.0 for row in rows}
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_rank.setdefault(ranks[row["evaluation_id"]], []).append(row)
    for front in by_rank.values():
        if len(front) <= 2:
            for row in front:
                output[row["evaluation_id"]] = math.inf
            continue
        for objective in ("AP", "APS"):
            ordered = sorted(front, key=lambda row: (float(row[objective]), row["evaluation_id"]))
            output[ordered[0]["evaluation_id"]] = math.inf
            output[ordered[-1]["evaluation_id"]] = math.inf
            minimum = float(ordered[0][objective])
            maximum = float(ordered[-1][objective])
            if maximum == minimum:
                continue
            for index in range(1, len(ordered) - 1):
                key = ordered[index]["evaluation_id"]
                if math.isinf(output[key]):
                    continue
                previous_value = float(ordered[index - 1][objective])
                next_value = float(ordered[index + 1][objective])
                output[key] += (next_value - previous_value) / (maximum - minimum)
    return output


def create_aggregate_outputs(
    unique_rows: list[dict[str, str]],
    membership_rows: list[dict[str, str]],
) -> dict[str, Any]:
    physical_rows = [metric_row(row) for row in unique_rows]
    if len(physical_rows) != EXPECTED_UNIQUE:
        raise RuntimeError("Physical metric-row count mismatch.")
    target, stage6 = target_and_stage6(physical_rows)
    recall_floor = float(target["recall"]) - 0.002
    recall_payload = {
        "created_utc": utc_now(),
        "definition": "Recall(TargetManual-B6-reference, Recovery-8, seed42) - 0.002",
        "recall_namespace": "YOLOv5 explicit FP32 evaluation on meta-inner-val1294",
        "targetmanual_evaluation_id": target["evaluation_id"],
        "targetmanual_recall": target["recall"],
        "offset": 0.002,
        "recall_floor_f1": recall_floor,
        "source_metrics_file": target["metrics_file"],
        "source_metrics_sha256": target["metrics_sha256"],
        "status": "FROZEN",
    }
    json_write(OUTPUT / "recall_floor_f1.json", recall_payload)
    json_write(OUTPUT / "targetmanual_f1_metrics.json", target)
    json_write(OUTPUT / "stage6_anchor_f1_metrics.json", stage6)

    physical_by_id = {row["evaluation_id"]: row for row in physical_rows}
    replicate_rows = []
    for membership in membership_rows:
        metric = physical_by_id[membership["evaluation_id"]]
        row = {
            "replicate": int(membership["replicate_base"]),
            "p0_sub_seed": int(membership["p0_sub_seed"]),
            "p0_slot": int(membership["p0_slot"]),
            "signature_id": membership["signature_id"],
            "p0_role": membership["p0_role"],
            "evaluation_id": membership["evaluation_id"],
            "signature_sha256": membership["signature_sha256"],
            **{
                key: metric[key]
                for key in (
                    "precision",
                    "recall",
                    "f1",
                    "AP",
                    "AP50",
                    "AP75",
                    "APS",
                    "APM",
                    "APL",
                    "params",
                    "flops_g",
                    "weights_size_mib",
                    "best_epoch_one_based",
                    "best_checkpoint_sha256",
                    "metrics_sha256",
                )
            },
        }
        row["recall_floor_f1"] = recall_floor
        row["recall_violation"] = max(0.0, recall_floor - float(row["recall"]))
        row["feasible"] = row["recall_violation"] <= 1e-15
        row["status"] = "PASS"
        replicate_rows.append(row)
    if len(replicate_rows) != EXPECTED_MEMBERSHIPS:
        raise RuntimeError("Replicate metric-row count mismatch.")

    method_rows = []
    pareto_output: dict[str, Any] = {
        "created_utc": utc_now(),
        "objectives": ["maximize AP", "maximize APS"],
        "constraint": f"Recall >= {recall_floor:.17g}",
        "offspring_generated": False,
        "replicates": {},
    }
    ap_only_output: dict[str, Any] = {
        "created_utc": utc_now(),
        "ranking_rule": [
            "feasible before infeasible",
            "higher AP",
            "smaller Recall violation among infeasible",
            "lower FLOPs",
            "lower Params",
            "deterministic architecture signature",
        ],
        "APS_AP50_AP75_APM_APL_used_for_ranking": False,
        "replicates": {},
    }
    for replicate in (1042, 2042, 3042):
        members = sorted(
            [row for row in replicate_rows if row["replicate"] == replicate],
            key=lambda row: row["p0_slot"],
        )
        if len(members) != 16:
            raise RuntimeError(f"Replicate {replicate} does not contain 16 P0 rows.")
        ranks = nondominated_rank(members)
        crowding = crowding_distance(members, ranks)
        pareto_output["replicates"][str(replicate)] = [
            {
                "p0_slot": row["p0_slot"],
                "evaluation_id": row["evaluation_id"],
                "signature_sha256": row["signature_sha256"],
                "feasible": row["feasible"],
                "recall_violation": row["recall_violation"],
                "AP": row["AP"],
                "APS": row["APS"],
                "constrained_pareto_rank": ranks[row["evaluation_id"]],
                "crowding_distance": "INF"
                if math.isinf(crowding[row["evaluation_id"]])
                else crowding[row["evaluation_id"]],
            }
            for row in members
        ]
        ap_sorted = sorted(
            members,
            key=lambda row: (
                not bool(row["feasible"]),
                -float(row["AP"]) if row["feasible"] else 0.0,
                float(row["recall_violation"]) if not row["feasible"] else 0.0,
                -float(row["AP"]) if not row["feasible"] else 0.0,
                float(row["flops_g"]),
                int(row["params"]),
                row["signature_sha256"],
            ),
        )
        ap_only_output["replicates"][str(replicate)] = [
            {
                "rank": index,
                "p0_slot": row["p0_slot"],
                "evaluation_id": row["evaluation_id"],
                "signature_sha256": row["signature_sha256"],
                "feasible": row["feasible"],
                "recall_violation": row["recall_violation"],
                "AP": row["AP"],
                "APS_analysis_only": row["APS"],
                "flops_g": row["flops_g"],
                "params": row["params"],
            }
            for index, row in enumerate(ap_sorted, start=1)
        ]
        for method in METHODS:
            for row in members:
                canonical = "\n".join(
                    [
                        method,
                        str(replicate),
                        "0",
                        str(row["p0_slot"]),
                        row["evaluation_id"],
                    ]
                )
                record = {
                    "search_record_id": hash_text(canonical),
                    "method": method,
                    "replicate": replicate,
                    "generation": 0,
                    "population_index": row["p0_slot"],
                    "evaluation_id": row["evaluation_id"],
                    "signature_sha256": row["signature_sha256"],
                    "p0_role": row["p0_role"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "AP": row["AP"],
                    "AP50": row["AP50"],
                    "AP75": row["AP75"],
                    "APS": row["APS"],
                    "APM": row["APM"],
                    "APL": row["APL"],
                    "params": row["params"],
                    "flops_g": row["flops_g"],
                    "recall_floor_f1": recall_floor,
                    "recall_violation": row["recall_violation"],
                    "feasible": row["feasible"],
                    "constrained_pareto_rank": ranks[row["evaluation_id"]]
                    if method == "NSGA-II AP+APS"
                    else "",
                    "crowding_distance": (
                        "INF"
                        if math.isinf(crowding[row["evaluation_id"]])
                        else crowding[row["evaluation_id"]]
                    )
                    if method == "NSGA-II AP+APS"
                    else "",
                    "ap_only_rank": (
                        next(
                            entry["rank"]
                            for entry in ap_only_output["replicates"][str(replicate)]
                            if entry["evaluation_id"] == row["evaluation_id"]
                        )
                        if method == "Evolutionary AP-only"
                        else ""
                    ),
                    "AP_only_objective": method == "Evolutionary AP-only",
                    "APS_used_for_AP_only_ranking": False,
                    "physical_checkpoint_shared": True,
                    "best_checkpoint_sha256": row["best_checkpoint_sha256"],
                    "metrics_sha256": row["metrics_sha256"],
                    "offspring_generated": False,
                    "status": "PASS",
                }
                method_rows.append(record)
    if len(method_rows) != EXPECTED_METHOD_RECORDS:
        raise RuntimeError("Method-level generation-0 record count mismatch.")
    if len({row["search_record_id"] for row in method_rows}) != EXPECTED_METHOD_RECORDS:
        raise RuntimeError("Method-level search record IDs are not unique.")

    csv_write(OUTPUT / "shared_evaluation_cache.csv", physical_rows)
    csv_write(OUTPUT / "physical_evaluation_manifest.csv", physical_rows)
    csv_write(OUTPUT / "p0_replicate_metrics.csv", replicate_rows)
    csv_write(OUTPUT / "p0_method_records.csv", method_rows)
    json_write(OUTPUT / "constrained_pareto_initialization.json", pareto_output)
    json_write(OUTPUT / "ap_only_initial_ranking.json", ap_only_output)
    random_payload = {
        "created_utc": utc_now(),
        "method": "Warm-started budget-constrained Random Search",
        "shared_p0_only": True,
        "new_random_samples_generated": 0,
        "offspring_generated": False,
        "replicates": {
            str(replicate): {
                "generation": 0,
                "p0_memberships": 16,
                "evaluation_ids": [
                    row["evaluation_id"]
                    for row in replicate_rows
                    if row["replicate"] == replicate
                ],
            }
            for replicate in (1042, 2042, 3042)
        },
        "status": "PASS",
    }
    json_write(OUTPUT / "random_initialization_manifest.json", random_payload)
    return {
        "physical_rows": physical_rows,
        "replicate_rows": replicate_rows,
        "method_rows": method_rows,
        "recall_floor": recall_payload,
    }


def build_output_manifest() -> list[dict[str, Any]]:
    rows = []
    excluded_parts = {"explicit_inner_val_predictions"}
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path.name == "stage04_output_manifest.json":
            continue
        if any(part in excluded_parts for part in path.parts):
            scientific_role = "explicit_prediction_evaluation_support"
        elif path.suffix.lower() == ".pt":
            scientific_role = "checkpoint"
        elif path.suffix.lower() in {".csv", ".json", ".yaml", ".yml"}:
            scientific_role = "evidence_or_manifest"
        elif path.suffix.lower() in {".log", ".txt", ".md"}:
            scientific_role = "command_log_or_report"
        else:
            scientific_role = "supporting_artifact"
        rows.append(
            {
                "relative_path": str(path.relative_to(OUTPUT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "last_write_time_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "scientific_role": scientific_role,
            }
        )
    json_write(
        OUTPUT / "stage04_output_manifest.json",
        {
            "created_utc": utc_now(),
            "output_root": str(OUTPUT),
            "file_count": len(rows),
            "files": rows,
        },
    )
    return rows


def finalize(
    unique_rows: list[dict[str, str]],
    membership_rows: list[dict[str, str]],
    source_before: dict[str, str],
) -> dict[str, Any]:
    aggregate = create_aggregate_outputs(unique_rows, membership_rows)
    disk_audit("after_stage04", EXPECTED_UNIQUE)
    source_after = source_snapshot()
    source_unchanged = source_before == source_after
    checks = {
        "shared_p0_physical_evaluations_35_pass": len(
            aggregate["physical_rows"]
        )
        == EXPECTED_UNIQUE
        and all(row["status"] == "PASS" for row in aggregate["physical_rows"]),
        "p0_replicate_memberships_48_accounted": len(
            aggregate["replicate_rows"]
        )
        == EXPECTED_MEMBERSHIPS,
        "p0_method_records_144_accounted": len(aggregate["method_rows"])
        == EXPECTED_METHOD_RECORDS,
        "targetmanual_f1_pass": (OUTPUT / "targetmanual_f1_metrics.json").is_file(),
        "stage6_anchor_f1_pass": (OUTPUT / "stage6_anchor_f1_metrics.json").is_file(),
        "recall_floor_f1_frozen": json_load(OUTPUT / "recall_floor_f1.json")[
            "status"
        ]
        == "FROZEN",
        "global_evaluation_cache_pass": len(
            {row["evaluation_id"] for row in aggregate["physical_rows"]}
        )
        == EXPECTED_UNIQUE,
        "explicit_inner_val_evaluations_35_pass": all(
            int(row["validation_image_count"]) == EXPECTED_VAL_IMAGES
            for row in aggregate["physical_rows"]
        ),
        "all_training_rows_equal_8": all(
            results_rows(evaluation_directory(unique) / "results.csv")
            == RECOVERY_EPOCHS
            for unique in unique_rows
        ),
        "official_val548_locked": all(
            row["official_val548_accessed"] is False
            for row in aggregate["physical_rows"]
        ),
        "kd_not_executed": all(
            row["kd_enabled"] is False for row in aggregate["physical_rows"]
        ),
        "search_offspring_not_generated": all(
            row["offspring_generated"] is False for row in aggregate["method_rows"]
        ),
        "stage05_not_started": not (PROJECT / "05_full_fidelity1_search").exists(),
        "source_inputs_unchanged": source_unchanged,
        "disk_threshold_respected": json_load(OUTPUT / "disk_usage_audit.json")[
            "latest_free_gib"
        ]
        >= DISK_STOP_GIB,
    }
    failed = [key for key, value in checks.items() if not value]
    status = "PASS" if not failed else "FAIL"
    validation = {
        "created_utc": utc_now(),
        "stage": "Stage04 Shared P0 Fidelity-1 Evaluation",
        "checks": checks,
        "failed_checks": failed,
        "counts": {
            "physical_evaluations": len(aggregate["physical_rows"]),
            "replicate_memberships": len(aggregate["replicate_rows"]),
            "method_records": len(aggregate["method_rows"]),
            "explicit_inner_val_evaluations": sum(
                int(row["validation_image_count"]) == EXPECTED_VAL_IMAGES
                for row in aggregate["physical_rows"]
            ),
        },
        "status": status,
    }
    json_write(OUTPUT / "stage04_validation.json", validation)
    state = {
        "created_utc": utc_now(),
        "stage": "Stage04",
        "shared_p0_physical_evaluations": status,
        "explicit_inner_val_evaluations": status,
        "recall_floor_f1": "FROZEN" if status == "PASS" else "INVALID",
        "approval_ready": status == "PASS",
        "official_val548_locked": True,
        "kd_executed": False,
        "search_offspring_generated": False,
        "stage05_started": False,
        "required_next_token": "APPROVE_STAGE04_SHARED_P0_F1",
        "status": status,
    }
    json_write(OUTPUT / "stage04_state.json", state)
    report = f"""# Stage04 Shared P0 Fidelity-1 Evaluation

## Scope

- Approval token: `APPROVE_STAGE03B_SEARCH_INITIALIZATION`
- Unique physical Recovery-8 evaluations: {len(aggregate['physical_rows'])}
- Replicate P0 memberships: {len(aggregate['replicate_rows'])}
- Method-level generation-0 records: {len(aggregate['method_rows'])}
- Training pool: meta-train5177
- Explicit evaluation pool: meta-inner-val1294
- Candidate training seed: 42
- KD: disabled
- Search offspring: not generated
- Official val548: locked and not accessed

## Locked Recall Constraint

`Recall_floor_F1 = {aggregate['recall_floor']['targetmanual_recall']:.12f} - 0.002 = {aggregate['recall_floor']['recall_floor_f1']:.12f}`

## Validation

- Status: `{status}`
- Failed checks: {failed}
- Source inputs unchanged: {source_unchanged}
- Stage05 started: false
"""
    text_write(OUTPUT / "stage04_report.md", report)
    build_output_manifest()
    if status == "PASS":
        text_write(
            OUTPUT / ".complete",
            "STAGE04_SHARED_P0_FIDELITY1_EVALUATION=PASS\n",
        )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Execute at most this many incomplete physical jobs, then stop safely.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not train; require all 35 jobs and create aggregate outputs.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (
        unique_rows,
        membership_rows,
        _gene_payload,
        groups,
        complexity_model,
        taylor_bank,
        source_before,
    ) = preflight()
    completed = sum(
        1 for row in unique_rows if (evaluation_directory(row) / ".complete").is_file()
    )
    if not args.finalize_only:
        completed = execute_jobs(
            unique_rows,
            groups,
            complexity_model,
            taylor_bank,
            args.max_jobs,
        )
    if completed != EXPECTED_UNIQUE:
        print(
            f"STAGE04_PROGRESS={completed}/{EXPECTED_UNIQUE}; "
            "aggregate finalization intentionally deferred.",
            flush=True,
        )
        return 0
    validation = finalize(unique_rows, membership_rows, source_before)
    print("STAGE04 SHARED P0 FIDELITY-1 EVALUATION")
    print(
        "SHARED_P0_PHYSICAL_EVALUATIONS="
        + (
            "35_PASS"
            if validation["checks"]["shared_p0_physical_evaluations_35_pass"]
            else "FAIL"
        )
    )
    print(
        "P0_REPLICATE_MEMBERSHIPS="
        + (
            "48_ACCOUNTED"
            if validation["checks"]["p0_replicate_memberships_48_accounted"]
            else "FAIL"
        )
    )
    print(
        "P0_METHOD_RECORDS="
        + (
            "144_ACCOUNTED"
            if validation["checks"]["p0_method_records_144_accounted"]
            else "FAIL"
        )
    )
    print(
        "TARGETMANUAL_F1="
        + ("PASS" if validation["checks"]["targetmanual_f1_pass"] else "FAIL")
    )
    print(
        "STAGE6_ANCHOR_F1="
        + ("PASS" if validation["checks"]["stage6_anchor_f1_pass"] else "FAIL")
    )
    print(
        "RECALL_FLOOR_F1="
        + ("FROZEN" if validation["checks"]["recall_floor_f1_frozen"] else "FAIL")
    )
    print(
        "GLOBAL_EVALUATION_CACHE="
        + ("PASS" if validation["checks"]["global_evaluation_cache_pass"] else "FAIL")
    )
    print(
        "EXPLICIT_INNER_VAL_EVALUATIONS="
        + (
            "35_PASS"
            if validation["checks"]["explicit_inner_val_evaluations_35_pass"]
            else "FAIL"
        )
    )
    print("OFFICIAL_VAL548_LOCKED=YES")
    print("KD_EXECUTED=NO")
    print("SEARCH_OFFSPRING_GENERATED=NO")
    print(f"APPROVAL_READY={'YES' if validation['status'] == 'PASS' else 'NO'}")
    print("STAGE05_STARTED=NO")
    print(f"OUTPUT_ROOT={OUTPUT}")
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
