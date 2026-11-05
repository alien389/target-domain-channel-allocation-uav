from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import csv
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import psutil


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
ROOT = PROJECT / "05_full_fidelity1_search"
SCRIPTS = ROOT / "scripts"
RUNTIME = ROOT / "runtime"
PHYSICAL = ROOT / "physical_evaluations"
METHOD_RECORDS = ROOT / "method_records"
ARCHIVES = ROOT / "generation_archives"

STAGE01 = PROJECT / "01_data_split"
STAGE01A = PROJECT / "01A_protocol_amendment"
STAGE03A = PROJECT / "03a_taylor_dependency_gene_space"
R1_ROOT = STAGE03A / "gene_space_R1"
STAGE03B = PROJECT / "03b_search_initialization"
STAGE04 = PROJECT / "04_shared_p0_fidelity1"
A2 = PROJECT / "protocol_amendment_A2_final_objectives_heldout"

PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))
SEARCH_PARENT_SHA256 = (
    "15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659"
)
RECALL_FLOOR_TEXT = "0.30035872606099606"
RECALL_FLOOR = Decimal(RECALL_FLOOR_TEXT)
PARAMS_LOWER = 2_440_209
PARAMS_UPPER = 2_515_679
FLOPS_RAW_LOWER = 7_618_383_104
FLOPS_RAW_UPPER = 7_854_003_200
FLOPS_G_LOWER_TEXT = "7.618383104"
FLOPS_G_UPPER_TEXT = "7.8540032"
TRAINING_SEED = 42
RECOVERY_EPOCHS = 8
TRAIN_IMAGES = 5177
VAL_IMAGES = 1294
IMGSZ = 640
BATCH_SIZE = 16
WARNING_FREE_GIB = 40.0
HARD_STOP_FREE_GIB = 25.0
EXPECTED_P0_RECORDS = 144
EXPECTED_NEW_RECORDS = 576
EXPECTED_TOTAL_RECORDS = 720
METHODS = (
    "NSGA-II AP+APS",
    "Single-objective evolutionary search (AP-only)",
    "Warm-started constrained Random Search",
)
REPLICATES = (1042, 2042, 3042)
RUN_ORDER = tuple((method, replicate) for method in METHODS for replicate in REPLICATES)
METHOD_STREAMS = {
    ("NSGA-II AP+APS", 1042): 104201,
    ("NSGA-II AP+APS", 2042): 204201,
    ("NSGA-II AP+APS", 3042): 304201,
    ("Single-objective evolutionary search (AP-only)", 1042): 104202,
    ("Single-objective evolutionary search (AP-only)", 2042): 204202,
    ("Single-objective evolutionary search (AP-only)", 3042): 304202,
    ("Warm-started constrained Random Search", 1042): 104203,
    ("Warm-started constrained Random Search", 2042): 204203,
    ("Warm-started constrained Random Search", 3042): 304203,
}
P0_STREAMS = {1042: 104200, 2042: 204200, 3042: 304200}

CAMPAIGN_STATE = RUNTIME / "stage05_campaign_state.json"
CONTROLLER_STATE = RUNTIME / "stage05_controller_state.json"
WORKER_STATE = RUNTIME / "stage05_worker_state.json"
WORKER_PID = RUNTIME / "stage05_worker.pid"
HEARTBEAT = RUNTIME / "stage05_heartbeat.json"
CAMPAIGN_LOCK = RUNTIME / "stage05_campaign.lock"
PAUSE_REQUESTED = RUNTIME / "stage05_pause_requested.flag"
PAUSED_FLAG = RUNTIME / "stage05_paused.flag"
STOP_REASON = RUNTIME / "stage05_stop_reason.json"
STATUS_JSON = RUNTIME / "stage05_status_report.json"
STATUS_MD = RUNTIME / "stage05_status_report.md"

PROTOCOL_LOCK = ROOT / "stage05_protocol_lock.json"
CODE_HASHES = ROOT / "stage05_code_hashes.json"
GLOBAL_CACHE = ROOT / "global_evaluation_cache.csv"
PHYSICAL_MANIFEST = ROOT / "physical_evaluation_manifest.csv"
SEARCH_RECORDS = ROOT / "search_records_all.csv"
PROVENANCE_CSV = ROOT / "generation_and_parent_provenance.csv"
FAILURE_AUDIT = ROOT / "failure_audit.json"
RETENTION_MANIFEST = ROOT / "checkpoint_retention_manifest.csv"
DISK_AUDIT = ROOT / "disk_usage_audit.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def json_default(value: Any) -> Any:
    """Preserve exact decimal values in JSON without a binary-float conversion."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=json_default,
    )


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_bytes(path, text.encode(encoding))


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n",
    )


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(
    path: Path, rows: Iterable[dict[str, Any]], headers: list[str] | None = None
) -> None:
    materialized = list(rows)
    if headers is None:
        headers = []
        for row in materialized:
            for key in row:
                if key not in headers:
                    headers.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_audit_row(path: Path, row: dict[str, Any]) -> None:
    rows = read_csv(path)
    rows.append({key: value for key, value in row.items()})
    atomic_csv(path, rows)


def ensure_directories() -> None:
    for path in (ROOT, SCRIPTS, RUNTIME, PHYSICAL, METHOD_RECORDS, ARCHIVES):
        path.mkdir(parents=True, exist_ok=True)


def free_gib() -> float:
    return shutil.disk_usage(ROOT.anchor).free / (1024**3)


def tree_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        process = psutil.Process(int(pid))
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def process_info(pid: int | None) -> dict[str, Any] | None:
    if not process_alive(pid):
        return None
    process = psutil.Process(int(pid))
    try:
        return {
            "pid": process.pid,
            "name": process.name(),
            "create_time_utc": datetime.fromtimestamp(
                process.create_time(), timezone.utc
            ).isoformat(),
            "command_line": process.cmdline(),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def stage05_worker_processes() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if "stage05_worker.py" not in command:
                continue
            matches.append(
                {
                    "pid": process.info["pid"],
                    "name": process.info["name"],
                    "command_line": command,
                    "create_time_utc": datetime.fromtimestamp(
                        float(process.info["create_time"]), timezone.utc
                    ).isoformat(),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def active_gpu_children() -> list[dict[str, Any]]:
    tokens = (
        "train_structural_recovery_drone.py",
        "evaluate_stage04_inner_val.py",
        "train_drone.py",
        "val.py",
    )
    matches: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if not any(token in command for token in tokens):
                continue
            matches.append(
                {
                    "pid": process.info["pid"],
                    "name": process.info["name"],
                    "command_line": command,
                    "create_time_utc": datetime.fromtimestamp(
                        float(process.info["create_time"]), timezone.utc
                    ).isoformat(),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def flops_full_precision(raw_flops: int) -> str:
    return format(Decimal(int(raw_flops)) / Decimal(1_000_000_000), "f")


def flops_raw_from_text(value: str | float | Decimal) -> int:
    raw = Decimal(str(value)) * Decimal(1_000_000_000)
    integral = raw.to_integral_value()
    if raw != integral:
        raise ValueError(f"FLOPs value is not an exact integer count: {value}")
    return int(integral)


def exact_budget_feasible(params: int, flops_raw: int) -> bool:
    return (
        PARAMS_LOWER <= int(params) <= PARAMS_UPPER
        and FLOPS_RAW_LOWER <= int(flops_raw) <= FLOPS_RAW_UPPER
    )


def recall_feasible(value: str | float | Decimal) -> bool:
    return Decimal(str(value)) >= RECALL_FLOOR


def decimal_equal(first: Any, second: Any) -> bool:
    return Decimal(str(first)) == Decimal(str(second))


def evaluation_id(
    architecture_signature: str, protocol_lock: dict[str, Any]
) -> tuple[str, str]:
    identity = {
        "schema": "stage05_global_physical_evaluation_id_v2",
        "architecture_signature": architecture_signature,
        "raw_parent_checkpoint_hash": protocol_lock["source_hashes"][
            "search_parent"
        ],
        "Taylor_ranking_hash": protocol_lock["source_hashes"]["taylor_ranking"],
        "dependency_manifest_hash": protocol_lock["source_hashes"][
            "dependency_manifest"
        ],
        "data_split_hash": protocol_lock["source_hashes"]["data_split"],
        "training_protocol_hash": protocol_lock["source_hashes"][
            "recovery_protocol"
        ],
        "evaluator_hash": protocol_lock["source_hashes"]["evaluator"],
        "training_seed": TRAINING_SEED,
        "evaluation_precision": "FP32",
        "evaluation_settings_hash": protocol_lock["evaluation_settings_hash"],
    }
    canonical = canonical_json(identity)
    return hash_text(canonical), canonical


def search_record_id(
    method: str,
    replicate: int,
    generation_or_batch: int,
    population_index: int,
    evaluation: str,
) -> str:
    canonical = "\n".join(
        [
            method,
            str(replicate),
            str(generation_or_batch),
            str(population_index),
            evaluation,
        ]
    )
    return hash_text(canonical)


def disk_audit(event: str, campaign_state: dict[str, Any] | None = None) -> None:
    payload = load_json(
        DISK_AUDIT,
        {
            "created_utc": utc_now(),
            "warning_threshold_free_gib": WARNING_FREE_GIB,
            "hard_stop_before_new_job_free_gib": HARD_STOP_FREE_GIB,
            "deleted_temporary_size_bytes": 0,
            "samples": [],
        },
    )
    current_free = free_gib()
    sample = {
        "timestamp_utc": utc_now(),
        "event": event,
        "free_gib": current_free,
        "stage05_retained_size_bytes": tree_size_bytes(ROOT),
        "warning_threshold_status": "WARNING"
        if current_free < WARNING_FREE_GIB
        else "OK",
        "hard_stop_threshold_status": "STOP"
        if current_free < HARD_STOP_FREE_GIB
        else "OK",
    }
    if campaign_state:
        accounting = campaign_state.get("accounting", {})
        sample.update(
            {
                "new_method_records": accounting.get("new_method_records", 0),
                "completed_physical_evaluations": accounting.get(
                    "completed_physical_evaluations", 0
                ),
                "cache_hits": accounting.get("cache_hits", 0),
            }
        )
    payload["samples"].append(sample)
    payload["latest"] = sample
    atomic_json(DISK_AUDIT, payload)


def runtime_durations(cache_rows: list[dict[str, str]]) -> list[float]:
    values: list[float] = []
    for row in cache_rows:
        if str(row.get("source_stage", "")).lower() != "stage05":
            continue
        first = row.get("training_duration_seconds")
        second = row.get("explicit_evaluation_duration_seconds")
        try:
            if first not in ("", None) and second not in ("", None):
                values.append(float(first) + float(second))
        except ValueError:
            continue
    return values


def status_payload() -> dict[str, Any]:
    campaign = load_json(CAMPAIGN_STATE, {})
    worker = load_json(WORKER_STATE, {})
    heartbeat = load_json(HEARTBEAT, {})
    cache_rows = read_csv(GLOBAL_CACHE)
    records = read_csv(SEARCH_RECORDS)
    new_records = [row for row in records if row.get("record_origin") == "STAGE05_NEW"]
    failed = [
        row
        for row in new_records
        if row.get("status") in {"FAILED_SCIENTIFIC", "FAILED_ENGINEERING"}
    ]
    durations = runtime_durations(cache_rows)
    completed_physical = sum(
        row.get("status") == "COMPLETE" and row.get("source_stage") == "Stage05"
        for row in cache_rows
    )
    cache_hits = sum(row.get("cache_hit", "").lower() == "true" for row in new_records)
    remaining_records = max(0, EXPECTED_NEW_RECORDS - len(new_records))
    mean_duration = statistics.mean(durations) if durations else None
    median_duration = statistics.median(durations) if durations else None
    estimated_remaining_physical = campaign.get("accounting", {}).get(
        "estimated_remaining_physical_evaluations"
    )
    if estimated_remaining_physical is None:
        estimated_remaining_physical = remaining_records
    estimated_seconds = (
        mean_duration * estimated_remaining_physical if mean_duration is not None else None
    )
    current_free = free_gib()
    output = {
        "created_utc": utc_now(),
        "campaign_uuid": campaign.get("campaign_uuid"),
        "overall_state": campaign.get("state", "NOT_STARTED"),
        "current_scientific_position": campaign.get("current_position", {}),
        "current_execution_state": worker.get("candidate_state", "NOT_STARTED"),
        "training_progress": {
            "current_epoch": heartbeat.get("current_epoch"),
            "total_epochs": RECOVERY_EPOCHS,
            "training_start_time": worker.get("training_start_time"),
            "latest_heartbeat": heartbeat.get("timestamp_utc"),
            "worker_pid": worker.get("worker_pid"),
            "child_process_pid": worker.get("child_pid"),
            "gpu_job_active": bool(worker.get("child_pid"))
            and process_alive(worker.get("child_pid")),
        },
        "global_accounting": {
            "p0_records_accounted": sum(
                row.get("record_origin") == "STAGE04_P0_REUSE" for row in records
            ),
            "p0_records_expected": EXPECTED_P0_RECORDS,
            "new_method_records_accounted": len(new_records),
            "new_method_records_expected": EXPECTED_NEW_RECORDS,
            "total_f1_records_accounted": len(records),
            "total_f1_records_expected": EXPECTED_TOTAL_RECORDS,
            "completed_new_method_records": sum(
                row.get("status") == "COMPLETE" for row in new_records
            ),
            "failed_method_records": len(failed),
            "unique_new_signatures": len(
                {
                    row.get("signature_sha256")
                    for row in new_records
                    if row.get("signature_sha256")
                }
            ),
            "completed_physical_evaluations": completed_physical,
            "cache_hits": cache_hits,
            "duplicate_rejections": campaign.get("accounting", {}).get(
                "duplicate_rejections", 0
            ),
            "scientific_failures": sum(
                row.get("status") == "FAILED_SCIENTIFIC" for row in new_records
            ),
            "engineering_retries": campaign.get("accounting", {}).get(
                "engineering_retries", 0
            ),
        },
        "per_run_progress": campaign.get("run_states", []),
        "time": {
            "campaign_started_utc": campaign.get("started_utc"),
            "total_active_wall_clock_seconds": campaign.get(
                "active_wall_clock_seconds", 0.0
            ),
            "paused_wall_clock_seconds": campaign.get(
                "paused_wall_clock_seconds", 0.0
            ),
            "mean_completed_physical_evaluation_seconds": mean_duration,
            "median_completed_physical_evaluation_seconds": median_duration,
            "estimated_remaining_physical_evaluations": estimated_remaining_physical,
            "estimated_remaining_active_seconds_approximate": estimated_seconds,
        },
        "storage": {
            "current_free_disk_gib": current_free,
            "stage05_retained_size_bytes": tree_size_bytes(ROOT),
            "deleted_temporary_size_bytes": load_json(DISK_AUDIT, {}).get(
                "deleted_temporary_size_bytes", 0
            ),
            "warning_threshold_status": "WARNING"
            if current_free < WARNING_FREE_GIB
            else "OK",
            "hard_stop_threshold_status": "STOP"
            if current_free < HARD_STOP_FREE_GIB
            else "OK",
        },
        "scientific_locks": {
            "recall_floor_f1": RECALL_FLOOR_TEXT,
            "val548_accessed": False,
            "heldout1610_accessed": False,
            "kd_executed": False,
            "fidelity2_started": False,
            "jetson_profiling_started": False,
        },
        "worker_process_alive": process_alive(worker.get("worker_pid")),
        "stage05_worker_process_count": len(stage05_worker_processes()),
        "active_gpu_child_count": len(active_gpu_children()),
    }
    return output


def status_markdown(payload: dict[str, Any]) -> str:
    accounting = payload["global_accounting"]
    position = payload["current_scientific_position"]
    training = payload["training_progress"]
    return f"""# Stage05 Status

- State: `{payload['overall_state']}`
- Method: `{position.get('method', 'NOT_STARTED')}`
- Replicate: `{position.get('replicate', 'NOT_STARTED')}`
- Generation/batch: `{position.get('generation_or_batch', 'NOT_STARTED')}`
- Candidate: `{position.get('candidate_index', 'NOT_STARTED')}`
- Candidate state: `{payload['current_execution_state']}`
- Worker PID: `{training.get('worker_pid')}`
- GPU child PID: `{training.get('child_process_pid')}`
- Current epoch: `{training.get('current_epoch')}/{training.get('total_epochs')}`
- P0 records: `{accounting['p0_records_accounted']}/{accounting['p0_records_expected']}`
- New records: `{accounting['new_method_records_accounted']}/{accounting['new_method_records_expected']}`
- Total F1 records: `{accounting['total_f1_records_accounted']}/{accounting['total_f1_records_expected']}`
- Completed Stage05 physical evaluations: `{accounting['completed_physical_evaluations']}`
- Cache hits: `{accounting['cache_hits']}`
- Free disk: `{payload['storage']['current_free_disk_gib']:.3f} GiB`
- Recall floor: `{RECALL_FLOOR_TEXT}`
- val548 accessed: `NO`
- heldout1610 accessed: `NO`
- KD executed: `NO`
- Fidelity-2 started: `NO`
"""


def write_status() -> dict[str, Any]:
    payload = status_payload()
    atomic_json(STATUS_JSON, payload)
    atomic_text(STATUS_MD, status_markdown(payload))
    return payload


def write_heartbeat(
    worker_pid: int,
    campaign_uuid: str,
    worker_state: dict[str, Any],
    current_epoch: int | None = None,
) -> None:
    payload = {
        "timestamp_utc": utc_now(),
        "worker_pid": worker_pid,
        "child_pid": worker_state.get("child_pid"),
        "campaign_uuid": campaign_uuid,
        "method": worker_state.get("method"),
        "replicate": worker_state.get("replicate"),
        "generation_or_batch": worker_state.get("generation_or_batch"),
        "candidate_index": worker_state.get("candidate_index"),
        "candidate_state": worker_state.get("candidate_state"),
        "current_epoch": current_epoch,
        "free_disk_gib": free_gib(),
        "gpu_job_active": bool(worker_state.get("child_pid"))
        and process_alive(worker_state.get("child_pid")),
    }
    atomic_json(HEARTBEAT, payload)
    if CAMPAIGN_LOCK.is_file():
        lock = load_json(CAMPAIGN_LOCK, {})
        lock["latest_heartbeat"] = payload["timestamp_utc"]
        atomic_json(CAMPAIGN_LOCK, lock)


def validate_numeric_row(row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    try:
        params = int(row["params_exact_integer"])
        raw = int(row["flops_raw_exact_integer"])
        displayed = str(row["flops_g_full_precision"])
        recall = str(row["recall_full_precision"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"missing_or_invalid_exact_numeric_field:{exc}"]
    if flops_full_precision(raw) != displayed:
        failures.append("flops_raw_vs_full_precision_mismatch")
    if not exact_budget_feasible(params, raw):
        failures.append("exact_budget_violation")
    if Decimal(recall) != Decimal(str(row.get("recall", recall))):
        failures.append("recall_full_precision_mismatch")
    expected_feasible = recall_feasible(recall)
    recorded = str(row.get("recall_feasible", expected_feasible)).lower() == "true"
    if expected_feasible != recorded:
        failures.append("recall_feasibility_mismatch")
    return failures


def method_filename(method: str) -> Path:
    if method == "NSGA-II AP+APS":
        return ROOT / "search_records_nsga2_ap_aps.csv"
    if method == "Single-objective evolutionary search (AP-only)":
        return ROOT / "search_records_evolutionary_ap_only.csv"
    if method == "Warm-started constrained Random Search":
        return ROOT / "search_records_random.csv"
    raise KeyError(method)


def rewrite_method_files(records: list[dict[str, Any]]) -> None:
    atomic_csv(SEARCH_RECORDS, records)
    for method in METHODS:
        atomic_csv(
            method_filename(method),
            [row for row in records if row.get("method") == method],
        )
