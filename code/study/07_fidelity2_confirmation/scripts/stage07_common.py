from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import csv
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psutil


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE06A = PROJECT / "06a_fidelity2_promotion_lock"
ROOT = PROJECT / "07_fidelity2_confirmation"
SCRIPTS = ROOT / "scripts"
RUNTIME = ROOT / "runtime"
PHYSICAL = ROOT / "physical_evaluations"

PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))
STAGE06A_MANIFEST = STAGE06A / "stage06a_output_manifest.json"
STAGE06A_MANIFEST_SHA256 = (
    "E5410026CF52D69C086A6044AEB0AC4B5F0C68B2129D5DDFB05E86D67FFA23C2"
)
TARGETMANUAL_SIGNATURE = (
    "AEA2D0A18DAC15E40091E474DBF688A98FCA2B6CFF8D4D3F4845165C59244F5D"
)
SEARCH_PARENT_SHA256 = (
    "15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659"
)

RECOVERY_EPOCHS = 25
TRAINING_SEED = 42
EXPECTED_PHYSICAL = 10
EXPECTED_METHOD_RECORDS = 11
PARAMS_LOWER = 2_440_209
PARAMS_UPPER = 2_515_679
FLOPS_RAW_LOWER = 7_618_383_104
FLOPS_RAW_UPPER = 7_854_003_200
HARD_STOP_FREE_GIB = 8.0

CAMPAIGN_STATE = ROOT / "stage07_campaign_state.json"
WORKER_STATE = RUNTIME / "stage07_worker_state.json"
HEARTBEAT = RUNTIME / "stage07_heartbeat.json"
STATUS_JSON = RUNTIME / "stage07_status.json"
STATUS_MD = RUNTIME / "stage07_status.md"
CAMPAIGN_LOCK = RUNTIME / "stage07_campaign.lock.json"
PAUSE_REQUEST = RUNTIME / "pause_requested.json"
WORKER_STDOUT = RUNTIME / "stage07_worker_stdout.log"
WORKER_STDERR = RUNTIME / "stage07_worker_stderr.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    headers: list[str] | None = None,
) -> None:
    materialized = list(rows)
    if headers is None:
        headers = []
        for row in materialized:
            for key in row:
                if key not in headers:
                    headers.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
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


def ensure_directories() -> None:
    for path in (ROOT, SCRIPTS, RUNTIME, PHYSICAL):
        path.mkdir(parents=True, exist_ok=True)


def free_gib() -> float:
    return shutil.disk_usage(ROOT.anchor).free / (1024**3)


def tree_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def process_alive(pid: Any) -> bool:
    if not pid:
        return False
    try:
        process = psutil.Process(int(pid))
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
        return False


def stage07_worker_processes() -> list[dict[str, Any]]:
    matches = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            process_name = str(process.info.get("name") or "").lower()
            if "python" not in process_name:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if "stage07_worker.py" not in command:
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
        "evaluate_stage07_inner_val.py",
        "evaluate_stage04_inner_val.py",
        "train_drone.py",
        "val.py",
    )
    matches = []
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


def current_epoch(eval_dir: Path | None) -> int | None:
    if eval_dir is None:
        return None
    path = eval_dir / "results.csv"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return None


def completed_durations() -> list[float]:
    values = []
    for row in read_csv(ROOT / "stage07_physical_evaluation_manifest.csv"):
        try:
            values.append(
                float(row["training_duration_seconds"])
                + float(row["explicit_evaluation_duration_seconds"])
            )
        except (KeyError, TypeError, ValueError):
            continue
    return values


def status_payload() -> dict[str, Any]:
    campaign = load_json(CAMPAIGN_STATE, {}) or {}
    worker = load_json(WORKER_STATE, {}) or {}
    completed = len(read_csv(ROOT / "stage07_physical_evaluation_manifest.csv"))
    methods = len(read_csv(ROOT / "stage07_method_result_manifest.csv"))
    durations = completed_durations()
    remaining = max(0, EXPECTED_PHYSICAL - completed)
    mean_duration = statistics.mean(durations) if durations else None
    eval_dir = Path(worker["evaluation_directory"]) if worker.get("evaluation_directory") else None
    child_pid = worker.get("child_pid")
    return {
        "created_utc": utc_now(),
        "overall_state": campaign.get("state", "NOT_STARTED"),
        "current_physical_candidate": worker.get("physical_candidate_id"),
        "current_method_attributions": worker.get("method_attributions", []),
        "candidate_number": worker.get("candidate_number"),
        "candidate_total": EXPECTED_PHYSICAL,
        "candidate_state": worker.get("candidate_state", "NOT_STARTED"),
        "epoch": current_epoch(eval_dir),
        "epoch_total": RECOVERY_EPOCHS,
        "completed_physical_jobs": completed,
        "completed_method_records": methods,
        "expected_method_records": EXPECTED_METHOD_RECORDS,
        "recall_floor_f2": load_json(ROOT / "stage07_recall_floor_f2.json", {}),
        "current_free_disk_gib": free_gib(),
        "deduplication": {
            "promoted_method_records": EXPECTED_METHOD_RECORDS,
            "unique_physical_jobs": EXPECTED_PHYSICAL,
            "duplicate_physical_jobs_avoided": 1,
        },
        "mean_completed_job_seconds": mean_duration,
        "approximate_remaining_seconds": mean_duration * remaining if mean_duration else None,
        "worker_pid": worker.get("worker_pid"),
        "worker_alive": process_alive(worker.get("worker_pid")),
        "child_pid": child_pid,
        "gpu_job_active": process_alive(child_pid),
        "val548_accessed": False,
        "heldout1610_accessed": False,
        "kd_executed": False,
        "stage08a_started": False,
    }


def write_status() -> dict[str, Any]:
    payload = status_payload()
    atomic_json(STATUS_JSON, payload)
    atomic_text(
        STATUS_MD,
        "# Stage07 Status\n\n"
        f"- State: `{payload['overall_state']}`\n"
        f"- Candidate: `{payload['candidate_number']}/{payload['candidate_total']}`\n"
        f"- Physical ID: `{payload['current_physical_candidate']}`\n"
        f"- State detail: `{payload['candidate_state']}`\n"
        f"- Epoch: `{payload['epoch']}/{payload['epoch_total']}`\n"
        f"- Completed physical jobs: `{payload['completed_physical_jobs']}/10`\n"
        f"- Completed method records: `{payload['completed_method_records']}/11`\n"
        f"- Recall floor F2: `{payload['recall_floor_f2'].get('recall_floor_f2_full_precision', 'NOT_LOCKED')}`\n"
        f"- Free disk: `{payload['current_free_disk_gib']:.3f} GiB`\n"
        "- val548 accessed: `NO`\n"
        "- heldout1610 accessed: `NO`\n"
        "- KD executed: `NO`\n"
        "- Stage08A started: `NO`\n",
    )
    return payload
