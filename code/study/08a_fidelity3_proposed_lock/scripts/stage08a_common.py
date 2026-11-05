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
ROOT = PROJECT / "08a_fidelity3_proposed_lock"
SCRIPTS = ROOT / "scripts"
RUNTIME = ROOT / "runtime"
PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))
RECOVERY_EPOCHS = 50
EXPECTED_RUNS = 3
PARAMS_LOWER, PARAMS_UPPER = 2_440_209, 2_515_679
FLOPS_LOWER, FLOPS_UPPER = 7_618_383_104, 7_854_003_200
HARD_STOP_FREE_GIB = 8.0

CAMPAIGN_STATE = ROOT / "stage08a_campaign_state.json"
WORKER_STATE = RUNTIME / "stage08a_worker_state.json"
STATUS_JSON = RUNTIME / "stage08a_status.json"
STATUS_MD = RUNTIME / "stage08a_status.md"
LOCK = RUNTIME / "stage08a_campaign.lock.json"
PAUSE_REQUEST = RUNTIME / "pause_requested.json"
WORKER_STDOUT = RUNTIME / "stage08a_worker_stdout.log"
WORKER_STDERR = RUNTIME / "stage08a_worker_stderr.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    headers: list[str] = []
    for row in rows:
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
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_directories() -> None:
    for path in (ROOT, SCRIPTS, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)


def free_gib() -> float:
    return shutil.disk_usage(ROOT.anchor).free / (1024**3)


def process_alive(pid: Any) -> bool:
    if not pid:
        return False
    try:
        process = psutil.Process(int(pid))
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
        return False


def worker_processes() -> list[dict[str, Any]]:
    result = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in str(process.info.get("name", "")).lower():
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if "stage08a_worker.py" in command:
                result.append({"pid": process.info["pid"], "command_line": command})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def gpu_children() -> list[dict[str, Any]]:
    result = []
    tokens = ("train_structural_recovery_drone.py", "evaluate_stage08a_inner_val.py")
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if any(token in command for token in tokens):
                result.append({"pid": process.info["pid"], "command_line": command})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def current_epoch(run_dir: Path | None) -> int | None:
    if run_dir is None or not (run_dir / "results.csv").is_file():
        return None
    try:
        return len(read_csv(run_dir / "results.csv"))
    except OSError:
        return None


def status_payload() -> dict[str, Any]:
    campaign = load_json(CAMPAIGN_STATE, {}) or {}
    worker = load_json(WORKER_STATE, {}) or {}
    results = read_csv(ROOT / "stage08a_all_results.csv")
    run_dir = Path(worker["run_directory"]) if worker.get("run_directory") else None
    durations = []
    for row in results:
        try:
            durations.append(float(row["training_duration_seconds"]) + float(row["evaluation_duration_seconds"]))
        except (KeyError, ValueError):
            pass
    mean = statistics.mean(durations) if durations else None
    remaining = max(0, EXPECTED_RUNS - len(results))
    return {
        "created_utc": utc_now(),
        "state": campaign.get("state", "NOT_STARTED"),
        "current_role": worker.get("candidate_role"),
        "current_candidate_id": worker.get("candidate_id"),
        "run_number": worker.get("run_number"),
        "run_total": 3,
        "run_state": worker.get("run_state", "NOT_STARTED"),
        "epoch": current_epoch(run_dir),
        "epoch_total": 50,
        "completed_runs": len(results),
        "recall_floor_f3": load_json(ROOT / "recall_floor_f3.json", {}),
        "free_disk_gib": free_gib(),
        "approximate_remaining_seconds": mean * remaining if mean else None,
        "worker_pid": worker.get("worker_pid"),
        "worker_alive": process_alive(worker.get("worker_pid")),
        "child_pid": worker.get("child_pid"),
        "gpu_job_active": process_alive(worker.get("child_pid")),
        "kd_executed": False,
        "teacher_loaded": False,
        "official_val548_accessed": False,
        "heldout1610_accessed": False,
        "stage08b_started": False,
    }


def write_status() -> dict[str, Any]:
    payload = status_payload()
    atomic_json(STATUS_JSON, payload)
    atomic_text(
        STATUS_MD,
        "# Stage08A Status\n\n"
        f"- State: `{payload['state']}`\n"
        f"- Candidate: `{payload['run_number']}/3 {payload['current_role']}`\n"
        f"- Run state: `{payload['run_state']}`\n"
        f"- Epoch: `{payload['epoch']}/50`\n"
        f"- Completed: `{payload['completed_runs']}/3`\n"
        f"- Recall floor F3: `{payload['recall_floor_f3'].get('recall_floor_f3_full_precision', 'NOT_LOCKED')}`\n"
        f"- Free disk: `{payload['free_disk_gib']:.3f} GiB`\n"
        "- KD executed: `NO`\n- Teacher loaded: `NO`\n"
        "- val548 accessed: `NO`\n- heldout1610 accessed: `NO`\n- Stage08B started: `NO`\n",
    )
    return payload
