"""Non-scientific orchestration and evidence I/O for the authorized fixed v6 arm."""
import release_paths as _release_paths  # Reviewed release-only path relocation.
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

O = Path(__file__).resolve().parent
P = Path(_release_paths.expand('${STUDY_ROOT}'))
B=Path(_release_paths.expand('${BASELINES_ROOT}'))
YOLO = Path(_release_paths.expand('${YOLOV5_ROOT}'))
PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))
S10 = P / "10_final_fulltrain_300e_matched"
V5 = B / 'B3_S04_JOINT_P345_STRUCTURAL_FEASIBILITY_20261102/snapshot_1ec4869c'
V4 = B / 'B3_S04_BUDGET_MATCHED_P5_HYBRID_20261102/snapshot_f2e924e7'
INITIALIZER = V5 / "models/SELECTED_JOINT_P345_UNTRAINED.pt"
INIT_SHA = "586AA92460AFBCC69BD53C0FF6721EAAE2FB249B691FFBD8102809DA28E4F35B"
PARENT = P / "02_search_parent_teacher/lite_v3_search_parent/weights/best.pt"
GENE = P / "03a_taylor_dependency_gene_space/gene_space_R1/gene_space_manifest_R1.json"
R1 = P / "scripts/stage03a_r1_gene_space_repair.py"
AUTHOR = Path(_release_paths.expand('${USER_WORKSPACE_ROOT}\\.codex\\attachments\\52144556-24fe-4dd2-b9d3-b300ecd5ad1d\\Pasted text.txt'))
MODEL = "B3_V6_JOINT_P345_D_COMPOUND_REALLOCATION"
SLUG = "b3_v6_joint_p345_d_compound"
LABEL = "NEW_B3_V6_STAGE10_POLICY_MATCHED_CURRENT_ENV_NOT_BITWISE_HISTORICAL_ENV"
SIG = "D01D19CD07B2C33775AA45A50C5FD2957A277DE702A6E7224C5CE4D9D6666B65"
PATH_GROUPS = ["DG-020", "DG-021", "DG-022", "DG-023", "DG-026", "DG-028", "DG-029", "DG-030", "DG-031", "DG-032", "DG-033", "DG-034", "DG-035"]
COMP = {"DG-010": 112, "DG-012": 256, "DG-014": 296, "DG-019": 288}
WIDTHS = [32,64,32,64,32,48,64,128,64,112,32,256,120,296,144,384,192,176,288,152,96,192,96,128,64,80,64,80,80,128,80,120,136,224,136]
PRE = O / "00_PROTOCOL_AND_PREFLIGHT"
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

def now():
    return datetime.now(timezone(timedelta(hours=3))).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

load = read

def write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(O.resolve()), path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def csvwrite(path, records, fields=None):
    path = Path(path)
    assert path.resolve().is_relative_to(O.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(records[0]))
        w.writeheader()
        w.writerows(records)

def text(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(O.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest().upper()

def event(kind, **fields):
    with (O / "EXECUTION_EVENTS.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time=now(),event=kind,**fields),ensure_ascii=False,default=str)+"\n")
        f.flush()

def observed_epochs(directory):
    p = Path(directory) / "results.csv"
    if not p.exists():
        return 0, []
    raw = p.read_text(encoding="utf-8-sig")
    if raw and not raw.endswith("\n"):
        raw = raw[:raw.rfind("\n") + 1]
    rr = [{k.strip(): v.strip() for k,v in r.items()} for r in csv.DictReader(io.StringIO(raw))]
    for i, r in enumerate(rr):
        assert int(float(r["epoch"])) == i, "NONCONTIGUOUS_ACTUAL_EPOCHS"
        assert all(math.isfinite(float(r[k])) for k in ("train/box_loss", "train/obj_loss", "train/cls_loss")), "NONFINITE_TRAINING_LOSS"
    assert len(rr) <= 300
    return len(rr), rr

def eval_out(split):
    return O / ("10_STAGE11_VAL548" if split == "val548" else "11_STAGE11B_TESTDEV1610")

def eval_source(split):
    return P / ("11_official_val_statistics/01_val548_evaluation" if split == "val548" else "11b_heldout1610_confirmation")

PROJECT = P

def verify_frozen():
    matrix = read(PRE / "06_FROZEN_THREE_SEED_RUN_AND_SIX_EVAL_MATRIX.json")
    for path, expected in matrix["code_and_input_sha256"].items():
        assert sha(path) == expected, ("FROZEN_SOURCE_CHANGED", path)
    assert sha(INITIALIZER) == INIT_SHA
    assert read(PRE / "INDEPENDENT_PRELAUNCH_GATE.json")["status"] == "PASS"
    return matrix

def health():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    lines = [x for x in result.stdout.splitlines() if x.strip()]
    assert len(lines) == 1, "EXACTLY_ONE_VISIBLE_GPU_REQUIRED"
    v = [x.strip() for x in lines[0].split(",")]
    return dict(time=now(),GPU=v[0],temperature_C=float(v[1]),memory_used_MiB=float(v[2]),memory_total_MiB=float(v[3]),gpu_utilization_pct=float(v[4]),free_disk_bytes=shutil.disk_usage(O).free)

def command_for(cfg):
    out = Path(cfg["output_directory"])
    return [str(cfg["training_source"]), "--img", "640", "--batch-size", "16", "--epochs", "300", "--data", cfg["train_only_yaml"],
            "--weights", cfg["initialization"], "--hyp", cfg["common_hyp"], "--seed", str(cfg["seed"]), "--device", "0", "--workers", "4",
            "--patience", "300", "--save-period", "25", "--noplots", "--project", str(out.parent), "--name", out.name, "--exist-ok", "--disable-validation"]
