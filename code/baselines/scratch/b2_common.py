"""B2 evidence/operational utilities, with no scientific training implementation."""
import release_paths as _release_paths  # Reviewed release-only path relocation.
import sys
sys.dont_write_bytecode=True
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import io
import math
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=Path(_release_paths.expand('${SCRATCH_ROOT}'))
TRACK=Path(_release_paths.expand('${BASELINES_ROOT}'))
PROJECT=Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE=PROJECT/"10_final_fulltrain_300e_matched"
YOLO=Path(_release_paths.expand('${YOLOV5_ROOT}'))
PYTHON=Path(_release_paths.expand('${TRAINING_PYTHON}'))
B01=TRACK/"B0_READINESS_AND_PROTOCOL_LOCK/B0_1_GAP_CLOSURE"
B1=TRACK/"B1_V3_FINAL_300E_THREE_SEEDS"
WORKER=PROJECT/"10_final_fulltrain_300e_matched/scripts/stage10_run_training.py"
TRAINER=STAGE/"scripts/train_drone_fixed_horizon_no_val.py"
META=PROJECT/"09_locked_architectures/raw_checkpoints/meta_ap_aps_b6_raw.pt"
META_YAML=PROJECT/"09_locked_architectures/yamls/meta_ap_aps_b6.yaml"
PLAN=PROJECT/"07_fidelity2_confirmation/physical_evaluations/F2_10_EEA3FB45FD3FD1BA/pruning_plan.json"
PROFILE=PROJECT/"03a_taylor_dependency_gene_space/structural_profile/litev3_searchparent_nc10_stage03a_unfused"
GENES=PROJECT/"03a_taylor_dependency_gene_space/gene_space_R1/gene_space_manifest_R1.json"
MODEL="Meta-AP+APS-B6 random scratch"
SLUG="meta_ap_aps_b6_random_scratch"
LABEL="NEW_B2_META_SCRATCH_STAGE10_POLICY_MATCHED_CURRENT_ENV_NOT_BITWISE_HISTORICAL_ENV"
SIG="EEA3FB45FD3FD1BADD263DE055266F4F6B5EE34EE818F4C1D7026B764E862C80"
os.environ.setdefault("MPLCONFIGDIR",str(HERE/"_matplotlib_cache"))


def now():return datetime.now(timezone.utc).isoformat()
def local():return datetime.now().astimezone().isoformat()
def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest().upper()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str,allow_nan=False).encode()).hexdigest().upper()
def own(path):
    path=Path(path).resolve();path.relative_to(OUT)
    if shutil.disk_usage(OUT).free<100000000:raise RuntimeError("B2_NEW_OUTPUT_DISK_GUARD")
    path.parent.mkdir(parents=True,exist_ok=True)
    return path
def write(path,value):
    path=own(path);temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,ensure_ascii=True,default=str,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def text(path,value):own(path).write_text(value,encoding="utf-8")
def rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:return [{k.strip():v.strip() if isinstance(v,str) else v for k,v in r.items()} for r in csv.DictReader(f)]
def csvout(path,records,fields=None):
    records=list(records);path=own(path);fields=fields or list(dict.fromkeys(k for row in records for k in row));temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(records);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def tensor_sha(t):
    a=t.detach().cpu().contiguous()
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.numpy().tobytes());return h.hexdigest().upper()
def state_digest(state):return digest({k:tensor_sha(v) for k,v in state.items()})
def observed_epochs(directory):
    path=Path(directory)/"results.csv"
    if not path.exists():return 0,[]
    raw=path.read_text(encoding="utf-8-sig")
    if raw and not raw.endswith("\n"):raw=raw[:raw.rfind("\n")+1]
    values=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(io.StringIO(raw))]
    for i,row in enumerate(values):
        if int(float(row["epoch"]))!=i:raise RuntimeError("B2_EPOCH_SEQUENCE_INVALID")
        if any(not math.isfinite(float(row[k])) for k in ("train/box_loss","train/obj_loss","train/cls_loss")):raise RuntimeError("B2_NONFINITE_TRAINING_LOSS")
    if len(values)>300:raise RuntimeError("B2_EXCESS_EPOCHS")
    return len(values),values
def eval_out(split):return OUT/("04_STAGE11_VAL548" if split=="val548" else "05_STAGE11B_HELDOUT1610")
def eval_source(split):return PROJECT/("11_official_val_statistics/01_val548_evaluation" if split=="val548" else "11b_heldout1610_confirmation")
def verify_frozen():
    matrix=load(OUT/"09_B2_EXACT_PRECOMMITTED_RUN_MATRIX.json")
    for p,h in matrix["code_and_input_sha256"].items():
        if sha(p)!=h:raise RuntimeError("FROZEN_B2_INPUT_CHANGED:"+p)
    return matrix
