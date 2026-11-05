"""B5 v5 prospective matched FT15 evidence and I/O utilities."""
import release_paths as _release_paths  # Reviewed release-only path relocation.
import sys
sys.dont_write_bytecode=True
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
from pathlib import Path
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parent
TRACK=Path(_release_paths.expand('${BASELINES_ROOT}'))
P=Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE=P/'10_final_fulltrain_300e_matched'
YOLO=Path(_release_paths.expand('${YOLOV5_ROOT}'))
PYTHON=Path(_release_paths.expand('${TRAINING_PYTHON}'))
TEACHER=P/'02_search_parent_teacher/lite_v3_search_parent/weights/best.pt'
TEACHER_SHA='15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659'
TRAINER=OUT/'train_ft15.py'
ORIGINAL_TRAINER=STAGE/'scripts/train_drone_fixed_horizon_no_val.py'
PRIOR=TRACK/'B5_KD_SOURCE_COMPATIBILITY_PRETRAIN_GATE_v1'
SIG='EEA3FB45FD3FD1BADD263DE055266F4F6B5EE34EE818F4C1D7026B764E862C80'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONIOENCODING']='utf-8'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'_matplotlib'))
os.environ['YOLO_CONFIG_DIR']=str(OUT/'_yolo_config')
(OUT/'_yolo_config/Ultralytics').mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(OUT),str(YOLO),_release_paths.expand('${CODE_ROOT}')]


def now():return datetime.now(timezone.utc).isoformat()
def local():return datetime.now().astimezone().isoformat()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest().upper()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str,separators=(',',':'),allow_nan=False).encode()).hexdigest().upper()
def own(p):
    p=Path(p).resolve();p.relative_to(OUT)
    if shutil.disk_usage(OUT).free<512*1024**2:raise RuntimeError('B5_V5_OUTPUT_SPACE_CRITICAL')
    p.parent.mkdir(parents=True,exist_ok=True);return p
def write(p,x):
    p=own(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=True,default=str,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def text(p,s):own(p).write_text(s,encoding='utf-8')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return [{k.strip():v.strip() if isinstance(v,str) else v for k,v in r.items()} for r in csv.DictReader(f)]
def csvout(p,data,fields=None):
    data=list(data);fields=fields or list(dict.fromkeys(k for r in data for k in r));p=own(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def tensor_sha(t):
    a=t.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.numpy().tobytes());return h.hexdigest().upper()
def state_digest(s):return digest({k:tensor_sha(v) for k,v in s.items()})
def epochs(p):
    p=Path(p)/'results.csv'
    if not p.is_file():return []
    s=p.read_text(encoding='utf-8-sig');s=s if s.endswith('\n') else s[:s.rfind('\n')+1]
    rr=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(io.StringIO(s))]
    if [int(float(r['epoch'])) for r in rr]!=list(range(len(rr))) or len(rr)>15:raise RuntimeError('FT15_EPOCH_SEQUENCE')
    if any(not math.isfinite(float(r[k])) for r in rr for k in ('train/box_loss','train/obj_loss','train/cls_loss')):raise RuntimeError('FT15_NONFINITE_LOSS')
    return rr
def verified_matrix():
    m=load(OUT/'FROZEN_SIX_JOB_MATRIX.json');g=load(OUT/'10_PRETRAIN_GATE_DECISION.json')
    if g['status']!='B5_V5_PRETRAIN_PASS' or g['run_matrix_SHA256']!=sha(OUT/'FROZEN_SIX_JOB_MATRIX.json'):raise RuntimeError('B5_V5_FINAL_GATE_NOT_PASS')
    for p,h in m['frozen_files'].items():
        if sha(p)!=h:raise RuntimeError('FROZEN_FILE_CHANGED:'+p)
    return m
def eval_out(split):return OUT/('EVALUATION_VAL548' if split=='val548' else 'EVALUATION_TESTDEV1610')
def eval_source(split):return P/('11_official_val_statistics/01_val548_evaluation' if split=='val548' else '11b_heldout1610_confirmation')


