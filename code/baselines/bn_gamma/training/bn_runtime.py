"""Operational plumbing for the one frozen BN-Gamma candidate, not a trainer."""
import release_paths as _release_paths  # Reviewed release-only path relocation.
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta

O=Path(__file__).resolve().parent
P=Path(_release_paths.expand('${STUDY_ROOT}'))
B=Path(_release_paths.expand('${BASELINES_ROOT}'))
YOLO=Path(_release_paths.expand('${YOLOV5_ROOT}'))
PYTHON=Path(_release_paths.expand('${TRAINING_PYTHON}'))
S10=P/'10_final_fulltrain_300e_matched'
V5=B/'M03_BN_GAMMA_GLOBAL_STRUCTURAL_FEASIBILITY/snapshot_4463f183'
INITIALIZER=V5/'bn_gamma_b6_untrained_raw.pt'
INIT_SHA='7A2C263DB5900AE474F6353D1643801B7E0685C6E80E295531D2FE93E89AC859'
PARENT=P/'02_search_parent_teacher/lite_v3_search_parent/weights/best.pt'
GENE=P/'03a_taylor_dependency_gene_space/gene_space_R1/gene_space_manifest_R1.json'
AUTHOR=Path(_release_paths.expand('${USER_WORKSPACE_ROOT}\\.codex\\attachments\\7cf548b9-a22e-4ab2-9116-77d6784045e2\\Pasted text.txt'))
MODEL='BN-Gamma-B6'
SLUG='bn_gamma_b6'
LABEL='NEW_BN_GAMMA_B6_STAGE10_POLICY_MATCHED_CURRENT_ENV_NOT_BITWISE_HISTORICAL_ENV'
SIG='AFE20F51139A5FF88FBF040CB2D36B06DEB332246B74C8EB0D2CEE62DC62EB91'
WIDTH_SIG='15067E1D35EC011376C29CA74690710E0A8193636A8D66A41C4AEFF3BA21CCFC'
INDEX_SIG='0CBE2364A1136AE4BD3206FB4DF0247EAA3DCEFE590BE596BDC28E25C62C424E'
WIDTHS=[32,64,32,64,32,120,64,128,64,40,120,88,128,384,192,384,192,192,384,192,40,192,96,128,64,72,64,104,40,176,96,192,144,384,192]
PARAMS,FLOPS=2486375,7819187200
PRE=O/'00_PROTOCOL_PREFLIGHT'
PROJECT=P
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'

def now():return datetime.now(timezone(timedelta(hours=3))).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest().upper()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
load=read

def guard(required=100*1024**2):
    if shutil.disk_usage(O).free<required:raise RuntimeError('ACTUAL_OUTPUT_DISK_MARGIN')

def write(path,value):
    path=Path(path).resolve()
    assert path.is_relative_to(O.resolve()),path
    guard();path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False,default=str)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def text(path,value):
    path=Path(path);assert path.resolve().is_relative_to(O.resolve())
    guard();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding='utf-8')

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def csvwrite(path,records,fields=None):
    path=Path(path);assert path.resolve().is_relative_to(O.resolve())
    guard();path.parent.mkdir(parents=True,exist_ok=True);records=list(records)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in records for k in r)))
        w.writeheader();w.writerows(records)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest().upper()

def event(kind,**fields):
    with (O/'EXECUTION_EVENTS.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time=now(),event=kind,**fields),ensure_ascii=False,default=str)+'\n');f.flush()

def observed_epochs(directory):
    path=Path(directory)/'results.csv'
    if not path.exists():return 0,[]
    raw=path.read_text(encoding='utf-8-sig')
    if raw and not raw.endswith('\n'):raw=raw[:raw.rfind('\n')+1]
    rr=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(io.StringIO(raw))]
    for i,r in enumerate(rr):
        assert int(float(r['epoch']))==i,'NONCONTIGUOUS_COMPLETED_EPOCHS'
        assert all(math.isfinite(float(r[k])) for k in ('train/box_loss','train/obj_loss','train/cls_loss')),'NONFINITE_TRAIN_LOSS'
    assert len(rr)<=300
    return len(rr),rr

def eval_out(split):return O/('10_STAGE11_VAL548' if split=='val548' else '11_STAGE11B_TESTDEV1610')
def eval_source(split):return P/('11_official_val_statistics/01_val548_evaluation' if split=='val548' else '11b_heldout1610_confirmation')

def verify_frozen():
    matrix=read(PRE/'06_FROZEN_SEED42_43_44_AND_SIX_EVAL_PLAN.json')
    for path,expected in matrix['code_and_input_sha256'].items():
        assert sha(path)==expected,('FROZEN_SOURCE_CHANGED',path)
    assert sha(INITIALIZER)==INIT_SHA
    assert read(PRE/'INDEPENDENT_PRELAUNCH_GATE.json')['status']=='PASS'
    return matrix

def health():
    result=subprocess.run(['nvidia-smi','--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    lines=[x for x in result.stdout.splitlines() if x.strip()];assert len(lines)==1,'ONE_GPU_REQUIRED'
    v=[x.strip() for x in lines[0].split(',')]
    return dict(time=now(),GPU=v[0],temperature_C=float(v[1]),memory_used_MiB=float(v[2]),memory_total_MiB=float(v[3]),gpu_utilization_pct=float(v[4]),free_disk_bytes=shutil.disk_usage(O).free)

def command_for(cfg):
    out=Path(cfg['output_directory'])
    return [cfg['training_source'],'--img','640','--batch-size','16','--epochs','300','--data',cfg['train_only_yaml'],'--weights',cfg['initialization'],'--hyp',cfg['common_hyp'],'--seed',str(cfg['seed']),'--device','0','--workers','4','--patience','300','--save-period','25','--noplots','--project',str(out.parent),'--name',out.name,'--exist-ok','--disable-validation']

def identity(model,genes,lineages):
    import torch
    modules=[]
    for n,m in model.named_modules():
        if isinstance(m,torch.nn.Conv2d):modules.append([n,m.in_channels,m.out_channels,m.groups,list(m.kernel_size)])
        elif isinstance(m,torch.nn.BatchNorm2d):modules.append([n,m.num_features])
    widths=[dict(model.named_modules())[g['source_modules']].out_channels for g in genes]
    return dict(architecture_sha256=digest(modules),width_vector_sha256=digest(widths),original_index_sha256=digest(lineages),widths=widths)
