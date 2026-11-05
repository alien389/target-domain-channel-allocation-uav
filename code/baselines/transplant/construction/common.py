"""CPU-only design evidence helpers. No training or dataset loading interfaces."""
import release_paths as _release_paths  # Reviewed release-only path relocation.
import csv,hashlib,importlib.util,json,os,shutil,sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
O=Path(__file__).resolve().parent.parent
P=Path(_release_paths.expand('${STUDY_ROOT}'))
B=Path(_release_paths.expand('${BASELINES_ROOT}'))
YOLO=Path(_release_paths.expand('${YOLOV5_ROOT}'))
R1=P/"scripts/stage03a_r1_gene_space_repair.py"
GENES=P/"03a_taylor_dependency_gene_space/gene_space_R1/gene_space_manifest_R1.json"
PARENT=P/"02_search_parent_teacher/lite_v3_search_parent/weights/best.pt"
PRUNER=Path(_release_paths.expand('${PRUNING_RESEARCH_ROOT}\\pruning\\structured_pruner.py'))
V4=B/'B3_S04_BUDGET_MATCHED_P5_HYBRID_20261102/snapshot_f2e924e7'
V2=B/"B3_S04_REVERSE_TRANSPLANT_Q1_20261102/run_20261102_v2"
V3=B/'B3_S04_PARTIAL_PATH_FEASIBILITY_20261102/snapshot_71ecba14'
AUTHOR=Path(_release_paths.expand('${USER_WORKSPACE_ROOT}\\.codex\\attachments\\703e830c-50f4-4660-90e7-1897cb5b2666\\Pasted text.txt'))
PATHS=["DG-020","DG-021","DG-022","DG-023","DG-026","DG-028","DG-029","DG-030","DG-031","DG-032","DG-033","DG-034","DG-035"]
REGIONS={"P3_INGRESS_SHARED":["DG-020","DG-021","DG-022","DG-023","DG-026"],"P4_P5":["DG-028","DG-029","DG-030","DG-031"],"LATE_P5":["DG-032","DG-033","DG-034","DG-035"]}
DIRECTIONS={"D":("Manual","Meta"),"R":("Meta","Manual")}
BOUNDS=(2440209,2515679,7618383104,7854003200)
sys.dont_write_bytecode=True
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
def now():return datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda:f.read(1048576),b""):h.update(x)
    return h.hexdigest().upper()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,value):
    p=Path(p);assert p.resolve().is_relative_to(O.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(O).free>128*1024*1024
    assert not p.exists(),p
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=str)+"\n",encoding="utf-8")
def text(p,value):
    p=Path(p);assert p.resolve().is_relative_to(O.resolve());assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding="utf-8")
def rows(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def csvwrite(p,rr,fields=None):
    rr=list(rr);p=Path(p);assert p.resolve().is_relative_to(O.resolve());assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rr[0]));w.writeheader();w.writerows(rr)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def signature(order,w):
    return hashlib.sha256(";".join(f"{g}={w[g]}" for g in order).encode("ascii")).hexdigest().upper()
def ratios(p,f):
    return [Fraction(abs(p-2507047),2507047),Fraction(abs(f-7847110400),7847110400),
            Fraction(abs(p-2512591),2512591),Fraction(abs(f-7775097600),7775097600)]
def tier(rs):
    m=max(rs)
    return 0 if m<=Fraction(1,100) else 1 if m<=Fraction(1,50) else 2 if m<=Fraction(1,20) else 3
def metrics(p,f):
    rr=ratios(p,f);pmin,pmax,fmin,fmax=BOUNDS
    return dict(params=p,raw_flops=f,macs=f//2,P_minus_Pmin=p-pmin,Pmax_minus_P=pmax-p,F_minus_Fmin=f-fmin,Fmax_minus_F=fmax-f,
                HARD_CAP_PASS=p<=pmax and f<=fmax,ORIGINAL_B6_BAND_PASS=pmin<=p<=pmax and fmin<=f<=fmax,
                matching_tier=tier(rr),matching_tier_label=["<=1%","(1%,2%]","(2%,5%]",">5%"][tier(rr)],
                max_relative_fraction=str(max(rr)),sum_relative_fraction=str(sum(rr)),
                Params_Meta_abs_relative=str(rr[0]),FLOPs_Meta_abs_relative=str(rr[1]),
                Params_Manual_abs_relative=str(rr[2]),FLOPs_Manual_abs_relative=str(rr[3]))
def rank(row):
    return (not row["HARD_CAP_PASS"],not row["ORIGINAL_B6_BAND_PASS"],row["matching_tier"],row["compensator_count"],
            Fraction(row["max_relative_fraction"]),Fraction(row["sum_relative_fraction"]),Fraction(row["normalized_offtarget_edit"]),
            0 if row["direction"]=="D" else 1,tuple(row["width_vector"]))
