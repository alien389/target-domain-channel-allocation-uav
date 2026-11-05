"""R1 semantic ties added to actual DepGraph closures; no dataset access."""
import release_paths as _release_paths  # Reviewed release-only path relocation.
import contextlib
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT=Path(_release_paths.expand('${STUDY_ROOT}'))
YOLO=Path(_release_paths.expand('${YOLOV5_ROOT}'))
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/"02_search_parent_teacher/lite_v3_search_parent/weights/best.pt"
GENES=ROOT/"03a_taylor_dependency_gene_space/gene_space_R1/gene_space_manifest_R1.json"
SOURCE_SHA="15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659"
GENES_SHA="27E0CA4844F076AB3D87B775FAB1566E8BEF9CA608D39AFDDA84BFB2070B67E2"
sys.path[:0]=[str(YOLO),_release_paths.expand('${CODE_ROOT}')]
os.chdir(YOLO)
import torch
import torch_pruning as tp
from thop import profile
from utils.general import LOGGER


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest().upper()


def disk_guard(n=150_000_000):
    if shutil.disk_usage(OUT).free<n:raise RuntimeError("BOUNDED_OUTPUT_DISK_GUARD")


def load_source():
    if sha(SOURCE)!=SOURCE_SHA or sha(GENES)!=GENES_SHA:raise RuntimeError("SOURCE_HASH_MISMATCH")
    with contextlib.redirect_stdout(None):
        model=torch.load(SOURCE,map_location="cpu",weights_only=False)["model"].cpu().float().eval()
    model.requires_grad_(True)  # DepGraph tracing, never backward.
    genes=json.loads(GENES.read_text(encoding="utf-8"))["groups"]
    if len(genes)!=35 or sum(g["allowed_width_count"]>1 for g in genes)!=25:raise RuntimeError("R1_COUNT_MISMATCH")
    return model,genes


def graph(model):
    return tp.DependencyGraph().build_dependency(model,example_inputs=torch.zeros(1,3,128,128))


def seed_paths(model,gene):
    paths=[gene["source_modules"]]
    if gene["scope_name"]=="main_hidden":
        block=dict(model.named_modules())[gene["block_name"]]
        for i,unit in enumerate(block.m):
            if gene["module_type"]=="C3Lite":
                paths.extend([f"{gene['block_name']}.m.{i}.dw",f"{gene['block_name']}.m.{i}.pw"])
            elif gene["module_type"]=="C3":
                paths.extend([f"{gene['block_name']}.m.{i}.cv1.conv",f"{gene['block_name']}.m.{i}.cv2.conv"])
            else:raise RuntimeError("UNSUPPORTED_MAIN_HIDDEN_CLASS")
    return paths


def axis_for(module,handler,dg):
    is_out=dg.is_out_channel_pruning_fn(handler)
    is_in=dg.is_in_channel_pruning_fn(handler)
    if is_in and is_out:return "coupled_in_out"
    if is_out:return "out"
    if is_in:return "in"
    return "handler_specific"


def composite_group(model,dg,gene,indices):
    named=dict(model.named_modules())
    reverse={module:name for name,module in named.items()}
    operations={}
    duplicate_count=0
    paths=seed_paths(model,gene)
    root=named[gene["source_modules"]]
    for path in paths:
        module=named[path]
        if module.out_channels!=root.out_channels:raise RuntimeError(f"SEMANTIC_WIDTH_DRIFT:{path}")
        native=dg.get_pruning_group(module,tp.prune_conv_out_channels,idxs=list(indices))
        for item in native:
            dep,idxs=item
            axis=axis_for(dep.target.module,dep.handler,dg)
            key=(dep.target.module,dep.handler,axis,dep.target.pruning_dim)
            record=operations.setdefault(key,{"dep":dep,"axis":axis,"mapping":{}})
            if item.root_idxs is None:raise RuntimeError("MISSING_NATIVE_ROOT_INDEX_MAP")
            for idx,root_idx in zip(idxs,item.root_idxs):
                if root_idx is None:raise RuntimeError("AMBIGUOUS_ROOT_INDEX_MAP")
                previous=record["mapping"].get(idx)
                if previous is not None:
                    if previous!=root_idx:raise RuntimeError(f"CONFLICTING_ROOT_INDEX:{path}:{idx}")
                    duplicate_count+=1
                else:record["mapping"][idx]=root_idx
    root_key=next(k for k,v in operations.items() if k[0] is root and dg.is_out_channel_pruning_fn(v["dep"].handler))
    ordered=[root_key]+sorted((k for k in operations if k!=root_key),key=lambda k:(reverse.get(k[0],str(k[0])),k[2],k[1].__qualname__))
    combined=tp.dependency.Group()
    combined._DG=dg
    pending=[]
    ledger=[]
    for key in ordered:
        operation=operations[key]
        pairs=sorted(operation["mapping"].items())
        combined.add_dep(operation["dep"],[i for i,r in pairs])
        pending.append([r for i,r in pairs])
        ledger.append({"module":reverse.get(key[0],str(key[0])),"handler":key[1].__qualname__,
                       "axis":key[2],"indices":[i for i,r in pairs],"root_idxs":[r for i,r in pairs]})
    for item,root_idxs in zip(combined,pending):item.root_idxs=root_idxs
    return combined,{"seed_paths":paths,"deduplicated_entries":duplicate_count,"operations":ledger}


def native_importance(group):
    criterion=tp.importance.MagnitudeImportance(p=1,group_reduction="mean",normalizer=None,bias=False)
    values=criterion(group)
    if values is None or not torch.isfinite(values).all() or (values<0).any():raise RuntimeError("INVALID_NATIVE_SALIENCY")
    return values


def counter(model,full640=False):
    old=LOGGER.level;LOGGER.setLevel("ERROR")
    try:
        with contextlib.redirect_stdout(None),torch.no_grad():
            fused=copy.deepcopy(model).cpu().float().eval().fuse()
            p=int(sum(t.numel() for t in fused.parameters()))
            stride=int(max(float(v) for v in fused.stride))
            size=640 if full640 else stride
            macs,_=profile(fused,inputs=(torch.zeros(1,3,size,size),),verbose=False)
    finally:LOGGER.setLevel(old)
    return p,int(round(2.0*float(macs)*(1 if full640 else (640/stride)**2)))


def forward_check(model,size):
    with torch.no_grad():output=model(torch.zeros(1,3,size,size))
    raw=output[1]
    if len(raw)!=3 or any(t.shape[-1]!=15 or not torch.isfinite(t).all().item() for t in raw):raise RuntimeError("NONFINITE_OR_INVALID_SYNTHETIC_OUTPUT")
    return [list(t.shape) for t in raw]


def scope_check(model,genes,lineages,anchor_ref,stride_ref,names_ref):
    named=dict(model.named_modules())
    by_scope={(g["block_name"],g["scope_name"]):len(lineages[g["group_id"]]) for g in genes}
    for g in genes:
        width=len(lineages[g["group_id"]])
        if width not in {int(v) for v in g["allowed_widths"].split(";")}:raise RuntimeError(f"ILLEGAL_R1_WIDTH:{g['group_id']}")
        if named[g["source_modules"]].out_channels!=width:raise RuntimeError("ROOT_LINEAGE_WIDTH_DRIFT")
        if g["scope_name"]=="main_hidden":
            block=named[g["block_name"]]
            for unit in block.m:
                if g["module_type"]=="C3Lite":
                    if (unit.dw.in_channels,unit.dw.out_channels,unit.dw.groups,unit.pw.in_channels,unit.pw.out_channels,unit.bn.num_features)!=(width,)*6:raise RuntimeError("C3LITE_SCOPE_INCOMPLETE")
                else:
                    if any((c.conv.in_channels,c.conv.out_channels,c.bn.num_features)!=(width,)*3 for c in (unit.cv1,unit.cv2)):raise RuntimeError("C3_SCOPE_INCOMPLETE")
            skip=by_scope[(g["block_name"],"skip_hidden")]
            if block.cv3.conv.in_channels!=width+skip:raise RuntimeError("C3_CONCAT_SEGMENT_DRIFT")
        if g["scope_name"]=="hidden":
            block=named[g["block_name"]]
            if block.cv2.conv.in_channels!=4*width:raise RuntimeError("SPPF_CONCAT_DRIFT")
    detect=model.model[-1]
    if model.nc!=10 or detect.nc!=10 or len(detect.m)!=3 or any(c.out_channels!=45 for c in detect.m):raise RuntimeError("PROTECTED_DETECT_CHANGED")
    if not torch.equal(detect.anchors,anchor_ref) or not torch.equal(model.stride,stride_ref):raise RuntimeError("ANCHOR_STRIDE_CHANGED")
    if model.names!=names_ref:raise RuntimeError("CLASS_MAPPING_CHANGED")
    for module in model.modules():
        if isinstance(module,torch.nn.Conv2d):
            if module.weight.dtype!=torch.float32 or module.in_channels%module.groups or module.out_channels%module.groups:raise RuntimeError("DTYPE_OR_GROUPED_CONV_INVALID")


def prune_action(model,genes,gene,indices,lineages):
    dg=graph(model)
    group,ledger=composite_group(model,dg,gene,indices)
    if not dg.check_pruning_group(group):raise RuntimeError("COMPOSITE_GROUP_INVALID")
    affected={}
    root_names={g["source_modules"]:g["group_id"] for g in genes}
    for operation in ledger["operations"]:
        if operation["module"] in root_names and operation["axis"] in ("out","coupled_in_out"):
            affected[root_names[operation["module"]]]=set(operation["indices"])
    group.prune(record_history=False)
    updated=copy.deepcopy(lineages)
    for gid,removed in affected.items():updated[gid]=[oid for i,oid in enumerate(updated[gid]) if i not in removed]
    model.zero_grad(set_to_none=True)
    return updated,ledger
