"""Reproduce the frozen B0.1 fresh constructor and exact Meta channel slices."""
import release_paths as _release_paths  # Reviewed release-only path relocation.
import contextlib
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
import b2_common as c
sys.path[:0]=[str(c.YOLO),_release_paths.expand('${CODE_ROOT}'),str(c.PROJECT/"scripts")]
import torch
import numpy as np
import yaml
torch.set_num_threads(4)


def rng():
    n=np.random.get_state()
    return {"torch_cpu_bytes":torch.get_rng_state().tolist(),"python_random":random.getstate(),"numpy":{"algorithm":n[0],"state":n[1].tolist(),"position":n[2],"has_gauss":n[3],"cached_gaussian":n[4]},"CUDA_initialized":torch.cuda.is_initialized()}


def topology(model):
    rows=[]
    for name,mod in model.named_modules():
        row={"name":name,"type":type(mod).__module__+"."+type(mod).__name__}
        for attr in ("in_channels","out_channels","kernel_size","stride","padding","dilation","groups","num_features","eps","momentum","affine","track_running_stats","f","nc","nl","na","no","add"):
            if hasattr(mod,attr):
                v=getattr(mod,attr)
                if isinstance(v,torch.Tensor):v=v.detach().cpu().tolist()
                row[attr]=v
        rows.append(row)
    return rows


def main():
    if (c.OUT/"04_B2_SCRATCH_INITIALIZATION_THREE_SEED_LOCK.json").exists():raise RuntimeError("INITIALIZER_LOCK_ALREADY_EXISTS")
    baseline=c.load(c.HERE/"SOURCE_BASELINE.json")
    for p,row in baseline["sources"].items():
        if c.sha(p)!=row["sha256"]:raise RuntimeError("SOURCE_CHANGED_BEFORE_CONSTRUCTOR:"+p)
    from models.yolo import Model
    import pruning_research.pruning.structured_pruner as pruner
    from thop import profile
    historical=torch.load(c.META,map_location="cpu",weights_only=False)["model"].float().eval()
    spec=yaml.safe_load(c.META_YAML.read_text(encoding="utf-8"))
    original_plan=c.load(c.PLAN)
    historical_state=historical.state_dict()
    source_anchor=historical.model[-1].anchors.detach().clone()
    original_load=torch.load
    original_loader=pruner.load_detection_model
    old_conv=pruner._copy_conv2d_subset
    old_dw=pruner._copy_depthwise_conv2d_subset
    old_bn=pruner._copy_bn_subset
    records=[]
    origins=[]
    for seed in (42,43,44):
        destination=c.OUT/f"INITIALIZERS/SEED{seed}"
        if destination.exists():raise RuntimeError("SCRATCH_INITIALIZER_DIRECTORY_COLLISION")
        destination.mkdir(parents=True,exist_ok=False)
        parent_path=destination/f"fresh_parent_seed{seed}.pt"
        # The frozen B0.1 recipe seeds Torch immediately before Model(cfg).
        torch.manual_seed(seed)
        before=rng()
        def forbid_load(*args,**kwargs):raise RuntimeError("CHECKPOINT_LOAD_IN_FRESH_CONSTRUCTOR_FORBIDDEN")
        torch.load=forbid_load
        try:
            fresh=Model(cfg=copy.deepcopy(spec["base_model_yaml"]),ch=3,nc=10).cpu().float().eval()
        finally:torch.load=original_load
        after=rng()
        fresh.nc=10
        fresh.names=["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]
        with torch.no_grad():fresh.model[-1].anchors.copy_(source_anchor)
        if fresh.model[-1].anchors.requires_grad:raise RuntimeError("ANCHORS_NOT_STRUCTURAL_CONSTANT")
        fresh_state={k:t.detach().clone() for k,t in fresh.state_dict().items()}
        if len(fresh_state)!=319:raise RuntimeError("FRESH_PARENT_SCHEMA_NOT_319")
        torch.save({"model":fresh,"optimizer":None,"ema":None,"epoch":-1},parent_path)
        plan=copy.deepcopy(original_plan)
        plan.update(created_utc=c.now(),plan_label="B2 same-architecture fresh random initializer",target_label=f"B2-Meta-Scratch-{seed}",source_checkpoint=str(parent_path),source_checkpoint_sha256=c.sha(parent_path),search_role="frozen_Meta_indices_no_search_no_training")
        plan_path=destination/"frozen_meta_indices_plan.json"
        c.write(plan_path,plan)
        axes={}
        module_names={}
        allowed_paths={parent_path.resolve()}
        def local_load(path,*args,**kwargs):
            if not isinstance(path,(str,Path)) or Path(path).resolve() not in allowed_paths:raise RuntimeError("NONFRESH_WEIGHT_LOAD_DURING_SCRATCH_SLICING:"+str(path))
            return original_load(path,*args,**kwargs)
        def loader(*args,**kwargs):
            artifact=original_loader(*args,**kwargs)
            module_names.update({id(mod):name for name,mod in artifact.model.named_modules()})
            return artifact
        def conv(mod,in_keep=None,out_keep=None):
            name=module_names[id(mod)]
            ii=list(in_keep) if in_keep is not None and len(in_keep) else list(range(mod.in_channels))
            oo=list(out_keep) if out_keep is not None and len(out_keep) else list(range(mod.out_channels))
            if name+".weight" in axes:raise RuntimeError("DUPLICATE_PHYSICAL_WEIGHT_COPY")
            axes[name+".weight"]={"axis0":oo,"axis1":ii,"kind":"Conv2d"}
            if mod.bias is not None:axes[name+".bias"]={"axis0":oo,"kind":"Conv_bias"}
            return old_conv(mod,in_keep=in_keep,out_keep=out_keep)
        def dw(mod,channel_keep):
            name=module_names[id(mod)];ids=list(channel_keep)
            axes[name+".weight"]={"axis0":ids,"kind":"depthwise"}
            if mod.bias is not None:axes[name+".bias"]={"axis0":ids,"kind":"depthwise_bias"}
            return old_dw(mod,channel_keep)
        def bn(mod,keep_out):
            name=module_names[id(mod)]
            for suffix in ("weight","bias","running_mean","running_var"):
                axes[name+"."+suffix]={"axis0":list(keep_out),"kind":"BN_fresh_affine_or_running_buffer"}
            axes[name+".num_batches_tracked"]={"kind":"BN_fresh_counter"}
            return old_bn(mod,keep_out)
        pruner.load_detection_model=loader
        pruner._copy_conv2d_subset=conv
        pruner._copy_depthwise_conv2d_subset=dw
        pruner._copy_bn_subset=bn
        # The original loader may reload its new output; no historic weight path
        # is ever permitted while the scratch constructor/slicer is executing.
        raw=destination/f"weights/b2_fresh_parent_seed{seed}/b2_meta_scratch_seed{seed}_init.pt"
        allowed_paths.add(raw.resolve())
        torch.load=local_load
        try:
            result=pruner.apply_structured_pruning(weights=str(parent_path),yolov5_root=str(c.YOLO),plan_json=str(plan_path),profile_dir=str(c.PROFILE),output_dir=str(destination/"structural_audit"),weights_output_dir=str(destination/"weights"),model_id=f"b2_fresh_parent_seed{seed}",pruned_model_id=f"b2_meta_scratch_seed{seed}_init",device="cpu",validate=True,validate_imgsz=640)
        finally:
            torch.load=original_load
            pruner.load_detection_model=original_loader
            pruner._copy_conv2d_subset=old_conv
            pruner._copy_depthwise_conv2d_subset=old_dw
            pruner._copy_bn_subset=old_bn
        candidate_path=Path(result["pruned_weights"])
        candidate=torch.load(candidate_path,map_location="cpu",weights_only=False)["model"].float().eval()
        actual=candidate.state_dict()
        trainable={k for k,_ in candidate.named_parameters()}
        for key,t in actual.items():
            rule=axes.get(key,{"kind":"unmodified_fresh_constructor_tensor"})
            wanted=fresh_state[key]
            for axis in (0,1):
                if "axis"+str(axis) in rule:wanted=wanted.index_select(axis,torch.tensor(rule["axis"+str(axis)],dtype=torch.long))
            if not torch.equal(t,wanted):raise RuntimeError("FRESH_PARENT_SLICE_PARITY_FAILURE:"+key)
            origins.append({"seed":seed,"key":key,"is_trainable":key in trainable,"shape":json.dumps(list(t.shape)),"tensor_sha256":c.tensor_sha(t),"source_parent":str(parent_path),"source_tensor_sha256":c.tensor_sha(fresh_state[key]),"slice_rule":json.dumps(rule),"origin":"structural_anchor_constant" if key.endswith(".anchors") else "fresh_seeded_constructor_then_original_index_slice","learned_historical_tensor_copied":False,"identical_to_inherited_tensor":bool(torch.equal(t,historical_state[key]))})
        if topology(candidate)!=topology(historical):raise RuntimeError("SAME_ARCHITECTURE_TOPOLOGY_NOT_EXACT")
        if set(actual)!=set(historical_state) or any(actual[k].shape!=historical_state[k].shape for k in actual):raise RuntimeError("MATCHED_META_SCHEMA_FAILURE")
        meaningful_equal=[k for k in trainable if torch.equal(actual[k],historical_state[k])]
        if meaningful_equal:raise RuntimeError("IDENTICAL_INHERITED_TRAINABLE_BLOCK_REQUIRES_INVESTIGATION:"+repr(meaningful_equal))
        if seed==42:
            old=c.B01/"scratch_weights/b01_scratch_parent_seed42/b01_meta_scratch_seed42_init.pt"
            diagnostic=torch.load(old,map_location="cpu",weights_only=False)["model"].float().state_dict()
            if set(actual)!=set(diagnostic) or any(not torch.equal(actual[k],diagnostic[k]) for k in actual):raise RuntimeError("B01_SEED42_TENSOR_REPRODUCTION_MISMATCH")
        with torch.no_grad():prediction=candidate(torch.zeros(1,3,640,640))
        shapes=[list(t.shape) for t in prediction[1]]
        if shapes!=[[1,3,80,80,15],[1,3,40,40,15],[1,3,20,20,15]] or not all(torch.isfinite(t).all() for t in prediction[1]):raise RuntimeError("SYNTHETIC640_FAILURE")
        fused=copy.deepcopy(candidate).eval().fuse()
        params=sum(p.numel() for p in fused.parameters())
        macs,_=profile(fused,inputs=(torch.zeros(1,3,32,32),),verbose=False)
        flops=int(round(2*float(macs)*(640/32)**2))
        if (params,flops)!=(2507047,7847110400):raise RuntimeError("EXACT_META_RESOURCE_FAILURE")
        c.write(destination/"fresh_constructor_rng_and_copy_provenance.json",{"seed":seed,"construction_seed_method":"torch.manual_seed(seed) immediately before Model(base_model_yaml)","NumPy_and_python_not_reseeded_in_frozen_B01_constructor":True,"before":before,"after_constructor":after,"copied_historical_values":[{"key":"model.24.anchors","learnable":False,"source":str(c.META),"source_sha256":c.sha(c.META),"values":source_anchor.tolist()}],"all_actual_state_keys_verified_against_fresh_slices":True,"source_slice_axes":axes,"fresh_parent_state_digest":c.state_digest(fresh_state)})
        record={"seed":seed,"path":str(candidate_path),"sha256":c.sha(candidate_path),"state_digest":c.state_digest(actual),"trainable_digest":c.state_digest({k:actual[k] for k in sorted(trainable)}),"parent_path":str(parent_path),"parent_sha256":c.sha(parent_path),"architecture_signature":c.SIG,"topology_digest":c.digest(topology(candidate)),"keys":len(actual),"named_parameters":len(trainable),"BN_modules":sum(isinstance(m,torch.nn.BatchNorm2d) for m in candidate.modules()),"fused_Params":params,"raw_FLOPs_640":flops,"synthetic640_shapes":shapes,"B01_seed42_all_tensors_equal":True if seed==42 else None,"same_serialized_bytes_as_B01":c.sha(candidate_path)=="6CA359ABE91B83D3EFF9F0A3DD6FE6D851C8F79D02DD3F9B6BBB0F5F54EB86B8" if seed==42 else None}
        if (record["keys"],record["named_parameters"],record["BN_modules"])!=(319,165,51):raise RuntimeError("META_KEY_PARAMETER_BN_COUNT_MISMATCH")
        records.append(record)
        c.write(destination/"constructor_certificate.json",record)
        print("B2_SCRATCH_CONSTRUCTED_SEED="+str(seed),flush=True)
    if len({r["trainable_digest"] for r in records})!=3:raise RuntimeError("SCRATCH_SEEDS_NOT_INDEPENDENT_INITIAL_VALUES")
    c.csvout(c.OUT/"05_B2_SCRATCH_STATE_KEY_TENSOR_ORIGIN_AUDIT.csv",origins)
    c.write(c.OUT/"04_B2_SCRATCH_INITIALIZATION_THREE_SEED_LOCK.json",{"status":"CONSTRUCTED_PENDING_INDEPENDENT_AND_LOADER_AUDIT","created_utc":c.now(),"seeds":records,"source_recipe":str(c.B01/"b01_probe.py"),"source_recipe_sha256":c.sha(c.B01/"b01_probe.py"),"constructor_sha256":c.sha(__file__),"initialization_RNG":"torch.manual_seed(s) immediately before fresh YOLO Model constructor; numpy/python not newly reseeded","training_RNG":"original trainer init_seeds(seed+1+RANK, deterministic=True), RANK=-1 means effective seed=s","no_historical_learnable_or_BN_running_state_copies":True,"nonlearnable_exception":"Detect anchors copied from frozen Meta structural metadata","serialization_note":"New checkpoint metadata, names, source path and serializer storage IDs can change byte SHA. Seed42 is required to match all319 B0.1 state tensors exactly, not to match pickle bytes."})
    c.write(c.OUT/"03_B2_STAGE09_META_ARCHITECTURE_AND_INDEX_LOCK.json",{"architecture_signature":c.SIG,"source_Meta_raw":str(c.META),"source_Meta_raw_sha256":c.sha(c.META),"source_yaml":str(c.META_YAML),"source_yaml_sha256":c.sha(c.META_YAML),"original_plan":str(c.PLAN),"original_plan_sha256":c.sha(c.PLAN),"retained_widths":spec["retained_widths"],"dependency_groups":35,"module_topology":topology(historical),"fused_Params":2507047,"raw_FLOPs_640":7847110400,"source_trained_tensors_used_for_reference_comparison_only":True})


if __name__=="__main__":main()
