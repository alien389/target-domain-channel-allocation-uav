"""One untrained joint architecture, original pruning semantics, CPU/no_grad only."""
import copy,hashlib,json,shutil
import common as c

def main():
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    assert c.read(c.O/"aux/INDEPENDENT_RESOURCE_REVIEW.json")["status"]=="PASS"
    selected=c.read(c.O/"aux/SELECTED_AND_LEADING.json")["selected"]
    groups=c.read(c.GENES)["groups"];order=[g["group_id"] for g in groups]
    w=dict(zip(order,selected["width_vector"]))
    r1=c.module("v5_original_r1_physical",c.R1)
    source=r1.load_source_model(c.PARENT,"V5_SOURCE_PARENT_ONLY").model
    bank,ranking=r1.taylor_scope_bank(groups)
    c.write(c.O/"aux/PHYSICAL_VALIDATION_PROTOCOL.json",dict(code_sha256=c.sha(__file__),source_parent=c.sha(c.PARENT),
        candidate_signature=selected["signature"],device="cpu",precision="FP32",input=[1,3,640,640],
        forward_reload_tolerance=dict(atol=1e-6,rtol=1e-5),tensor_lineage="EXACT_TORCH_EQUAL",
        optimizer_steps=0,backward=0,training=0,GPU=0,dataset_images=0))
    plan=r1.build_pruning_plan("V5_STRICT_JOINT_UNTRAINED",groups,tuple(w[g] for g in order),bank)
    c.write(c.O/"models/pruning_plan.json",plan)
    from pruning_research.pruning.structured_pruner import apply_structured_pruning
    result=apply_structured_pruning(weights=str(c.PARENT),yolov5_root=str(c.YOLO),plan_json=str(c.O/"models/pruning_plan.json"),
        profile_dir=str(r1.PROFILE_DIR),output_dir=str(c.O/"models/pruner_reports"),weights_output_dir=str(c.O/"models/weights"),
        model_id="v5_joint_p345",pruned_model_id="JOINT_P345_UNTRAINED",device="cpu",validate=False,validate_imgsz=640)
    raw=c.Path(result["pruned_weights"])
    saved=torch.load(raw,map_location="cpu",weights_only=False)
    child=saved["model"].float().eval()
    assert saved["epoch"]==-1 and saved["optimizer"] is None and saved["ema"] is None
    keeps={}
    for g in groups:
        removed=bank[g["group_id"]]["indices_low_to_high"][:g["original_width"]-w[g["group_id"]]]
        keeps[g["group_id"]]=sorted(set(range(g["original_width"]))-set(removed))
        assert len(keeps[g["group_id"]])==w[g["group_id"]]
    c.csvwrite(c.O/"aux/ALL_35_RETAINED_CHANNELS.csv",[dict(group_id=g["group_id"],block=g["block_name"],scope=g["scope_name"],
        original_width=g["original_width"],retained_width=w[g["group_id"]],original_ids=";".join(map(str,keeps[g["group_id"]])),
        frozen_ranking_sha256=c.sha(c.P/"03a_taylor_dependency_gene_space/taylor_ranking_manifest.json")) for g in groups])
    sd=source.state_dict();expected=dict(sd);axes={k:(None,None) for k in sd}
    scope={(g["block_index"],g["scope_name"]):keeps[g["group_id"]] for g in groups}
    outmap={};original_out={}
    def conv(name,ii,oo,dw=False):
        k=name+".weight";expected[k]=sd[k][oo] if dw else sd[k][oo][:,ii];axes[k]=(oo,None if dw else ii)
        if name+".bias" in sd:expected[name+".bias"]=sd[name+".bias"][oo];axes[name+".bias"]=(oo,None)
    def bn(name,oo):
        for field in ("weight","bias","running_mean","running_var"):
            k=name+"."+field;expected[k]=sd[k][oo];axes[k]=(oo,None)
    def block(name,ii,oo):conv(name+".conv",ii,oo);bn(name+".bn",oo)
    for i,m in enumerate(source.model):
        typ=type(m).__name__;n=f"model.{i}";fs=[m.f] if isinstance(m.f,int) else m.f
        src=[x if x>=0 else i+x for x in fs]
        ii=list(range(3)) if i==0 else list(outmap[src[0]]) if len(src)==1 else []
        if len(src)>1:
            offset=0
            for j in src:ii.extend(offset+x for x in outmap[j]);offset+=original_out[j]
        if typ=="Conv":
            oo=scope[i,"output"];block(n,ii,oo);outmap[i]=oo;original_out[i]=m.conv.out_channels
        elif typ in ("C3","C3Lite"):
            main,skip,oo=scope[i,"main_hidden"],scope[i,"skip_hidden"],scope[i,"output"]
            block(n+".cv1",ii,main);block(n+".cv2",ii,skip)
            for j in range(len(m.m)):
                if typ=="C3":block(f"{n}.m.{j}.cv1",main,main);block(f"{n}.m.{j}.cv2",main,main)
                else:conv(f"{n}.m.{j}.dw",main,main,True);conv(f"{n}.m.{j}.pw",main,main);bn(f"{n}.m.{j}.bn",main)
            block(n+".cv3",main+[m.cv1.conv.out_channels+x for x in skip],oo)
            outmap[i]=oo;original_out[i]=m.cv3.conv.out_channels
        elif typ=="SPPF":
            hidden,oo=scope[i,"hidden"],scope[i,"output"]
            block(n+".cv1",ii,hidden);block(n+".cv2",[seg*m.cv1.conv.out_channels+x for seg in range(4) for x in hidden],oo)
            outmap[i]=oo;original_out[i]=m.cv2.conv.out_channels
        elif typ=="Upsample":outmap[i]=ii;original_out[i]=original_out[src[0]]
        elif typ=="Concat":outmap[i]=ii;original_out[i]=sum(original_out[j] for j in src)
        elif typ=="Detect":
            for j,k in enumerate(src):conv(f"{n}.m.{j}",outmap[k],list(range(m.m[j].out_channels)))
    actual=child.state_dict();assert len(actual)==len(expected)==319 and set(actual)==set(expected)
    lineage=[]
    for k,value in actual.items():
        assert torch.equal(value,expected[k]),k
        axis0,axis1=axes[k]
        lineage.append(dict(key=k,axis0_original_indices=json.dumps(axis0),axis1_original_indices=json.dumps(axis1),
           parent_shape=json.dumps(list(sd[k].shape)),child_shape=json.dumps(list(value.shape)),exact_parent_slice=True,
           tensor_sha256=hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest().upper()))
    c.csvwrite(c.O/"aux/PARENT_TENSOR_LINEAGE_319.csv",lineage)
    scope_checks=[]
    for g in groups:
        m=child.model[g["block_index"]];s=g["scope_name"];typ=type(m).__name__
        width=m.conv.out_channels if typ=="Conv" else m.cv1.conv.out_channels if s in ("main_hidden","hidden") else m.cv2.conv.out_channels if s=="skip_hidden" or typ=="SPPF" else m.cv3.conv.out_channels
        assert width==w[g["group_id"]]
        scope_checks.append(dict(group_id=g["group_id"],expected=width,actual=width,legal=width in r1.parse_allowed(g["allowed_widths"])))
    for m in child.model:
        typ=type(m).__name__
        if typ in ("C3","C3Lite"):
            main,skip=m.cv1.conv.out_channels,m.cv2.conv.out_channels
            assert m.cv3.conv.in_channels==main+skip
            for u in m.m:
                if typ=="C3":assert u.cv1.conv.in_channels==u.cv1.conv.out_channels==u.cv2.conv.in_channels==u.cv2.conv.out_channels==main
                else:assert u.dw.groups==u.dw.in_channels==u.dw.out_channels==u.pw.in_channels==u.pw.out_channels==u.bn.num_features==main
        if typ=="SPPF":assert m.cv2.conv.in_channels==4*m.cv1.conv.out_channels
    for n,m in child.named_modules():
        if hasattr(m,"conv") and hasattr(m,"bn"):assert m.conv.out_channels==m.bn.num_features,n
    detect=child.model[-1]
    assert detect.nc==10 and detect.na==3 and detect.no==15 and detect.f==[17,20,23]
    assert [x.out_channels for x in detect.m]==[45,45,45]
    assert torch.equal(detect.anchors,source.model[-1].anchors) and torch.equal(child.stride,source.stride)
    traces=[];hooks=[]
    def shapes(x):
        if isinstance(x,torch.Tensor):return list(x.shape)
        if isinstance(x,(tuple,list)):return [shapes(y) for y in x]
        return type(x).__name__
    for i,m in enumerate(child.model):
        def hook(module,args,out,i=i):
            traces.append(dict(block_index=i,module_type=type(module).__name__,input_shape=shapes(args),output_shape=shapes(out)))
        hooks.append(m.register_forward_hook(hook))
    initial={k:v.clone() for k,v in actual.items()}
    with torch.no_grad():output=child(torch.zeros(1,3,640,640,dtype=torch.float32))
    for h in hooks:h.remove()
    raw_shapes=[list(t.shape) for t in output[1]]
    assert raw_shapes==[[1,3,80,80,15],[1,3,40,40,15],[1,3,20,20,15]]
    assert all(bool(torch.isfinite(t).all()) for t in output[1]) and bool(torch.isfinite(output[0]).all())
    assert all(torch.equal(initial[k],child.state_dict()[k]) for k in initial)
    c.write(c.O/"aux/SYNTHETIC640_INTERMEDIATE_GRAPH.json",traces)
    from thop import profile
    fused=copy.deepcopy(child).eval().fuse()
    params=sum(x.numel() for x in fused.parameters());stride=int(fused.stride.max())
    macs,_=profile(fused,inputs=(torch.zeros(1,3,stride,stride),),verbose=False)
    raw_flops=int(round(2*float(macs)*(640/stride)**2))
    assert (params,raw_flops)==(selected["params"],selected["raw_flops"])
    canonical=c.O/"models/SELECTED_JOINT_P345_UNTRAINED.pt"
    shutil.copy2(raw,canonical);assert c.sha(raw)==c.sha(canonical)
    reload=torch.load(canonical,map_location="cpu",weights_only=False)["model"].float().eval()
    assert all(torch.equal(actual[k],reload.state_dict()[k]) for k in actual)
    with torch.no_grad():second=reload(torch.zeros(1,3,640,640))
    max_error=max(float((a-b).abs().max()) for a,b in zip(output[1],second[1]))
    assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(output[1],second[1]))
    assert not torch.cuda.is_initialized()
    c.write(c.O/"17_BEST_JOINT_CANDIDATE_PHYSICAL_CONSTRUCTION_INDEX_AND_TENSOR_AUDIT.json",dict(
        status="PHYSICAL_UNTRAINED_VALIDATED",candidate=selected["candidate_id"],width_signature=selected["signature"],
        checkpoint=str(canonical),checkpoint_sha256=c.sha(canonical),source_parent=str(c.PARENT),source_parent_sha256=c.sha(c.PARENT),
        exact_state_tensors=319,parent_index_slice_parity=True,scope_checks=scope_checks,retained_index_source="Original frozen Taylor + unchanged R1 build_pruning_plan",
        original_pruner_path=str(c.PRUNER),original_pruner_sha256=c.sha(c.PRUNER),epoch=-1,optimizer=None,teacher_forward=0,
        v4_checkpoint_loaded=False,training=False,backward=False,dataset_images=0))
    c.write(c.O/"18_SYNTHETIC_FORWARD_PHYSICAL_PARAMS_FLOPS_PARITY.json",dict(status="PHYSICAL_UNTRAINED_VALIDATED",device="cpu",dtype="FP32",
        input=[1,3,640,640],decoded_shape=list(output[0].shape),raw_shapes=raw_shapes,finite=True,
        output_channels_per_scale=[45,45,45],outputs_per_anchor=15,anchors_per_scale=3,
        correction="45 channels per scale, not per anchor. Source nc10 implies15 outputs per anchor.",
        anchors_preserved=True,strides=child.stride.tolist(),physical_fused_params=params,physical_raw_flops=raw_flops,
        THOP_MACs_at_stride=float(macs),stride=stride,accounting="Original Stage10 scaled-stride THOP convention",
        reload_tensor_exact=True,reload_forward_max_abs_error=max_error,CPU_initialized_CUDA_context=False,
        optimizer_steps=0,backward_calls=0,real_image_inference=0,AP_evaluations=0))
    print("PHYSICAL_UNTRAINED_PASS",params,raw_flops,"319 tensors","SHA",c.sha(canonical),flush=True)
if __name__=="__main__":main()
