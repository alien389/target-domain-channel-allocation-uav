"""Independent reference uses original custom-pruner operations, not the adapter."""
import copy
import torch
import torch_pruning as tp
from torch_pruning.pruner import function
from models.common import Conv,C3,C3Lite,SPPF,Concat
from models.yolo import Detect
from pruning_research.pruning import structured_pruner as original


def original_reference(parent,genes,lineages):
    model=copy.deepcopy(parent).cpu().float().eval()
    original_widths={}
    outputs={}
    keep={(g["block_name"],g["scope_name"]):lineages[g["group_id"]] for g in genes}
    def source(i,f):return i-1 if f==-1 else f
    for i,block in enumerate(parent.model):
        if isinstance(block,Conv):width=block.conv.out_channels
        elif isinstance(block,(C3,C3Lite)):width=block.cv3.conv.out_channels
        elif isinstance(block,SPPF):width=block.cv2.conv.out_channels
        elif isinstance(block,torch.nn.Upsample):width=original_widths[source(i,block.f)]
        elif isinstance(block,Concat):width=sum(original_widths[source(i,f)] for f in block.f)
        elif isinstance(block,Detect):width=135
        else:raise RuntimeError(f"REFERENCE_UNSUPPORTED:{type(block).__name__}")
        original_widths[i]=width
    for i,block in enumerate(model.model):
        name=f"model.{i}"
        if isinstance(block,Detect):
            original._adapt_detect_branch_inputs(block,outputs,i)
            continue
        if i==0:input_keep=list(range(3))
        elif isinstance(block.f,int):input_keep=outputs[source(i,block.f)]
        else:input_keep=[]
        if isinstance(block,torch.nn.Upsample):outputs[i]=list(input_keep)
        elif isinstance(block,Concat):
            offset=0;values=[]
            for f in block.f:
                s=source(i,f)
                values.extend(offset+j for j in outputs[s]);offset+=original_widths[s]
            outputs[i]=values
        elif isinstance(block,Conv):
            output=keep[(name,"output")]
            original._prune_conv_block(block,input_keep,output);outputs[i]=output
        elif isinstance(block,C3Lite):
            output=keep[(name,"output")]
            original._prune_c3lite_block(block,input_keep,keep[(name,"main_hidden")],keep[(name,"skip_hidden")],output);outputs[i]=output
        elif isinstance(block,C3):
            output=keep[(name,"output")]
            original._prune_c3_block(block,input_keep,keep[(name,"main_hidden")],keep[(name,"skip_hidden")],output);outputs[i]=output
        elif isinstance(block,SPPF):
            output=keep[(name,"output")]
            original._prune_sppf_block(block,input_keep,keep[(name,"hidden")],output);outputs[i]=output
        else:raise RuntimeError("REFERENCE_UNSUPPORTED_BLOCK")
    return model.eval()


def tensor_and_shape_compare(actual,reference):
    a,b=actual.state_dict(),reference.state_dict()
    errors=[]
    if set(a)!=set(b):errors.append("STATE_KEYS_DIFFER")
    for name in sorted(set(a)&set(b)):
        if a[name].shape!=b[name].shape or not torch.equal(a[name],b[name]):errors.append(name)
    ar=dict(actual.named_modules());br=dict(reference.named_modules())
    for name,layer in ar.items():
        other=br.get(name)
        if type(layer)!=type(other):errors.append(f"MODULE_CLASS:{name}");continue
        if isinstance(layer,torch.nn.Conv2d):
            if (layer.in_channels,layer.out_channels,layer.groups,layer.kernel_size)!=(other.in_channels,other.out_channels,other.groups,other.kernel_size):errors.append(f"CONV_METADATA:{name}")
        if isinstance(layer,torch.nn.BatchNorm2d):
            if (layer.num_features,layer.eps,layer.momentum)!=(other.num_features,other.eps,other.momentum):errors.append(f"BN_METADATA:{name}")
    return errors


def manual_native_saliency(group,width):
    """Float64 mathematical check of exactly the installed API's recognized terms."""
    summed=torch.zeros(width,dtype=torch.float64)
    count=0;components=[];ignored=[]
    for item in group:
        dep,idxs=item
        layer=dep.layer;fn=dep.pruning_fn
        values=None;kind=None
        if isinstance(layer,torch.nn.Conv2d) and fn==function.prune_conv_out_channels:
            values=layer.weight.detach().double()[idxs].flatten(1).abs().sum(1);kind="Conv_output"
        elif isinstance(layer,torch.nn.Conv2d) and fn==function.prune_conv_in_channels:
            values=layer.weight.detach().double().transpose(0,1).flatten(1).abs().sum(1)[idxs];kind="Conv_input"
        elif isinstance(layer,torch.nn.modules.batchnorm._BatchNorm) and fn==function.prune_batchnorm_out_channels and layer.affine:
            values=layer.weight.detach().double()[idxs].abs();kind="BN_weight"
        elif isinstance(layer,torch.nn.Conv2d):ignored.append({"handler":fn.__qualname__,"groups":layer.groups})
        if values is not None:
            summed.index_add_(0,torch.tensor(item.root_idxs),values)
            count+=1;components.append(kind)
    if count==0:raise RuntimeError("NO_NATIVE_SALIENCY_COMPONENTS")
    return summed/count,{"components":components,"native_ignored_conv_handlers":ignored}
