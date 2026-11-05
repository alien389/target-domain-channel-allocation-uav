"""One prospective, author-selected constant-coefficient KD2-style objective."""
import torch
import torch.nn.functional as F


def objectness_weights(prob,indices):
    weights=torch.full_like(prob,0.005)
    b,a,j,i=indices
    if b.numel():
        weights[b,a,j,i]=1.0
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0:continue
                jj=(j+dy).clamp(0,prob.shape[2]-1);ii=(i+dx).clamp(0,prob.shape[3]-1)
                weights[b,a,jj,ii]=torch.maximum(weights[b,a,jj,ii],torch.full_like(weights[b,a,jj,ii],0.35))
    return torch.where(prob>0.40,torch.maximum(weights,torch.full_like(weights,0.60)),weights)


def kd_terms(student,teacher,targets,compute_loss):
    if len(student)!=3 or len(teacher)!=3:raise RuntimeError('KD_RAW_THREE_SCALES_REQUIRED')
    _,_,indices,_=compute_loss.build_targets(student,targets)
    obj=[];box=[]
    for s,t,ii in zip(student,teacher,indices):
        if s.shape!=t.shape or s.shape[-1]!=15:raise RuntimeError('KD_RAW_SCHEMA_MISMATCH')
        t=t.detach();b,a,j,i=ii
        prob=t[...,4].sigmoid()
        obj.append((F.binary_cross_entropy_with_logits(s[...,4],prob,reduction='none')*objectness_weights(prob,ii)).mean())
        if b.numel():box.append(F.smooth_l1_loss(s[b,a,j,i,:4],t[b,a,j,i,:4],beta=1.0,reduction='mean'))
    lo=sum(obj)/len(obj)
    lb=sum(box)/len(box) if box else student[0][...,0].sum()*0.0
    return 0.20*lo+0.20*lb,lo,lb,indices


