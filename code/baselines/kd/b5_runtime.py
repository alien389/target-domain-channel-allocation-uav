"""KD adapter and disposable zero-optimizer-step runtime witness."""
import copy
import inspect
import math
import random
import time
import numpy as np
import torch
import b5_common as c
from b5_loss import kd_terms


class ProbeDone(BaseException):pass


def rng():
    ns=np.random.get_state()
    return c.digest({'python':random.getstate(),'numpy':[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],'cpu':torch.get_rng_state().tolist(),'cuda':[s.tolist() for s in torch.cuda.get_rng_state_all()]})


def detect(m):
    d=m.model[-1]
    return {'nc':int(d.nc),'no':int(d.no),'nl':int(d.nl),'na':int(d.na),'names':m.names,'stride':d.stride.detach().cpu().tolist(),'anchors_normalized':d.anchors.detach().cpu().tolist(),'anchors_pixels':(d.anchors*d.stride.view(-1,1,1)).detach().cpu().tolist(),'head_outputs':[v.out_channels for v in d.m],'state_keys':len(m.state_dict())}


class KDContext:
    def __init__(self,model,compute_loss,device,output):
        before=rng()
        if c.sha(c.TEACHER)!=c.TEACHER_SHA:raise RuntimeError('SEARCHPARENT_HASH_CHANGED')
        self.teacher=torch.load(c.TEACHER,map_location='cpu',weights_only=False)['model'].float().to(device).eval()
        for p in self.teacher.parameters():p.requires_grad_(False)
        self.compute_loss=compute_loss;self.output=output;self.batches=0;self.last={};self.teacher_initial=c.state_digest(self.teacher.state_dict())
        if before!=rng():raise RuntimeError('TEACHER_LOAD_CONSUMED_RNG')
        td,sd=self.teacher.model[-1],model.model[-1]
        if not torch.equal(td.anchors,sd.anchors) or not torch.equal(td.stride,sd.stride) or self.teacher.names!=model.names:raise RuntimeError('POST_AUTOANCHOR_TEACHER_STUDENT_MISMATCH')
        if td.nc!=10 or sd.nc!=10 or td.no!=15 or sd.no!=15:raise RuntimeError('DETECT_SEMANTICS')
        self.bn={n:c.tensor_sha(v) for n,v in self.teacher.named_buffers()}
    def terms(self,pred,imgs,targets):
        r=rng()
        with torch.no_grad():out=self.teacher(imgs)
        if r!=rng():raise RuntimeError('TEACHER_FORWARD_CONSUMED_RNG')
        if self.teacher.training or any(p.requires_grad or p.grad is not None for p in self.teacher.parameters()):raise RuntimeError('TEACHER_FREEZE_VIOLATION')
        if any(c.tensor_sha(v)!=self.bn[n] for n,v in self.teacher.named_buffers()):raise RuntimeError('TEACHER_BUFFER_CHANGED')
        raw=out[1]
        total,obj,box,indices=kd_terms(pred,raw,targets,self.compute_loss)
        return total,obj,box,indices,raw
    def loss(self,pred,imgs,targets):
        total,obj,box,indices,raw=self.terms(pred,imgs,targets)
        self.last={'batch':self.batches,'obj':float(obj.detach()),'box':float(box.detach()),'weighted_total':float(total.detach()),'positive_counts':[int(x[0].numel()) for x in indices],'teacher_raw_shapes':[list(x.shape) for x in raw],'teacher_raw_dtype':[str(x.dtype) for x in raw]}
        if not all(math.isfinite(self.last[k]) for k in ('obj','box','weighted_total')):raise RuntimeError('NONFINITE_KD_COMPONENT')
        self.batches+=1
        path=c.own(self.output/'kd_batch_terms.csv')
        import csv
        with path.open('a',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['batch','obj','box','weighted_total','positive_counts'])
            if f.tell()==0:w.writeheader()
            w.writerow({k:self.last[k] for k in w.fieldnames})
        return total


def setup_b5(model,opt,compute_loss,device):
    frame=inspect.currentframe().f_back
    # Only the authenticated passive-cost wrapper may be transparent to this
    # guard. Still require the real train frame and exact frozen trainer path.
    if frame.f_code.co_name=='observed_setup' and frame.f_code.co_filename==str(c.OUT/'run_ft15.py'):
        frame=frame.f_back
    if frame.f_code.co_name!='train' or frame.f_code.co_filename!=str(c.TRAINER):raise RuntimeError('UNEXPECTED_TRAINING_BOUNDARY')
    output=c.own(opt.save_dir);output.mkdir(parents=True,exist_ok=True)
    if str(output)!=opt.b5_expected_output:raise RuntimeError('UNAPPROVED_OUTPUT_AUTOINCREMENT')
    ck=torch.load(opt.weights,map_location='cpu',weights_only=False)
    original=ck['model'].float();wanted=copy.deepcopy(original).half().float()
    if c.state_digest(model.state_dict())!=c.state_digest(wanted.state_dict()):raise RuntimeError('START_STATE_CHANGED_AFTER_AUTOANCHOR')
    opti=frame.f_locals['optimizer'];ema=frame.f_locals['ema'];scaler=frame.f_locals['scaler'];loader=frame.f_locals['train_loader']
    if type(opti).__name__!='SGD' or opti.state or len(loader.dataset)!=6471 or len(loader)!=405 or loader.num_workers!=4:raise RuntimeError('FT15_START_POLICY')
    ids={id(p):n for n,p in model.named_parameters()}
    groups=[{k:v for k,v in g.items() if k!='params'}|{'parameters':[ids[id(p)] for p in g['params']]} for g in opti.param_groups]
    if any(g['lr']!=0.0001 or not g['nesterov'] or g['momentum']!=.937 for g in groups):raise RuntimeError('FT15_OPTIMIZER_LR')
    if frame.f_locals['nw']!=-1 or frame.f_locals['accumulate']!=4 or not frame.f_locals['amp']:raise RuntimeError('FT15_WARMUP_ACCUMULATION_AMP')
    if any(frame.f_locals['scheduler'].lr_lambdas[0](e)!=1.0 for e in range(16)):raise RuntimeError('FT15_CONSTANT_LR')
    if c.state_digest(ema.ema.state_dict())!=c.state_digest(original.state_dict()):raise RuntimeError('NEW_EMA_INITIALIZATION')
    before=rng();ctx=KDContext(model,compute_loss,device,output) if opt.b5_kd_enabled else None
    if before!=rng():raise RuntimeError('BRANCH_SETUP_CHANGED_RNG')
    witness={'created':c.now(),'seed':opt.seed,'kd_enabled':opt.b5_kd_enabled,'initial_file':opt.weights,'initial_SHA256':c.sha(opt.weights),'effective_FP32_state_digest':c.state_digest(model.state_dict()),'new_EMA_state_digest':c.state_digest(ema.ema.state_dict()),'319_keys':len(model.state_dict()),'detect':detect(model),'optimizer_groups':groups,'optimizer_state_entries':len(opti.state),'constant_lr_values':[frame.f_locals['scheduler'].lr_lambdas[0](e)*0.0001 for e in range(16)],'warmup_iterations':frame.f_locals['nw'],'accumulate':frame.f_locals['accumulate'],'AMP':frame.f_locals['amp'],'GradScaler_initial_scale':scaler.get_scale(),'train_images':len(loader.dataset),'batches':len(loader),'workers':loader.num_workers,'rng_at_boundary':rng(),'no_validation_loader':frame.f_locals['val_loader'] is None,'optimizer_steps':0,'teacher':detect(ctx.teacher) if ctx else None,'teacher_owns_optimizer_parameters':False,'teacher_load_and_setup_RNG_unchanged':True}
    c.write(output/'PREUPDATE_WITNESS.json',witness)
    if not getattr(opt,'b5_probe',False):return ctx
    batch=next(iter(loader));imgs,targets,paths,_=batch
    batchsha=c.tensor_sha(imgs);targetsha=c.tensor_sha(targets)
    imgs=imgs.to(device,non_blocking=True).float()/255;targets=targets.to(device)
    model.train();opti.zero_grad();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    with torch.cuda.amp.autocast(frame.f_locals['amp']):
        pred=model(imgs);supervised,items=compute_loss(pred,targets)
        duplicate,duplicate_items=compute_loss(pred,targets)
        if not torch.equal(supervised,duplicate) or not torch.equal(items,duplicate_items):raise RuntimeError('KD_OFF_SUPERVISED_LOSS_PARITY')
        kd=obj=box=None;raw=None
        if ctx:
            kd,obj,box,indices,raw=ctx.terms(pred,imgs,targets);total=supervised+kd
        else:
            total=supervised;_,_,indices,_=compute_loss.build_targets(pred,targets)
    params=[p for p in model.parameters() if p.requires_grad]
    direct=torch.autograd.grad(supervised,params,retain_graph=True,allow_unused=True)
    off=torch.autograd.grad(duplicate,params,retain_graph=True,allow_unused=True)
    off_equal=all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(direct,off))
    if not off_equal:raise RuntimeError('KD_OFF_GRADIENT_PARITY')
    if any(g is not None and not torch.isfinite(g).all() for g in direct):raise RuntimeError('UNSCALED_SUPERVISED_GRAD_NONFINITE')
    kd_nonzero=None;kd_finite=None
    if ctx:
        kg=torch.autograd.grad(kd,params,retain_graph=True,allow_unused=True)
        kd_nonzero=any(g is not None and bool(g.count_nonzero()) for g in kg);kd_finite=all(g is None or bool(torch.isfinite(g).all()) for g in kg)
        if not kd_nonzero or not kd_finite or float(obj)<=0 or float(box)<=0:raise RuntimeError('KD_GRADIENT_OR_COMPONENTS_INVALID')
        del kg
    total_grad=torch.autograd.grad(total,params,retain_graph=True,allow_unused=True)
    if not all(g is None or bool(torch.isfinite(g).all()) for g in total_grad):raise RuntimeError('TOTAL_UNSCALED_GRAD_NONFINITE')
    del direct,off,total_grad
    scaler.scale(total).backward();scaler.unscale_(opti)
    found={str(k):float(v) for k,v in scaler._found_inf_per_device(opti).items()}
    scaled_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
    scale_before=scaler.get_scale();scaler.update();scale_after=scaler.get_scale()
    overflow=sum(found.values())>0
    if overflow and (scaled_finite or scale_after>=scale_before):raise RuntimeError('GRADSCALER_OVERFLOW_HANDLING_INVALID')
    if not overflow and not scaled_finite:raise RuntimeError('UNDETECTED_GRADSCALER_NONFINITE')
    if opti.state:raise RuntimeError('UNAUTHORIZED_OPTIMIZER_UPDATE')
    if ctx and (any(p.grad is not None for p in ctx.teacher.parameters()) or ctx.teacher_initial!=c.state_digest(ctx.teacher.state_dict())):raise RuntimeError('TEACHER_MUTATION')
    for s,t,ii in zip(pred,raw if raw else pred,indices):
        b,a,j,i=ii
        if s.shape!=t.shape or bool(((b<0)|(b>=16)).any()) or bool(((a<0)|(a>=3)).any()) or bool(((j<0)|(j>=s.shape[2])).any()) or bool(((i<0)|(i>=s.shape[3])).any()):raise RuntimeError('POSITIVE_ASSIGNMENT_AXIS_ERROR')
    # No optimizer/scheduler call. Save the untouched fresh EMA, not trained weights.
    temp=c.own(output/'disposable_EMA_write_reload.pt');torch.save({'epoch':14,'model':copy.deepcopy(ema.ema).half(),'ema':None,'optimizer':None},temp)
    reloaded=torch.load(temp,map_location='cpu',weights_only=False)['model'].float()
    if c.state_digest(reloaded.state_dict())!=c.state_digest(copy.deepcopy(ema.ema).cpu().half().float().state_dict()):raise RuntimeError('EMA_SERIALIZATION_ROUNDTRIP')
    proof={'status':'PASS','created':c.now(),'seed':opt.seed,'kd_enabled':opt.b5_kd_enabled,'actual_augmented_image_shape':list(imgs.shape),'paths':[str(p) for p in paths],'augmented_uint8_batch_SHA256':batchsha,'augmented_targets_SHA256':targetsha,'raw_shapes':[list(x.shape) for x in pred],'raw_dtype':[str(x.dtype) for x in pred],'supervised_loss':float(supervised.detach()),'kd_obj':float(obj.detach()) if obj is not None else None,'kd_box':float(box.detach()) if box is not None else None,'kd_weighted':float(kd.detach()) if kd is not None else None,'total_loss':float(total.detach()),'positive_counts':[int(x[0].numel()) for x in indices],'KD_OFF_exact_loss_and_gradients':off_equal,'unscaled_student_gradients_finite':True,'KD_student_gradients_nonzero':kd_nonzero,'KD_student_gradients_finite':kd_finite,'teacher_parameters_all_frozen':bool(ctx) and not any(p.requires_grad for p in ctx.teacher.parameters()),'teacher_gradients_absent':bool(ctx) and not any(p.grad is not None for p in ctx.teacher.parameters()),'teacher_BN_and_all_state_unchanged':bool(ctx) and ctx.teacher_initial==c.state_digest(ctx.teacher.state_dict()),'teacher_RNG_unchanged':True,'optimizer_state_entries':len(opti.state),'optimizer_steps':0,'scheduler_steps':0,'epochs_completed':0,'GradScaler':{'initial_scale':scale_before,'found_inf_per_device':found,'scaled_grads_finite':scaled_finite,'update_scale':scale_after,'overflow_handled_without_optimizer_step':overflow,'limitation':'Dynamic initial-scale overflow, if observed, is not hidden or fixed by changing training scale. Unscaled gradient finiteness was tested separately. Original training scaler skip/update policy remains unchanged.'},'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),'probe_wall_seconds':time.perf_counter()-t0,'GPU_name':torch.cuda.get_device_name(0),'new_ema_write_reload_SHA256':c.sha(temp),'no_training_checkpoint_selected':True}
    c.write(output/'REAL_BATCH_GRADIENT_PROOF.json',proof)
    it=getattr(loader,'iterator',None)
    if it is not None and hasattr(it,'_shutdown_workers'):it._shutdown_workers()
    raise ProbeDone()


