"""One matched FT15 run or an authorized disposable no-update runtime probe."""
import argparse
import copy
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import b5_common as c
import yaml
import numpy as np
import torch
import cv2
from b5_runtime import ProbeDone
torch.set_num_threads(4)

class EntryProbeDone(BaseException):pass


def flags():
    return {'matmul_TF32':torch.backends.cuda.matmul.allow_tf32,'cudnn_TF32':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'env':{k:os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVIDIA_TF32_OVERRIDE','CUDA_VISIBLE_DEVICES','RANK','LOCAL_RANK','WORLD_SIZE')}}


def environment():
    before=flags();initialized=torch.cuda.is_initialized();version=torch.backends.cudnn.version();after=flags()
    if before!=after:raise RuntimeError('METADATA_QUERY_CHANGED_FLAGS')
    packages={x.metadata['Name'].lower():x.version for x in importlib.metadata.distributions() if x.metadata['Name']}
    stable={'python':sys.version,'torch':torch.__version__,'torchvision':importlib.metadata.version('torchvision'),'numpy':np.__version__,'cv2':cv2.__version__,'CUDA':torch.version.cuda,'cuDNN':version,'OS':platform.platform(),'flags':after,'packages':packages,'driver':subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()}
    return {'created':c.now(),'stable':stable,'fingerprint':c.digest(stable),'python_executable':sys.executable,'sys_path':sys.path,'cwd':str(Path.cwd()),'numpy_path':np.__file__,'cv2_path':cv2.__file__,'torch_path':torch.__file__,'metadata_CUDA_context_before':initialized,'metadata_CUDA_context_after':torch.cuda.is_initialized(),'context_initialization_is_metadata_not_inference':True}


def dataset(path,*args,**kwargs):
    d=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if 'val' in d or d['nc']!=10 or not Path(d['train']).is_file():raise RuntimeError('TRAIN_ONLY_DATA_ROLE')
    d['val']=None
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--probe',action='store_true');p.add_argument('--entry-probe',action='store_true');p.add_argument('--probe-tag',default='');a=p.parse_args()
    if a.probe and a.entry_probe:raise RuntimeError('ONE_TECHNICAL_PROBE_MODE_ONLY')
    config=c.load(a.config);job=config['run_id']
    if not (a.probe or a.entry_probe):
        matrix=c.verified_matrix()
        if next(x for x in matrix['jobs'] if x['run_id']==job)['config_path']!=str(Path(a.config).resolve()):raise RuntimeError('UNFROZEN_JOB_CONFIG')
    if c.sha(config['initialization'])!=config['initialization_SHA256']:raise RuntimeError('ORIGINAL_SEED_EMA_CHANGED')
    output=c.OUT/('TECHNICAL_PREFLIGHT' if a.probe else 'ENTRY_PREFLIGHT')/a.probe_tag/job if (a.probe or a.entry_probe) else Path(config['output_directory'])
    if output.exists():raise RuntimeError('OUTPUT_COLLISION_NO_AUTO_INCREMENT_OR_RESUME:'+str(output))
    meta=c.OUT/'RUNTIME_METADATA'/('PROBE' if a.probe else 'ENTRY' if a.entry_probe else 'TRAIN')/a.probe_tag/job
    meta.mkdir(parents=True,exist_ok=False)
    os.chdir(c.YOLO)
    before=environment();m=c.module('b5_prospective_train',c.TRAINER);after=environment()
    c.write(meta/'ENVIRONMENT_BEFORE_AFTER_IMPORTS.json',{'before':before,'after':after})
    if not (a.probe or a.entry_probe):
        expected=c.load(c.OUT/'FROZEN_ENVIRONMENT.json')
        if before['fingerprint']!=expected['before_fingerprint'] or after['fingerprint']!=expected['after_fingerprint']:raise RuntimeError('PROSPECTIVE_ENVIRONMENT_CHANGED')
    m.ROOT=c.YOLO;m.check_dataset=dataset;m.strip_optimizer=lambda *x,**k:None
    # No downloads/installations or network-version checks in operational plumbing.
    m.check_git_status=lambda *x,**k:None
    requirements=m.check_requirements
    m.check_requirements=lambda *x,**k:requirements(*x,**dict(k,install=False))
    template=torch.load(config['initialization'],map_location='cpu',weights_only=False)['model'].float()
    factory_calls=[]
    def factory(*x,**k):factory_calls.append(1);return copy.deepcopy(template)
    m.Model=factory
    def local_weight(path,*x,**k):
        if str(Path(path).resolve())!=str(Path(config['initialization']).resolve()) or not Path(path).is_file():raise RuntimeError('UNAPPROVED_MODEL_DOWNLOAD_OR_SUBSTITUTION')
        return str(path)
    m.attempt_download=local_weight
    import models.common as common
    original_backend=common.DetectMultiBackend
    class NoFallbackBackend(original_backend):
        def __init__(self,*x,**k):raise RuntimeError('AMP_FALLBACK_OTHER_MODEL_FORBIDDEN')
    common.DetectMultiBackend=NoFallbackBackend
    ampcheck=m.check_amp
    def amp(model):
        h=c.state_digest(model.state_dict());value=ampcheck(model)
        if not value or c.state_digest(model.state_dict())!=h:raise RuntimeError('SOURCE_AMP_CHECK_FALSE_OR_MODEL_MUTATION')
        return value
    m.check_amp=amp
    loader=m.create_dataloader
    def checked_loader(*x,**k):
        if k.get('augment') is not True or k.get('rect') is not False or k.get('shuffle') is not True or k.get('workers')!=4 or k.get('seed')!=config['seed'] or int(x[1])!=640 or int(x[2])!=16:raise RuntimeError('TRAIN_LOADER_POLICY')
        expected=c.OUT/'INPUTS/train6471_ordered.txt'
        if Path(x[0]).resolve()!=expected:raise RuntimeError('NONTRAIN_OR_UNAPPROVED_DATA_LIST')
        cp=expected.with_suffix('.cache');h=c.sha(cp)
        from utils.dataloaders import get_hash,img2label_paths,LoadImagesAndLabels
        images=sorted(str(Path(s)) for s in expected.read_text().splitlines() if s.strip());cache=np.load(cp,allow_pickle=True).item()
        if cache['hash']!=get_hash(img2label_paths(images)+images) or cache['version']!=LoadImagesAndLabels.cache_version:raise RuntimeError('READONLY_DATA_CACHE_INVALID_NO_REGENERATION')
        result=loader(*x,**k)
        if c.sha(cp)!=h:raise RuntimeError('LOCAL_VALIDATED_CACHE_CHANGED')
        return result
    m.create_dataloader=checked_loader
    anchor=m.check_anchors
    def aa(ds,model,**kw):
        before=model.model[-1].anchors.detach().cpu().clone();r=anchor(ds,model=model,**kw)
        c.write(meta/'AUTOANCHOR_ACTUAL.json',{'train_images':len(ds),'before_normalized':before.tolist(),'after_normalized':model.model[-1].anchors.detach().cpu().tolist(),'stride':model.model[-1].stride.detach().cpu().tolist(),'original_source':str(c.YOLO/'utils/autoanchor.py'),'SHA256':c.sha(c.YOLO/'utils/autoanchor.py'),'source_function_actually_executed':True})
        return r
    m.check_anchors=aa
    # Passive training-cost observation only. The loss/RNG assertions stay in
    # the tested adapter. No optimizer policy or tensor is modified by hooks.
    teacher_events=[];teacher_timings=[];original_setup=m.setup_b5;entry_state={}
    def observed_setup(*x,**k):
        if a.entry_probe:
            import inspect
            train_frame=inspect.currentframe().f_back
            entry_state.update(optimizer=train_frame.f_locals['optimizer'],ema=train_frame.f_locals['ema'],loader=train_frame.f_locals['train_loader'],parameters_before=c.digest({n:c.tensor_sha(v) for n,v in x[0].named_parameters()}))
        ctx=original_setup(*x,**k)
        if ctx is not None:
            def start_teacher(mod,inputs):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();teacher_events.append((start,end))
            def end_teacher(mod,inputs,output):teacher_events[-1][1].record()
            ctx.teacher.register_forward_pre_hook(start_teacher);ctx.teacher.register_forward_hook(end_teacher)
        return ctx
    m.setup_b5=observed_setup
    from utils.callbacks import Callbacks
    callbacks=Callbacks()
    def cost_batch(*x,**k):
        if teacher_events:
            start,end=teacher_events.pop(0);end.synchronize();teacher_timings.append(float(start.elapsed_time(end)))
        if a.entry_probe:
            model,ni,imgs,targets,paths,*_=x
            if ni!=0 or entry_state['optimizer'].state or entry_state['ema'].updates!=0:raise RuntimeError('ENTRY_PROBE_UPDATE_OR_BATCH_LIMIT')
            if c.digest({n:c.tensor_sha(v) for n,v in model.named_parameters()})!=entry_state['parameters_before']:raise RuntimeError('ENTRY_PROBE_LEARNED_WEIGHT_CHANGED')
            c.write(output/'ACTUAL_LOOP_ENTRY_PROOF.json',{'status':'PASS','seed':config['seed'],'arm':config['arm'],'actual_normal_loop_batch':ni,'optimizer_steps':0,'scheduler_steps':0,'epochs_completed':0,'learned_parameter_bytes_unchanged':True,'EMA_updates':entry_state['ema'].updates,'teacher_forward_events':teacher_timings,'paths':[str(p) for p in paths],'actual_shared_runner_SHA256':c.sha(__file__),'guard_source_SHA256':c.sha(c.OUT/'b5_runtime.py'),'training_loop_source_SHA256':c.sha(c.TRAINER),'source':'Actual normal startup, observer wrapper, loss forward/backward and batch-end callback, interrupted before any optimizer opportunity.'})
            entry_state['loader'].iterator._shutdown_workers()
            raise EntryProbeDone()
    callbacks.register_action('on_train_batch_end','b5_passive_teacher_cost',cost_batch)
    argv=[str(c.TRAINER),'--img','640','--batch-size','16','--epochs','15','--data',config['data'],'--weights',config['initialization'],'--hyp',config['hyp'],'--seed',str(config['seed']),'--device','0','--workers','4','--patience','300','--save-period','25','--noplots','--project',str(output.parent),'--name',output.name,'--disable-validation']
    sys.argv=argv;opt=m.parse_opt();opt.b5_kd_enabled=config['kd_enabled'];opt.b5_probe=a.probe;opt.b5_expected_output=str(output)
    c.write(meta/'EFFECTIVE_OPTIONS.json',vars(opt));c.text(meta/'COMMAND.txt',subprocess.list2cmdline([sys.executable,'-B',str(c.OUT/'run_ft15.py'),'--config',str(Path(a.config).resolve()),*(['--probe'] if a.probe else [])])+'\n'+subprocess.list2cmdline([sys.executable,*argv])+'\n')
    t0=time.perf_counter();start=c.now();torch.cuda.reset_peak_memory_stats()
    try:m.main(opt,callbacks=callbacks)
    except (ProbeDone,EntryProbeDone):
        if not (a.probe or a.entry_probe):raise
        if (output/'weights/last.pt').exists() or c.epochs(output):raise RuntimeError('PROBE_UNAUTHORIZED_OPTIMIZER_EPOCH')
        c.write(meta/'PROBE_COMPLETE.json',{'status':'PASS','optimizer_steps':0,'epochs':0,'source_SHA256':c.sha(c.TRAINER)})
        print(('B5_V5_ACTUAL_ENTRY_ZERO_STEP_PASS=' if a.entry_probe else 'B5_V5_REAL_BATCH_ZERO_STEP_PROBE_PASS=')+job,flush=True);return
    if a.probe or a.entry_probe:raise RuntimeError('PROBE_DID_NOT_INTERRUPT_BEFORE_TRAINING')
    if len(factory_calls)!=1 or len(c.epochs(output))!=15:raise RuntimeError('FT15_FULL_HORIZON_OR_FACTORY_FAILED')
    if (output/'weights/best.pt').exists():raise RuntimeError('FORBIDDEN_VALIDATION_SELECTION')
    last=output/'weights/last.pt';ck=torch.load(last,map_location='cpu',weights_only=False)
    if ck['epoch']!=14 or ck['ema'] is None or len(ck['ema'].state_dict())!=319:raise RuntimeError('TERMINAL_EPOCH_EMA_INVALID')
    terminal=output/'weights/terminal_ema.pt'
    payload={'epoch':14,'best_fitness':None,'model':copy.deepcopy(ck['ema']).half(),'ema':None,'optimizer':None,'updates':ck['updates'],'opt':ck['opt'],'git':ck['git'],'date':c.now(),'b5_policy':'terminal EMA after15 new optimizer-reset FT epochs, cumulative exposure300+15, no validation selection','original_initialization_SHA256':config['initialization_SHA256'],'kd_enabled':config['kd_enabled']}
    torch.save(payload,terminal)
    record={'status':'PASS','run_id':job,'seed':config['seed'],'arm':config['arm'],'new_epochs':15,'epoch_sequence':list(range(15)),'original_initialization':config['initialization'],'original_initialization_SHA256':c.sha(config['initialization']),'effective_initial_state_digest':c.load(output/'PREUPDATE_WITNESS.json')['effective_FP32_state_digest'],'terminal_ema':str(terminal),'terminal_ema_SHA256':c.sha(terminal),'last':str(last),'last_SHA256':c.sha(last),'teacher_SHA256':c.sha(c.TEACHER) if config['kd_enabled'] else None,'source_SHA256':c.sha(c.TRAINER),'start_UTC':start,'end_UTC':c.now(),'wall_seconds':time.perf_counter()-t0,'duration_definition':'wall-clock train invocation including loader/serialization, NOT utilization-integrated GPU hours','peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),'EMA_updates':ck['updates'],'optimizer_step_count':'NOT_SEPARATELY_RECORDED_AMP_SKIPS_NOT_INFERRED_FROM_EMA','resume_policy':'No automatic resume, full RNG/GradScaler/scheduler state not serialized by source writer','validation_during_training':False,'AP_evaluated':False}
    if len(teacher_timings)!=(6075 if config['kd_enabled'] else 0):raise RuntimeError('TEACHER_FORWARD_COST_COVERAGE')
    c.csvout(output/'TEACHER_FORWARD_CUDA_EVENTS.csv',[{'batch':i,'teacher_forward_CUDA_ms':v} for i,v in enumerate(teacher_timings)],['batch','teacher_forward_CUDA_ms'])
    record['teacher_forward_CUDA_total_seconds']=sum(teacher_timings)/1000
    record['teacher_forward_cost_definition']='CUDA event spans during actual training, not a standalone deployment benchmark, wall cost also includes extra loss/audit/logging overhead'
    if c.sha(config['initialization'])!=config['initialization_SHA256'] or c.sha(c.TEACHER)!=c.TEACHER_SHA:raise RuntimeError('IMMUTABLE_INPUT_CHANGED')
    c.write(output/'TRAINING_COMPLETENESS.json',record)
    print('B5_V5_FT15_COMPLETE='+job,flush=True)


if __name__=='__main__':main()


