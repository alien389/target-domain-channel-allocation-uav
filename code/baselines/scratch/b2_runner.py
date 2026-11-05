"""Observer around the unchanged Stage10 worker, with a fail-closed scratch witness."""
import argparse
import copy
import inspect
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
import b2_common as c
sys.path[:0]=[str(c.YOLO),str(c.PROJECT/"scripts")]
os.chdir(c.YOLO)
import torch
import numpy as np
import cv2
import importlib.metadata


class ProbeBoundaryReached(BaseException):pass


def flags():
    return {"cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,"float32_matmul_precision":torch.get_float32_matmul_precision(),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,"environment":{k:os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE","NVIDIA_TF32_OVERRIDE","CUDA_VISIBLE_DEVICES","PYTHONHASHSEED","CUDNN_DETERMINISTIC","LOCAL_RANK","RANK","WORLD_SIZE")}}


def rng():
    n=np.random.get_state()
    return {"torch_cpu_bytes":torch.get_rng_state().tolist(),"torch_cuda_bytes":[x.tolist() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None,"python":random.getstate(),"numpy":{"algorithm":n[0],"state":n[1].tolist(),"position":n[2],"has_gauss":n[3],"cached_gaussian":n[4]}}


def environment():
    before=flags();cuda_before=torch.cuda.is_initialized()
    cudnn=torch.backends.cudnn.version()
    after=flags();cuda_after=torch.cuda.is_initialized()
    if before!=after:raise RuntimeError("METADATA_QUERY_CHANGED_PRECISION_FLAGS")
    return {"python_executable":sys.executable,"python":sys.version,"sys_path":list(sys.path),"cwd":str(Path.cwd()),"sys_prefix":sys.prefix,"CONDA_DEFAULT_ENV":os.environ.get("CONDA_DEFAULT_ENV"),"CONDA_PREFIX":os.environ.get("CONDA_PREFIX"),"PYTHONPATH":os.environ.get("PYTHONPATH"),"torch":torch.__version__,"torch_import":torch.__file__,"torchvision":importlib.metadata.version("torchvision"),"numpy":np.__version__,"numpy_import":np.__file__,"cv2":cv2.__version__,"cv2_import":cv2.__file__,"cuda":torch.version.cuda,"cudnn":cudnn,"gpu":torch.cuda.get_device_name(0),"OS":platform.platform(),"packages":{d.metadata["Name"].lower():d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},"flags":after,"gpu_driver_reading":subprocess.run(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.strip(),"dependency_source_hashes":{str(c.YOLO/p):c.sha(c.YOLO/p) for p in ("models/common.py","models/yolo.py","utils/dataloaders.py","utils/augmentations.py","utils/general.py","utils/torch_utils.py","utils/loss.py","utils/autoanchor.py")},"metadata_query":{"CUDA_before":cuda_before,"CUDA_after":cuda_after,"TF32_before":before,"TF32_after":after,"context_initialization_is_not_model_inference":True}}


def stable_env(record):return {k:v for k,v in record.items() if k not in ("metadata_query",)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-config",required=True);parser.add_argument("--environment-probe",action="store_true");parser.add_argument("--zero-step-probe",action="store_true");args=parser.parse_args()
    config=c.load(args.run_config)
    proposed=c.load(c.HERE/"PROPOSED_JOBS.json")
    seed=config["seed"]
    if c.sha(config["initialization"])!=config["initialization_sha256"]:raise RuntimeError("SCRATCH_INITIALIZER_HASH_CHANGED")
    if not args.environment_probe and not args.zero_step_probe:
        lock=c.load(c.OUT/"09_B2_EXACT_PRECOMMITTED_RUN_MATRIX.json")
        gate=c.load(c.OUT/"10_B2_PRETRAIN_GATE_DECISION.json")
        if gate["status"]!="PASS" or c.sha(c.OUT/"09_B2_EXACT_PRECOMMITTED_RUN_MATRIX.json")!=gate["run_matrix_sha256"]:raise RuntimeError("B2_PRETRAIN_GATE_NOT_PASS")
        for p,h in lock["code_and_input_sha256"].items():
            if c.sha(p)!=h:raise RuntimeError("B2_FROZEN_CODE_OR_INPUT_CHANGED:"+p)
    output=c.HERE/"ENVIRONMENT_PROBE" if args.environment_probe else c.OUT/f"TECHNICAL_PREFLIGHT/SEED{seed}_VERIFIED" if args.zero_step_probe else Path(config["output_directory"])
    if args.environment_probe or args.zero_step_probe:
        output.mkdir(parents=True,exist_ok=False)
    worker=c.module("b2_original_worker",c.WORKER)
    original_loader=worker.load_training_module
    kept={}
    def recorded_loader(source,initialization):
        before=environment();module=original_loader(source,initialization);after=environment()
        snapshot={"created_utc":c.now(),"seed":seed,"execution_label":c.LABEL,"before_original_training_imports":before,"after_original_training_imports":after,"stable_fingerprint":c.digest({"before":stable_env(before),"after":stable_env(after)})}
        if not args.environment_probe:
            golden=c.load(c.HERE/"ENVIRONMENT_PROBE/current_environment_before_after_imports.json")
            if snapshot["stable_fingerprint"]!=golden["stable_fingerprint"]:raise RuntimeError("B2_CURRENT_ENVIRONMENT_CHANGED_FROM_FROZEN_PROSPECTIVE_STACK")
        c.write(output/"current_environment_before_after_imports.json",snapshot)
        seed_impl=module.init_seeds
        def seed_observer(*a,**kw):
            result=seed_impl(*a,**kw)
            c.write(output/"runtime_seed_configuration.json",{"seed_arguments":a,"kwargs":kw,"rng_after_original_seeding":rng(),"flags":flags(),"observer_changes_to_RNG":False})
            return result
        module.init_seeds=seed_observer
        parse=module.parse_opt
        def parsed(*a,**kw):
            opt=parse(*a,**kw);value=json.loads(json.dumps(vars(opt),default=str))
            expected=copy.deepcopy(proposed["effective_arguments_by_seed"][str(seed)])
            if args.zero_step_probe:
                for field in ("project","name"):
                    expected[field]=str(output.parent) if field=="project" else output.name
            if value!=expected:raise RuntimeError("B2_EFFECTIVE_ARGUMENTS_CHANGED_BEFORE_UPDATE")
            c.write(output/"executed_effective_args.json",value)
            return opt
        module.parse_opt=parsed
        import utils.downloads as downloads
        import models.common as common
        old_download=downloads.attempt_download
        def local_only(path,*a,**kw):
            if not Path(path).is_file():raise RuntimeError("UNAPPROVED_ASSET_DOWNLOAD_FORBIDDEN:"+str(path))
            return old_download(path,*a,**kw)
        module.attempt_download=local_only;downloads.attempt_download=local_only
        if hasattr(common,"attempt_download"):common.attempt_download=local_only
        original_amp=module.check_amp
        def observed_amp(model):
            initial=c.state_digest(model.state_dict())
            if args.zero_step_probe:
                # Exercise the original function's built-in synthetic offline
                # branch, without reading bus.jpg or downloading another model.
                gl=original_amp.__globals__;oldroot=gl["ROOT"];online=gl["check_online"]
                original_backend=common.DetectMultiBackend
                class ForbiddenBackend(original_backend):
                    def __init__(self,*a,**kw):raise RuntimeError("PROBE_AMP_FALLBACK_PRETRAINED_MODEL_FORBIDDEN")
                gl["ROOT"]=output/"synthetic_asset_root_absent";gl["check_online"]=lambda *a,**kw:False;common.DetectMultiBackend=ForbiddenBackend
                try:value=original_amp(model)
                finally:gl["ROOT"]=oldroot;gl["check_online"]=online;common.DetectMultiBackend=original_backend
            else:value=original_amp(model)
            if c.state_digest(model.state_dict())!=initial:raise RuntimeError("AMP_CHECK_MUTATED_ACCEPTED_INITIAL_STATE")
            c.write(output/"runtime_amp_check.json",{"AMP_result":bool(value),"original_function_sha256":c.sha(c.YOLO/"utils/general.py"),"initial_model_unchanged":True,"source_defined_synthetic_offline_input_for_disposable_probe_only":bool(args.zero_step_probe),"actual_training_original_asset_and_function_unchanged":not args.zero_step_probe})
            if not value:raise RuntimeError("ORIGINAL_STAGE10_AMP_TRUE_CANNOT_EXECUTE")
            return value
        module.check_amp=observed_amp
        intersect=module.intersect_dicts
        def transfer(a,b,*values,**kw):
            result=intersect(a,b,*values,**kw)
            if set(result)!=set(b) or len(result)!=319:raise RuntimeError("B2_ACTUAL_TRANSFER_NOT_COMPLETE_META_SCHEMA")
            c.write(output/"runtime_state_transfer.json",{"source_keys":len(a),"target_keys":len(b),"transferred_keys":len(result),"missing":sorted(set(b)-set(result)),"initializer_sha256":c.sha(initialization),"all319_transferred":True})
            return result
        module.intersect_dicts=transfer
        dataloader=module.create_dataloader
        def loader(*a,**kw):
            before_rng=rng()
            if int(a[1])!=640 or int(a[2])!=16 or kw.get("rect") is not False or kw.get("shuffle") is not True or kw.get("workers")!=4 or kw.get("seed")!=seed:raise RuntimeError("B2_TRAIN_DATALOADER_POLICY_MISMATCH")
            expected=worker.train_only_dataset(config["train_only_yaml"])["train"]
            if Path(a[0]).resolve()!=Path(expected).resolve():raise RuntimeError("NONTRAIN_LOADER_FORBIDDEN")
            result=dataloader(*a,**kw);ld,ds=result
            if len(ds)!=6471 or len(ld)!=405 or ld.num_workers!=4:raise RuntimeError("B2_ACTUAL_TRAIN_LOADER_COUNTS")
            kept["loader"]=ld
            c.write(output/"runtime_dataloader_contract.json",{"images":len(ds),"batches":len(ld),"workers":ld.num_workers,"seed":seed,"rng_before":before_rng,"rng_after":rng(),"first_image_tensor_not_consumed_by_observer":True})
            return result
        module.create_dataloader=loader
        original_anchor=module.check_anchors
        def anchors(dataset,model,**kw):
            before_anchor=model.model[-1].anchors.detach().cpu().clone()
            result=original_anchor(dataset,model=model,**kw)
            c.write(output/"runtime_autoanchor.json",{"before":before_anchor.tolist(),"after":model.model[-1].anchors.detach().cpu().tolist(),"original_function":True,"source_sha256":c.sha(c.YOLO/"utils/autoanchor.py"),"dataset_images":len(dataset),"label_only_source_train":True,"arguments":kw})
            return result
        module.check_anchors=anchors
        compute=module.ComputeLoss
        def boundary(model,*a,**kw):
            loss=compute(model,*a,**kw)
            frame=inspect.currentframe().f_back
            if frame.f_code.co_filename!=str(c.TRAINER):raise RuntimeError("PREUPDATE_WITNESS_NOT_FROM_AUTHENTIC_TRAIN_FRAME")
            raw=torch.load(initialization,map_location="cpu",weights_only=False)["model"].float()
            wanted=copy.deepcopy(raw).half().float().state_dict()
            actual=model.state_dict()
            mapping=[]
            for k,v in actual.items():
                equal=torch.equal(v.detach().cpu(),wanted[k])
                is_anchor=k.endswith(".anchors")
                if not equal and not is_anchor:raise RuntimeError("PREUPDATE_SCRATCH_TREATMENT_REPLACED_OR_MUTATED:"+k)
                mapping.append({"key":k,"source_raw_tensor_sha256":c.tensor_sha(raw.state_dict()[k]),"expected_half_float_sha256":c.tensor_sha(wanted[k]),"effective_sha256":c.tensor_sha(v),"transform":"original_train_only_AutoAnchor_then_half_float" if is_anchor else "original_whole_model_half_float","exact_expected":equal,"is_trainable":k in dict(model.named_parameters())})
            opt=frame.f_locals["optimizer"]
            groupids={id(p):name for name,p in model.named_parameters()}
            optgroups=[{"parameters":[groupids[id(p)] for p in g["params"]],"lr":g["lr"],"initial_lr":g.get("initial_lr"),"momentum":g.get("momentum"),"nesterov":g.get("nesterov"),"weight_decay":g.get("weight_decay")} for g in opt.param_groups]
            if opt.state:raise RuntimeError("OPTIMIZER_STATE_ALREADY_UPDATED_BEFORE_ZERO_STEP_WITNESS")
            witness={"status":"PASS","seed":seed,"created_utc":c.now(),"boundary":"original train() ComputeLoss construction immediately before on_train_start, before any real model training batch or optimizer step","initializer":str(initialization),"initializer_sha256":c.sha(initialization),"source_raw_state_digest":c.state_digest(raw.state_dict()),"effective_preupdate_state_digest":c.state_digest(actual),"tensor_witness":mapping,"all319_checked":len(mapping)==319,"trainable_tensors":len(list(model.named_parameters())),"optimizer_type":type(opt).__name__,"optimizer_groups":optgroups,"optimizer_state_entries":len(opt.state),"optimizer_steps":0,"training_RNG_at_boundary":rng(),"train_loader_batches":len(frame.f_locals["train_loader"]),"warmup_iterations":frame.f_locals["nw"],"AMP":frame.f_locals["amp"],"scaler_initial_scale":frame.f_locals["scaler"].get_scale(),"source":str(c.TRAINER),"source_sha256":c.sha(c.TRAINER),"worker_sha256":c.sha(c.WORKER),"disposable_probe":bool(args.zero_step_probe),"accepted_scratch_checkpoint_unchanged":c.sha(initialization)==config["initialization_sha256"]}
            c.write(output/"preupdate_scratch_purity_witness.json",witness)
            if args.zero_step_probe:raise ProbeBoundaryReached()
            return loss
        module.ComputeLoss=boundary
        return module
    worker.load_training_module=recorded_loader
    if args.environment_probe:
        recorded_loader(c.TRAINER,Path(config["initialization"]))
        print("B2_ENVIRONMENT_PROBE_COMPLETE",flush=True);return
    if args.zero_step_probe:
        disposable=copy.deepcopy(config);disposable["output_directory"]=str(output)
        p=output/"disposable_exact_loader_config.json";c.write(p,disposable)
        sys.argv=[str(c.WORKER),"--run-config",str(p)]
        try:worker.main()
        except ProbeBoundaryReached:
            if c.sha(config["initialization"])!=config["initialization_sha256"]:raise RuntimeError("PROBE_MUTATED_FROZEN_SCRATCH_FILE")
            if (output/"results.csv").exists() or any((output/"weights").glob("*.pt")):raise RuntimeError("ZERO_STEP_PROBE_UNEXPECTED_TRAINING_RESULT")
            print("B2_ZERO_OPTIMIZER_STEP_EXACT_LOADER_WITNESS_PASS="+str(seed),flush=True)
        else:raise RuntimeError("PROBE_DID_NOT_STOP_BEFORE_OPTIMIZATION")
        finally:
            ld=kept.get("loader")
            it=getattr(ld,"iterator",None)
            if it is not None and hasattr(it,"_shutdown_workers"):it._shutdown_workers()
        return
    sys.argv=[str(c.WORKER),"--run-config",str(Path(args.run_config).resolve())]
    worker.main()


if __name__=="__main__":main()
