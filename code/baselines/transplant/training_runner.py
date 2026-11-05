"""Passive recording wrapper around the exact original Stage10 worker."""
import argparse,copy,importlib.metadata as md,json,os,platform,subprocess,sys
from pathlib import Path
import v6_runtime as c
sys.path[:0]=[str(c.YOLO),str(c.P/"scripts")]
os.chdir(c.YOLO)
import torch,numpy as np,cv2

def flags():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
      float32_matmul_precision=torch.get_float32_matmul_precision(),deterministic=torch.are_deterministic_algorithms_enabled(),
      cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark)
def environment():
    return dict(python=sys.version,python_executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cv2=cv2.__version__,
      cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),GPU=torch.cuda.get_device_name(0),OS=platform.platform(),
      package_versions={d.metadata["Name"].lower():d.version for d in md.distributions() if d.metadata["Name"]},flags=flags(),
      environment={k:os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE","NVIDIA_TF32_OVERRIDE")})
def main():
    p=argparse.ArgumentParser();p.add_argument("--run-config",required=True);p.add_argument("--probe",action="store_true");args=p.parse_args()
    cfg=c.read(args.run_config)
    out=c.O/"EXACT_RUNTIME_ENVIRONMENT_PROBE" if args.probe else Path(cfg["output_directory"])
    if args.probe:out.mkdir(exist_ok=False)
    else:
        matrix=c.verify_frozen()
        assert c.read(c.PRE/"INDEPENDENT_PRELAUNCH_GATE.json")["status"]=="PASS"
    worker=c.module("authentic_stage10_worker_b3v6",c.S10/"scripts/stage10_run_training.py")
    original=worker.load_training_module
    def observed_loader(source,initialization):
        before=environment();module=original(source,initialization);after=environment()
        record=dict(before=before,after=after,label=c.LABEL)
        fingerprint=c.digest(record)
        c.write(out/"current_environment_before_after_imports.json",dict(record=record,fingerprint=fingerprint))
        if not args.probe:
            golden=c.read(c.O/"EXACT_RUNTIME_ENVIRONMENT_PROBE/current_environment_before_after_imports.json")
            assert fingerprint==golden["fingerprint"],"CURRENT_ENVIRONMENT_CHANGED"
        factory=module.Model
        def observed_factory(*a,**kw):
            model=factory(*a,**kw)
            initial=torch.load(initialization,map_location="cpu",weights_only=False)["model"].float().state_dict()
            state=model.state_dict()
            assert len(state)==len(initial)==319 and all(torch.equal(state[k],initial[k]) for k in state)
            c.write(out/"preupdate_initializer_witness.json",dict(initializer_sha256=c.sha(initialization),all319_checked=True,
                optimizer_steps=0,model_factory="ORIGINAL_STRUCTURAL_FACTORY_UNCHANGED",no_random_reinitialization=True))
            return model
        module.Model=observed_factory
        old_parse=module.parse_opt
        def observed_parse(*a,**kw):
            opt=old_parse(*a,**kw);actual=json.loads(json.dumps(vars(opt),default=str))
            expected=c.read(c.PRE/"PROPOSED_RUN_AND_EVAL_MATRIX.json")["jobs"][[42,43,44].index(cfg["seed"])]["effective_args"]
            assert actual==expected,("EFFECTIVE_CONFIG_MISMATCH",actual,expected)
            c.write(out/"executed_effective_args.json",actual);return opt
        module.parse_opt=observed_parse
        amp=module.check_amp
        def observed_amp(model):
            value=amp(model);c.write(out/"actual_original_AMP_decision.json",dict(enabled=bool(value),source="ORIGINAL_CHECK_AMP"))
            assert value,"ORIGINAL_AMP_RETURNED_FALSE_BEFORE_OPTIMIZER"
            return value
        module.check_amp=observed_amp
        seed_fn=module.init_seeds
        def observed_seed(*a,**kw):
            value=seed_fn(*a,**kw);c.write(out/"runtime_seed_configuration.json",dict(args=a,kwargs=kw,flags=flags(),
                   CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG")));return value
        module.init_seeds=observed_seed
        old_loader=module.create_dataloader
        def observed_data(*a,**kw):
            result=old_loader(*a,**kw)
            assert len(result[1])==6471 and len(result[0])==405
            c.write(out/"actual_train_loader_contract.json",dict(images=6471,batches=405,batch=16,workers=kw.get("workers"),
               seed=kw.get("seed"),shuffle=kw.get("shuffle"),rect=kw.get("rect"),validation_loader=False))
            return result
        module.create_dataloader=observed_data
        # Passive source-contract witnesses never change returned objects or training math.
        intersect=module.intersect_dicts
        def observed_intersect(source,destination,exclude=()):
            result=intersect(source,destination,exclude=exclude)
            assert not exclude and set(result)==set(source)==set(destination) and len(result)==319
            c.write(out/"actual_319_state_transfer.json",dict(transferred=319,required=319,missing=[],unexpected=[],exclude=list(exclude)))
            return result
        module.intersect_dicts=observed_intersect
        anchor=module.check_anchors
        def observed_anchor(dataset,model,*a,**kw):
            assert len(dataset)==6471
            before=model.model[-1].anchors.detach().cpu().tolist()
            result=anchor(dataset,model,*a,**kw)
            c.write(out/"actual_train_only_autoanchor.json",dict(before=before,after=model.model[-1].anchors.detach().cpu().tolist(),
                stride=model.model[-1].stride.detach().cpu().tolist(),train_images=6471,seed=cfg["seed"],original_algorithm=True))
            return result
        module.check_anchors=observed_anchor
        optimizer_factory=module.smart_optimizer
        observations=dict(actual_optimizer_steps=0,completed_saved_epochs=0)
        def observed_optimizer(*a,**kw):
            optimizer=optimizer_factory(*a,**kw)
            assert type(optimizer).__name__=="SGD" and optimizer.defaults["nesterov"]
            def actual_step(*unused):
                observations["actual_optimizer_steps"]+=1
            optimizer.register_step_post_hook(actual_step)
            c.write(out/"actual_optimizer_param_groups.json",dict(type=type(optimizer).__name__,defaults=optimizer.defaults,
                groups=[dict(parameter_tensors=len(g["params"]),weight_decay=g["weight_decay"],initial_lr=g["lr"]) for g in optimizer.param_groups]))
            return optimizer
        module.smart_optimizer=observed_optimizer
        callbacks=module.main.__defaults__[0]
        def on_save(last,epoch,final_epoch,best_fitness,fi):
            observations["completed_saved_epochs"]=int(epoch)+1
            c.write(out/"passive_optimizer_and_epoch_witness.json",dict(**observations,epoch=int(epoch),time=c.now(),
                final_epoch=bool(final_epoch),optimizer_count_definition="Actual SGD step post-hook, excludes GradScaler-skipped attempts",validation_called=False))
        callbacks.register_action("on_model_save","b3v6_passive_saved_epoch",on_save)
        # Read-only metadata adapters only. Training math, options and libraries
        # are untouched. All installed requirements were independently checked.
        requirement_check=module.check_requirements
        def read_only_requirements(*a,**kw):
            kw["install"]=False
            result=requirement_check(*a,**kw)
            c.write(out/"read_only_dependency_check.json",dict(installation_permitted=False,result=result,
                    independent_installed_requirement_check="ALL_19_SATISFIED",source_unmodified=True))
            return result
        module.check_requirements=read_only_requirements
        def read_only_git(*a,**kw):
            commands=[["git","--no-optional-locks","rev-parse","HEAD"],["git","--no-optional-locks","status","--porcelain","-uno"]]
            records=[]
            for command in commands:
                r=subprocess.run(command,cwd=c.YOLO,capture_output=True,text=True,check=False)
                records.append(dict(command=command,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
            c.write(out/"read_only_git_metadata.json",dict(records=records,fetch_performed=False,
                    reason="Original advisory check_git_status would git fetch into immutable source .git."))
        module.check_git_status=read_only_git
        # This is inert for present local assets. It prevents an unrecorded download.
        import utils.downloads as downloads
        old_download=downloads.attempt_download
        def local_only(file,*a,**kw):
            assert Path(file).is_file(),("UNAPPROVED_REMOTE_ASSET_REQUIRED",str(file))
            return old_download(file,*a,**kw)
        module.attempt_download=local_only
        return module
    worker.load_training_module=observed_loader
    if args.probe:
        worker.structural_preflight(cfg,Path(cfg["initialization"]),out)
        observed_loader(Path(cfg["training_source"]),Path(cfg["initialization"]))
        return
    sys.argv=[str(worker.__file__),"--run-config",str(Path(args.run_config).resolve())]
    worker.main()
if __name__=="__main__":main()
