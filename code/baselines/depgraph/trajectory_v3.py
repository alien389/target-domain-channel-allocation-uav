"""Exactly one approved scope-complete, joint-resource structural trajectory."""
import copy
import csv
import json
import math
import os
import stat
import time
import traceback
from datetime import datetime,timezone
import scope_adapter as a
import torch

PMIN,PMAX,FMIN,FMAX=2440209,2515679,7618383104,7854003200


def commit(writer,stream,row):
    a.disk_guard();writer.writerow(row);stream.flush();os.fsync(stream.fileno())


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    lockpath=a.OUT/"01_B1_V3_AUTHORITY_AND_PROTOCOL_LOCK.json"
    lock=json.loads(lockpath.read_text(encoding="utf-8"))
    if lockpath.stat().st_mode&stat.S_IWRITE:raise RuntimeError("PROTOCOL_LOCK_NOT_READ_ONLY")
    for path,expected in lock["frozen_code_sha256"].items():
        if a.sha(path)!=expected:raise RuntimeError("FROZEN_CODE_HASH_CHANGED")
    if lock["implementation_gate"]!="PASS":raise RuntimeError("IMPLEMENTATION_GATE_NOT_PASS")
    parent,genes=a.load_source()
    p,f=a.counter(parent)
    if (p,f)!=(3985767,10825984000):raise RuntimeError("PARENT_COUNTER_MISMATCH")
    anchors=parent.model[-1].anchors.detach().clone();stride=parent.stride.detach().clone();names=copy.deepcopy(parent.names)
    model=parent
    lineages={g["group_id"]:list(range(g["original_width"])) for g in genes}
    trials_fields=["step","group_id","current_original_ids","native_saliencies_FP32","selected_current_indices","selected_original_ids",
                   "raw_l1_eight","I_g","params_before","flops_before","params_trial","flops_trial","delta_P","delta_F","R_P","R_F","J_g","S_g",
                   "decision","reason","complete_composite_closure_json","elapsed_seconds"]
    trace_fields=["step","selected_group","selected_original_ids","selected_current_indices","I_g","J_g","S_g","delta_P","delta_F",
                  "params_before","flops_before","params_after","flops_after","admissible_actions","rejected_actions",
                  "lineage_json","complete_composite_closure_json","utc"]
    a.disk_guard()
    trialstream=(a.OUT/"05_B1_V3_ALL_TRIALS_AND_REJECTIONS.csv").open("x",newline="",encoding="utf-8")
    tracestream=(a.OUT/"06_B1_V3_ACCEPTED_STEP_TRACE.csv").open("x",newline="",encoding="utf-8")
    tw=csv.DictWriter(trialstream,fieldnames=trials_fields);sw=csv.DictWriter(tracestream,fieldnames=trace_fields)
    tw.writeheader();sw.writeheader();trialstream.flush();tracestream.flush()
    start=time.monotonic();accepted=0;trial_count=0;rejections=0;error=None;candidate=None
    status="B1_V3_INFEASIBLE_UNDER_FROZEN_RULE";reason="NO_ADMISSIBLE_ACTION"
    try:
        for step in range(1,513):
            if PMIN<=p<=PMAX and FMIN<=f<=FMAX:
                status="CANDIDATE_PENDING_INDEPENDENT_VERIFICATION";reason="FIRST_JOINT_B6_INTERSECTION";break
            rp,rf=max(p-PMAX,0),max(f-FMAX,0)
            if rp==rf==0:raise RuntimeError("WORKING_STATE_BELOW_LOWER_BOUND")
            dg=a.graph(model)
            best=None;admissible=0;step_rejected=0
            for gene in genes:
                gid=gene["group_id"];width=len(lineages[gid])
                if width-8 not in {int(x) for x in gene["allowed_widths"].split(";")}:continue
                trial_start=time.monotonic()
                group,_=a.composite_group(model,dg,gene,list(range(width)))
                saliencies=a.native_importance(group)
                if len(saliencies)!=width:raise RuntimeError("LOGICAL_SALIENCY_LENGTH_MISMATCH")
                indices=sorted(range(width),key=lambda i:(float(saliencies[i]),lineages[gid][i]))[:8]
                originals=tuple(sorted(lineages[gid][i] for i in indices))
                eight=[float(saliencies[i]) for i in indices]
                importance=math.fsum(eight)/8.0
                row={"step":step,"group_id":gid,"current_original_ids":json.dumps(lineages[gid]),
                     "native_saliencies_FP32":json.dumps([float(x) for x in saliencies]),
                     "selected_current_indices":json.dumps(indices),"selected_original_ids":json.dumps(originals),
                     "raw_l1_eight":json.dumps(eight),"I_g":repr(importance),"params_before":p,"flops_before":f,"R_P":rp,"R_F":rf}
                trial_count+=1
                try:
                    clone=copy.deepcopy(model)
                    proposed,ledger=a.prune_action(clone,genes,gene,indices,lineages)
                    a.scope_check(clone,genes,proposed,anchors,stride,names)
                    a.forward_check(clone,128)
                    pp,ff=a.counter(clone)
                    dp,df=p-pp,f-ff
                    row.update({"params_trial":pp,"flops_trial":ff,"delta_P":dp,"delta_F":df,
                                "complete_composite_closure_json":json.dumps(ledger,separators=(",",":"))})
                    invalid=None
                    if dp<=0 or df<=0:invalid="NONPOSITIVE_RESOURCE_SAVINGS"
                    elif pp<PMIN or ff<FMIN:invalid="B6_LOWER_BOUND_CROSSING"
                    if invalid:
                        row.update({"decision":"REJECTED","reason":invalid});step_rejected+=1;rejections+=1
                    else:
                        fractions=[]
                        if rp>0:fractions.append(dp/rp)
                        if rf>0:fractions.append(df/rf)
                        progress=min(fractions);score=importance/progress
                        if not math.isfinite(progress) or progress<=0 or not math.isfinite(score) or score<0:raise RuntimeError("INVALID_JOINT_PROGRESS_OR_SCORE")
                        row.update({"J_g":repr(progress),"S_g":repr(score),"decision":"ADMISSIBLE","reason":""})
                        key=(score,-progress,gid,originals)
                        admissible+=1
                        if best is None or key<best[0]:best=(key,clone,proposed,dict(row))
                except Exception as exc:
                    row.update({"decision":"TECHNICAL_FAILURE","reason":repr(exc),"elapsed_seconds":time.monotonic()-trial_start})
                    commit(tw,trialstream,row)
                    raise
                row["elapsed_seconds"]=time.monotonic()-trial_start
                commit(tw,trialstream,row)
            if best is None:
                reason=f"NO_ADMISSIBLE_ACTION_AT_STEP_{step}";break
            _,model,lineages,chosen=best
            previous_p,previous_f=p,f
            p,f=int(chosen["params_trial"]),int(chosen["flops_trial"])
            accepted+=1
            commit(sw,tracestream,{"step":step,"selected_group":chosen["group_id"],
                "selected_original_ids":chosen["selected_original_ids"],"selected_current_indices":chosen["selected_current_indices"],
                "I_g":chosen["I_g"],"J_g":chosen["J_g"],"S_g":chosen["S_g"],"delta_P":chosen["delta_P"],"delta_F":chosen["delta_F"],
                "params_before":previous_p,"flops_before":previous_f,"params_after":p,"flops_after":f,
                "admissible_actions":admissible,"rejected_actions":step_rejected,
                "lineage_json":json.dumps(lineages,separators=(",",":")),
                "complete_composite_closure_json":chosen["complete_composite_closure_json"],"utc":datetime.now(timezone.utc).isoformat()})
            print(f"step={step} group={chosen['group_id']} P={p} F={f} admissible={admissible} rejected={step_rejected}",flush=True)
            if PMIN<=p<=PMAX and FMIN<=f<=FMAX:
                status="CANDIDATE_PENDING_INDEPENDENT_VERIFICATION";reason="FIRST_JOINT_B6_INTERSECTION";break
        else:reason="512_ACCEPTED_STEP_LIMIT"
    except Exception as exc:
        status="B1_V3_IMPLEMENTATION_BLOCKED";reason=repr(exc);error=traceback.format_exc()
    finally:
        for stream in (trialstream,tracestream):stream.flush();os.fsync(stream.fileno());stream.close()
    result={"status":status,"reason":reason,"error":error,"accepted_steps":accepted,"trial_actions":trial_count,
            "rejected_actions":rejections,"params_fused":p,"raw_flops":f,"lineages":lineages,
            "elapsed_seconds":time.monotonic()-start,"protocol_sha256":a.sha(lockpath),"source_sha256":a.SOURCE_SHA,
            "training_steps":0,"accuracy_images":0,"metaheuristic_searches":0}
    if status=="CANDIDATE_PENDING_INDEPENDENT_VERIFICATION":
        a.disk_guard()
        path=a.OUT/"b1_v3_scope_complete_joint_resource_raw.pt"
        torch.save({"model":model.cpu().float().eval(),"source_sha256":a.SOURCE_SHA,"protocol_sha256":a.sha(lockpath),
                    "lineages":lineages,"model_identity":"B1-V3-R1-scope-complete-DepGraph-group-L1-joint-resource-nc10-raw"},path)
        result["candidate_path"]=str(path);result["candidate_sha256"]=a.sha(path)
    a.disk_guard()
    (a.OUT/"trajectory_v3_result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="lineages"},indent=2),flush=True)


if __name__=="__main__":main()
