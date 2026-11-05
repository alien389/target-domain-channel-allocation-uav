"""Exactly one source-frozen BN-gamma trajectory, no training or dataset access."""
from bn_common import *
import time
import traceback

FIELDS = ['step','group_id','status','reason','attempted','params_before','flops_before','params_trial','flops_trial',
          'delta_P','delta_F','R_P','R_F','I_g','J_g','S_g','current_original_ids','selected_current_indices',
          'selected_original_ids','eight_frozen_BN_scores','closure','finite128_shapes']
ACCEPT = ['step','group_id','params_before','flops_before','params_after','flops_after','delta_P','delta_F',
          'I_g','J_g','S_g','selected_current_indices','selected_original_ids','lineages','architecture']


def encode(row):
    return {k: json.dumps(v, separators=(',', ':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()}


def main():
    started, wall, cpu = now(), time.perf_counter(), time.process_time()
    lockpath = OUT/'04_BN_GAMMA_PREDECLARED_METHOD_LOCK.json'
    lock = json.loads(lockpath.read_text(encoding='utf-8'))
    for path, expected in lock['executable_code_sha256'].items():
        assert a.sha(path) == expected, ('PRETRAJECTORY_CODE_HASH', path)
    assert a.sha(OUT/'source_bn_scores_and_component_mappings.json') == lock['frozen_BN_scores_sha256']
    assert not (OUT/'trajectory_result.json').exists(), 'NO_RERUN_OF_FROZEN_TRAJECTORY'
    model, genes = a.load_source()
    lineage = initial_lineages(genes)
    anchors, stride, names = refs(model)
    frozen = json.loads((OUT/'source_bn_scores_and_component_mappings.json').read_text(encoding='utf-8'))['scores']
    p, f = a.counter(model)
    accepted, attempted = 0, 0
    reason, candidate, error = '', None, None
    dump('trajectory_start_receipt.json', {'started_utc':started, 'protocol_sha256':a.sha(lockpath),
                                         'parent_params':p,'parent_flops':f, 'seed':42})
    with (OUT/'06_BN_GAMMA_ALL_TRIAL_ACTIONS.csv').open('x',encoding='utf-8-sig',newline='') as ts, \
         (OUT/'07_BN_GAMMA_ACCEPTED_TRAJECTORY.csv').open('x',encoding='utf-8-sig',newline='') as ac:
        tw, aw = csv.DictWriter(ts,fieldnames=FIELDS), csv.DictWriter(ac,fieldnames=ACCEPT)
        tw.writeheader(); aw.writeheader(); ts.flush(); ac.flush()
        try:
            for step in range(1,514):
                if legal(p,f):
                    reason = 'FIRST_SIMULTANEOUS_B6_INTERSECTION'
                    break
                if accepted >= 512:
                    reason = 'MAXIMUM_512_ACCEPTED_ACTIONS'
                    break
                a.disk_guard(200_000_000)
                rows, best = [], None
                rp, rf = max(p-PMAX,0), max(f-FMAX,0)
                for g in genes:
                    gid = g['group_id']
                    if g['allowed_width_count'] <= 1:
                        continue
                    row = {'step':step,'group_id':gid,'params_before':p,'flops_before':f,
                           'R_P':rp,'R_F':rf,'current_original_ids':lineage[gid], 'attempted':False}
                    allowed = {int(x) for x in g['allowed_widths'].split(';')}
                    if len(lineage[gid])-8 not in allowed:
                        row.update(status='INELIGIBLE',reason='NEXT_DIV8_WIDTH_NOT_R1_LEGAL')
                        rows.append(row); continue
                    ids = sorted(range(len(lineage[gid])),key=lambda i:(frozen[gid][lineage[gid][i]],lineage[gid][i]))[:8]
                    originals = sorted(lineage[gid][i] for i in ids)
                    values = [frozen[gid][lineage[gid][i]] for i in ids]
                    imp = math.fsum(values)/8
                    row.update(selected_current_indices=ids,selected_original_ids=originals,
                               eight_frozen_BN_scores=values,I_g=imp,attempted=True)
                    attempted += 1
                    try:
                        trial = copy.deepcopy(model)
                        updated, closure = a.prune_action(trial,genes,g,ids,lineage)
                        row['closure'] = closure
                        a.scope_check(trial,genes,updated,anchors,stride,names)
                        row['finite128_shapes'] = a.forward_check(trial,128)
                        pp, ff = a.counter(trial)
                        dp, df = p-pp, f-ff
                        row.update(params_trial=pp,flops_trial=ff,delta_P=dp,delta_F=df)
                        if dp <= 0 or df <= 0:
                            row.update(status='REJECTED',reason='NONPOSITIVE_RESOURCE_SAVING')
                        elif pp < PMIN or ff < FMIN:
                            row.update(status='REJECTED',reason='B6_LOWER_BOUND_CROSSING')
                        else:
                            progress = min(([dp/rp] if rp else []) + ([df/rf] if rf else []))
                            score = imp/progress
                            if not math.isfinite(progress) or progress <= 0 or not math.isfinite(score):
                                raise RuntimeError('NONFINITE_BN_OR_JOINT_SCORE')
                            row.update(J_g=progress,S_g=score,status='ADMISSIBLE',reason='')
                            key = (score,-progress,gid,tuple(originals))
                            if best is None or key < best[0]:
                                best = (key,trial,updated,row)
                    except Exception as exc:
                        row.update(status='REJECTED',reason='TECHNICAL_ACTION_FAILURE:'+repr(exc))
                    rows.append(row)
                for row in rows:
                    if best is not None and row is best[3]: row['status']='SELECTED'
                    tw.writerow(encode(row))
                ts.flush()
                if best is None:
                    reason = 'NO_ADMISSIBLE_ACTION_REMAINS'
                    break
                _, model, lineage, chosen = best
                oldp, oldf = p,f
                p,f = chosen['params_trial'],chosen['flops_trial']
                accepted += 1
                aw.writerow(encode({'step':accepted,'group_id':chosen['group_id'],'params_before':oldp,
                    'flops_before':oldf,'params_after':p,'flops_after':f,'delta_P':oldp-p,'delta_F':oldf-f,
                    'I_g':chosen['I_g'],'J_g':chosen['J_g'],'S_g':chosen['S_g'],
                    'selected_current_indices':chosen['selected_current_indices'],
                    'selected_original_ids':chosen['selected_original_ids'],'lineages':lineage,
                    'architecture':architecture(model,genes,lineage)}))
                ac.flush()
                dump('trajectory_live_status.json',{'state':'STRUCTURAL_ONLY','accepted_steps':accepted,
                    'trial_actions':attempted,'params_fused':p,'raw_flops':f,'last_update_utc':now()})
                if accepted % 10 == 0 or legal(p,f):
                    print(f'STEP {accepted}: P={p}, F={f}, trial_actions={attempted}',flush=True)
            if legal(p,f):
                candidate = str(OUT/'bn_gamma_b6_untrained_raw.pt')
                guard()
                torch.save({'model':model.cpu().float().eval(),'ema':None,'epoch':-1,
                            'model_identity':'BN-Gamma-B6 / source-frozen BN scores + full-R1 joint-resource scheduler',
                            'source_sha256':a.SOURCE_SHA,'lineages':lineage,'protocol_sha256':a.sha(lockpath),
                            'untrained':True,'optimizer':None},candidate)
        except Exception:
            error = traceback.format_exc()
            reason = 'TRAJECTORY_TECHNICAL_BLOCKER'
        finally:
            result = {'reason':reason,'error':error,'accepted_steps':accepted,'trial_actions':attempted,
                'params_fused':p,'raw_flops':f,'margins':margins(p,f),'B6_joint':legal(p,f),
                'candidate_path':candidate,'candidate_sha256':a.sha(candidate) if candidate else None,
                'lineages':lineage,'identity':architecture(model,genes,lineage),
                'protocol_sha256':a.sha(lockpath),'started_utc':started,'finished_utc':now(),
                'wall_seconds':time.perf_counter()-wall,'process_CPU_seconds':time.process_time()-cpu,
                'training_epochs':0,'optimizer_steps':0,'backward_calls':0,'real_images':0,'AP_evaluations':0}
            dump('trajectory_result.json',result)
            print(json.dumps({k:v for k,v in result.items() if k not in ['lineages','identity']},indent=2),flush=True)


if __name__ == '__main__':
    main()
