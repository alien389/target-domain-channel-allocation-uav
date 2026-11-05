"""Freeze one unambiguous source-only BN importance definition before resource trajectory."""
from bn_common import *


def main():
    target=OUT/'04_BN_GAMMA_PREDECLARED_METHOD_LOCK.json'
    assert not target.exists(), 'NO_LOCK_OVERWRITE'
    preparation=json.loads((OUT/'environment_and_counter_prechecks.json').read_text(encoding='utf-8'))
    assert preparation['status']=='PASS'
    files=[OUT/n for n in ['bn_common.py','prepare_bn.py','trajectory_bn.py','independent_bn.py','freeze_bn.py']]
    codes={str(p):a.sha(p) for p in files}
    method={'created_utc':now(),'phase':'ONE_STRUCTURAL_TRAJECTORY_ONLY','author_authorized_training':False,
        'method_identity':'Global BN-gamma magnitude pruning of the existing pretrained SearchParent, with R1 dependency-safe resource scheduling',
        'not_reproduction_of':'Full three-stage Network Slimming / Liu et al. 2017',
        'sparse_training':False,'seed':42,'device':'CPU_FP32','source_checkpoint':str(a.SOURCE),'source_sha256':a.SOURCE_SHA,
        'R1_manifest':str(a.GENES),'R1_sha256':a.GENES_SHA,'groups':35,'variable':25,'fixed':10,
        'B6_inclusive':{'Pmin':PMIN,'Pmax':PMAX,'Fmin':FMIN,'Fmax':FMAX},
        'importance':{'API':'torch_pruning.importance.BNScaleImportance',
            'distribution':preparation['environment']['torch_pruning_distribution'],
            'module':preparation['environment']['torch_pruning_module'],
            'source':str(IMPORTANCE),'source_sha256':a.sha(IMPORTANCE),
            'source_lines':'GroupMagnitudeImportance.__call__:174-269; _reduce:139-172; _normalize:115-137; BNScaleImportance:272-295',
            'p':1,'group_reduction':'mean','normalizer':'mean','bias':False,'target_types':['real affine BatchNorm'],
            'components':'For each complete composite logical group, real BN prune_batchnorm_out_channels entries only; abs(weight[index]); scatter to native root_idxs; FP32 sum divided by count of recognized BN entries; divide by its full original root-channel mean.',
            'score_freeze':'Evaluate the installed API ONCE on the immutable full source group. Freeze normalized source s_g(original_id). Later only slice surviving original IDs. No current-survivor renormalization, gamma updates or score-source changes.',
            'why_this_freeze':'Explicit prompt requires source-frozen BN scores except surviving indices. Native API behavior is reproduced at source, including mean normalization. Recalling API on a shortened group would introduce a different normalization denominator; that is NOT this precommitted baseline.',
            'group_coverage':'35/35 mapped; all 25 variable groups have more than one distinct finite score',
            'independent_probe':'FP64 scalar sum of actual mapped absolute gamma; tolerance rtol2e-6/atol2e-7 vs native FP32'},
        'source_score_file':'source_bn_scores_and_component_mappings.json',
        'frozen_BN_scores_sha256':a.sha(OUT/'source_bn_scores_and_component_mappings.json'),
        'per_action':'Remove exactly eight lowest frozen scores among surviving original root IDs; equal scores prefer ascending original ID. Next width must be R1 legal.',
        'full_physical_scope':'Reuse unchanged B1-V3 scope_adapter.py composite_group/prune_action; explicit original logical ties supplement DepGraph. Never root-only V1/V2.',
        'trial':'Fresh current model clone for every legal action; scope_check + finite synthetic128 + exact fused Stage10 physical P/F. One invalid action logged/rejected; other actions continue.',
        'reject':['illegal R1 width','missing BN','broken protected nc10/head45/anchors/strides/classes','nonfinite output or score',
                  'P_trial<Pmin','F_trial<Fmin','dP<=0','dF<=0','technical action failure'],
        'scheduler':{'R_P':'max(P-Pmax,0)','R_F':'max(F-Fmax,0)','J_g':'min(dP/R_P if R_P>0, dF/R_F if R_F>0)',
                     'I_g':'math.fsum(eight fixed source BN scores)/8','S_g':'I_g/J_g',
                     'order':['lower S_g','larger J_g','smaller canonical group_id','lexically smaller sorted original-ID tuple'],
                     'numeric':'native saliency FP32; score/tie math Python binary64, math.fsum, exact comparisons; no epsilon'},
        'stop':['first simultaneous inclusive B6 membership','zero admissible actions','512 accepted actions','genuine technical/source failure'],
        'counter':{'source':str(STAGE10),'source_sha256':a.sha(STAGE10),'function_lines':'structural_preflight:99-159',
                   'expression':'deepcopy.cpu.float.eval.fuse; real tensor Params; THOP at stride32; int(round(2.0*float(MACs)*(640/stride)**2))',
                   'independent_final':'separate source reconstruction through original structured_pruner + separate counter + direct640 THOP'},
        'frozen_execution_engine_sha256':a.sha(Path(a.__file__)), 'executable_code_sha256':codes,
        'preparation':preparation,'no_accuracy_or_solution_oracle':True,
        'forbidden_inputs':['Meta/Manual/Random solved allocations and channel indices','Taylor saliency','AP outcomes','real dataset tensors'],
        'zero_authorized':['epochs','optimizer steps','backward','real dataset inference','AP','TensorRT','GPU benchmarks']}
    dump(target.name,method)
    write('05_BN_GAMMA_SCORE_MAPPING_AND_INDEPENDENT_UNIT_TESTS.md',
        '# Real BN score mapping and independent source tests\n\n'
        '35/35 composite R1 groups have real affine BN contributions, including all 25 variable groups. '
        'Every variable group has finite, distinguishable source scores. Exact contributing modules, gamma and original root-index mappings are in file03 and source_bn_scores_and_component_mappings.json.\n\n'
        'The installed BNScaleImportance uses abs(gamma), p=1, no bias, real BN target types only, FP32 scatter-add mean reduction then division by the group mean. '
        'An independently written FP64 scalar reconstruction agrees for every group (rtol=2e-6, atol=2e-7). '
        'Source native-normalized scores are frozen once. Only original surviving IDs are sliced later; current-survivor renormalization is deliberately absent to honor source-frozen scores. '
        'This distinction is precommitted before any candidate resource trial.\n\n'
        'No Conv weights, Taylor scores, bias magnitudes, synthetic gamma, sparsity regularization, or real images enter saliency. '
        'No full Network Slimming reproduction is claimed. Composite topology is reused unchanged from the audited B1-V3 engine; the final candidate, if present, requires independent original-pruner slice reconstruction.\n')
    for p in files+[target,OUT/'source_bn_scores_and_component_mappings.json']:
        p.chmod(0o444)
    dump('pretrajectory_freeze_receipt.json',{'created_utc':now(),'protocol_sha256':a.sha(target),
                                            'code_sha256':codes,'state':'FROZEN_BEFORE_FIRST_RESOURCE_TRIAL'})
    print('METHOD_FROZEN '+a.sha(target),flush=True)


if __name__=='__main__':main()
