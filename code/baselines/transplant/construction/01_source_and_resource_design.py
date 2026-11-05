"""Frozen joint 13/13 experiment definitions and deterministic exact resource design."""
import ast,itertools,json,stat,time
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import common as c

def main():
    started=c.now()
    assert not (c.O/"SEARCH_SUMMARY.json").exists()
    inventory={}
    def protect(p,expected=None,role="SOURCE_VERIFIED"):
        p=c.Path(p).resolve();h=c.sha(p)
        if expected:assert h==expected,(p,h,expected)
        inventory[str(p)]=dict(path=str(p),sha256=h,expected_sha256=expected,size_bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,role=role)
    protect(c.AUTHOR,role="AUTHOR_DECISION")
    # Historical original manifest supplies parent, ranking, graph and R1 digests.
    historical=c.P/"09_locked_architectures/stage09_source_hashes.json";protect(historical)
    for r in c.read(historical)["authoritative_files"]:
        if any(x in r["path"] for x in ("gene_space_manifest_R1.json","taylor_ranking_manifest.json","dependency_graph.json","lite_v3_search_parent")):
            protect(r["path"],r["sha256"])
    protect(c.R1,"318C5E12F10BE0AF6533BC1E8BC4889CA2277B3325C988135122A4A9957A3B42")
    protect(c.PRUNER,"6AFDCF48BC707E88CA76A90C627944864D75946C71D02E48D7A1C50F68DD249E")
    protect(c.V2/"01_PRIMARY_SOURCE_AND_HASH_INVENTORY.csv",role="PREVIOUS_INDEX")
    protect(c.V2/"independent_verify_resources.py",role="DISTINCT_ARITHMETIC_CHECKER")
    index=c.rows(c.V2/"01_PRIMARY_SOURCE_AND_HASH_INVENTORY.csv")
    for name in ("R1_COUNTER","HISTORICAL_YOLO_COMMON","HISTORICAL_YOLO_MODEL"):
        r=next(r for r in index if r["source_id"]==name);assert c.sha(r["source_path"])==r["actual_sha256"]
    locked=c.P/"09_locked_architectures/locked_models_manifest.csv";protect(locked)
    ref_records={}
    for r in c.rows(locked):
        if r["model"] not in ("Meta-AP+APS-B6","MatchedManual-B6"):continue
        n="Meta" if r["model"]=="Meta-AP+APS-B6" else "Manual";ref_records[n]=r
        for key,hkey in (("genome","genome_sha256"),("channel_index_map","channel_index_map_sha256"),("canonical_yaml","canonical_yaml_sha256"),
                         ("canonical_raw_checkpoint","canonical_raw_checkpoint_sha256"),("source_reconstruction_manifest","source_reconstruction_manifest_sha256")):
            protect(r[key],r[hkey])
    for p in [c.P/"09_locked_architectures/scripts/stage09_final_architecture_lock.py",c.P/"scripts/stage04_shared_p0_fidelity1.py",
              c.P/"03a_taylor_dependency_gene_space/structural_profile/litev3_searchparent_nc10_stage03a_unfused/channel_topology.json",
              c.P/"03a_taylor_dependency_gene_space/structural_profile/litev3_searchparent_nc10_stage03a_unfused/pruning_constraints.json",
              c.P/"10_final_fulltrain_300e_matched/stage10_protocol_lock.json",
              c.V2/"04_RESOURCE_B6_AND_PAIRWISE_1PCT_CERTIFICATE.json",
              c.V3/"05_FOUR_CANDIDATE_COMPLETE_RESOURCE_CERTIFICATES.csv",
              c.V4/"ANALYTICAL_RESULTS.json",c.V4/"CURRENT_STATUS.json",c.V4/"AUTHOR_STOP_NO_AUTORESTART.json",
              c.V4/"AUTHOR_STOP_20261102_232332/STOP_VERIFICATION.json",
              c.B/"B0_READINESS_AND_PROTOCOL_LOCK/B0_1_GAP_CLOSURE/technical_probe.json",
              c.B/"B0_READINESS_AND_PROTOCOL_LOCK/B0_1_GAP_CLOSURE/B0_1_OUTPUT_SHA256_MANIFEST.json"]:
        protect(p)
    stop=c.read(c.V4/"AUTHOR_STOP_20261102_232332/STOP_VERIFICATION.json")
    protect(stop["checkpoint_path"],stop["checkpoint_sha256"],"CANCELED_RUN_HASH_ONLY_NEVER_LOADED")
    protect(stop["original_last_path"],stop["original_last_sha256"],"CANCELED_RUN_HASH_ONLY_NEVER_LOADED")
    assert c.read(c.V4/"CURRENT_STATUS.json")["state"]=="STOPPED_BY_AUTHOR"
    c.write(c.O/"aux/PROTECTED_INPUTS_BEFORE.json",list(inventory.values()))
    groups=c.read(c.GENES)["groups"];order=[g["group_id"] for g in groups]
    assert order==[f"DG-{i:03d}" for i in range(1,36)]
    refs={n:c.read(r["genome"])["retained_widths"] for n,r in ref_records.items()}
    for n,r in ref_records.items():assert c.signature(order,refs[n])==r["architecture_signature"]
    r1=c.module("immutable_r1_v5_design",c.R1)
    loaded=r1.load_source_model(c.PARENT,"V5_CPU_SOURCE_METADATA_ONLY")
    parent=loaded.model;head=parent.model[-1]
    assert head.nc==10 and head.na==3 and head.no==15 and head.f==[17,20,23]
    assert [x.out_channels for x in head.m]==[45,45,45]
    different=[g for g in order if refs["Meta"][g]!=refs["Manual"][g]]
    back=[g["group_id"] for g in groups if g["block_index"]<=9 and refs["Meta"][g["group_id"]]!=refs["Manual"][g["group_id"]]]
    expected_back=["DG-002","DG-006","DG-010","DG-011","DG-012","DG-013","DG-014","DG-015","DG-016","DG-017","DG-018","DG-019"]
    assert back==expected_back and len(different)==25
    assert set(different)==set(back+c.PATHS) and len(c.PATHS)==13
    budget=json.loads(c.GENES.read_text(encoding="utf-8-sig"),parse_float=Decimal)["budget"]
    assert c.BOUNDS==(budget["params_lower_inclusive"],budget["params_upper_inclusive"],
                      int(budget["flops_lower_g_inclusive"]*10**9),int(budget["flops_upper_g_inclusive"]*10**9))
    regions={}
    reach={}
    for g in groups:
        b=g["block_index"]
        regions[g["group_id"]]="BACKBONE_ORIGIN" if b<=9 else "P3_INGRESS_SHARED" if b<=17 else "P4_P5" if b<=20 else "LATE_P5"
        seen={b}
        for i,m in enumerate(parent.model):
            f=[m.f] if isinstance(m.f,int) else m.f
            if i and any((s if s>=0 else i+s) in seen for s in f):seen.add(i)
        reach[g["group_id"]]=[f"P{j+3}" for j,x in enumerate(head.f) if x in seen]
    assert all(reach[g]==["P3","P4","P5"] for g in c.REGIONS["P3_INGRESS_SHARED"])
    assert all(reach[g]==["P4","P5"] for g in c.REGIONS["P4_P5"])
    assert all(reach[g]==["P5"] for g in c.REGIONS["LATE_P5"])
    # Fixed BEFORE computing the new direct/reverse or any compensated resources.
    protocol=dict(time=c.now(),author_path=str(c.AUTHOR),author_sha256=c.sha(c.AUTHOR),path_groups=c.PATHS,path_regions=c.REGIONS,
       backbone_compensators=back,source_legal_group_order=order,directions=c.DIRECTIONS,bounds=c.BOUNDS,
       generator_sha256=c.sha(__file__),helpers_sha256=c.sha(c.O/"scripts/common.py"),random_seed=None,randomness="NONE",
       level1="Exhaust both binary donor/recipient choices in all12 eligible backbone groups. Enumerate by count0..12, lexicographic subsets, D then R at each count.",
       level1_logical_states=8192,level1_time_bound_seconds=900,
       level2_trigger="No non-replica strict13/13 Level1 model with cap+band pass and <=1% four-way quality.",
       level2_if_triggered=dict(method="Deterministic bounded beam by changed-compensator count1..12 over original R1 widths",
          beam_width=256,max_unique_oracle_calls=100000,time_bound_seconds=900,rank="same frozen scientific ranking",no_global_optimality_claim=True),
       lower_coverage_trigger="No reliable full13/13 reasonable-match model after documented levels1/2. Not run when full13/13 tier<=1% exists.",
       ranking=["13of13 exact donor coverage","source-valid+hard caps","original band preferred","lowest quality tier","fewest compensators",
                "lowest maximum exact relative difference","lowest sum exact relative differences",
                "sum(abs(width_edit)/SearchParent_original_width)","D over R on exact tie","lexicographic35 widths"],
       top_alternatives_per_direction=10,leading_for_controls_per_direction=1,
       donor_or_recipient_identity="Report replicas, never propose a preexisting donor/recipient as a newly constructed ablation",
       control_rule="Exact same compensation first. If mismatched, design one nonzero backbone-only Level1 alternative preserving recipient paths. Rank cap,band,quality tier vsboth refs+RPC, off-target non-equivalence, max and sum deviation, smallest edit count, width tuple.",
       stopping="Stop after exhaustive Level1 if a non-replica full13/13 band/tier1 solution exists. Level2 is conditional, not automatically added.",
       no_training=True,no_optimizer=True,no_backward=True,no_GPU=True,no_real_dataset=True,no_accuracy=True,no_v4_restart=True)
    c.write(c.O/"aux/FROZEN_V5_DESIGN_PROTOCOL.json",protocol)
    (c.O/"aux/FROZEN_V5_DESIGN_PROTOCOL.json").chmod(stat.S_IREAD)
    c.text(c.O/"01_AUTHOR_EDITOR_S04_SCOPE_CONTRACT_EN.md","# Author and editor scope\n\nThe authorized endpoint is joint13-of13 structural design and causal-control feasibility only. The joint set covers shared P3-ingress, P4/P5 and late P5 producers. Downstream reachability is not exclusive ownership. No training, optimizer, backward, real-image inference, AP, new teacher or v4 continuation is authorized. Caps, historical matching band and author quality tiers are separate. The proposed comparison must disclose backbone compensation and inherited channel identity.\n")
    bank,rank=r1.taylor_scope_bank(groups)
    joins=[];maprows=[];elig=[]
    for g in groups:
        gid=g["group_id"];indices={}
        for n in refs:
            removed=set(bank[gid]["indices_low_to_high"][:g["original_width"]-refs[n][gid]])
            kept=sorted(set(range(g["original_width"]))-removed)
            rr=next(r for r in c.rows(ref_records[n]["channel_index_map"]) if r["group_id"]==gid)
            assert kept==[int(x) for x in rr["retained_channel_indices"].split(";")]
            indices[n]=";".join(map(str,kept))
        joins.append(dict(group_id=gid,producer_block=g["block_name"],module_type=g["module_type"],scope=g["scope_name"],
            source_modules=g["source_modules"],consumer_modules=g["dependent_modules"],parent_width=g["original_width"],
            Meta_width=refs["Meta"][gid],Manual_width=refs["Manual"][gid],allowed_widths=g["allowed_widths"],region=regions[gid],
            downstream=";".join(reach[gid]),Meta_retained_original_ids=indices["Meta"],Manual_retained_original_ids=indices["Manual"],
            residual=g["residual_contract"],concat=g["concat_contract"],depthwise=g["depthwise_contract"],Detect=g["detect_contract"],
            source_locator=f"R1.groups[group_id={gid}]; frozen Taylor importance_bank.{g['block_name']}.{g['scope_name']}"))
        maprows.append(dict(group_id=gid,producer=g["block_name"],scope=g["scope_name"],primary_owner_region=regions[gid],
             consumer_modules=g["dependent_modules"],downstream_detect=";".join(reach[gid]),exclusive_P3=reach[gid]==["P3"],
             full_joint_member=gid in c.PATHS,backbone_compensator=gid in back,shared=len(reach[gid])>1))
        elig.append(dict(group_id=gid,producer=g["block_name"],region=regions[gid],eligible_backbone=gid in back,
             reason="Original variable backbone group with distinct donor/recipient widths" if gid in back else "Frozen path gene" if gid in c.PATHS else "Equal or singleton width, not an effective Level1 compensator",
             level1_Meta=refs["Meta"][gid],level1_Manual=refs["Manual"][gid],level2_allowed=g["allowed_widths"],div8=True))
    c.csvwrite(c.O/"03_CANONICAL_35_GENE_META_MANUAL_SOURCE_JOIN.csv",joins)
    c.csvwrite(c.O/"04_ORIGINAL_DAG_P3_P4_P5_PATH_OWNER_AND_SHARED_DEPENDENCY_MAP.csv",maprows)
    c.csvwrite(c.O/"07_UPSTREAM_COMPENSATOR_ELIGIBILITY_LEGALITY_AND_RANKING.csv",elig)
    graph=dict(blocks=[dict(index=i,type=type(m).__name__,f=m.f,repeats=len(m.m) if type(m).__name__ in ("C3","C3Lite") else None)
                       for i,m in enumerate(parent.model)],detect_sources=head.f,anchors=head.anchors.cpu().tolist(),
               strides=head.stride.cpu().tolist(),class_names=parent.names,reach=reach,regions=regions)
    c.write(c.O/"aux/SOURCE_RUNTIME_DAG.json",graph)
    counter=r1.ExactComplexityModel(parent,groups)
    cache={};oracle_calls=0;cache_hits=0
    def oracle(w):
        nonlocal oracle_calls,cache_hits
        sig=c.signature(order,w)
        if sig in cache:cache_hits+=1;return cache[sig]
        a=counter.calculate(w);oracle_calls+=1
        value=(a["params"],2*a["macs"]);cache[sig]=value;return value
    baselines={n:oracle(w) for n,w in refs.items()}
    assert baselines==dict(Meta=(2507047,7847110400),Manual=(2512591,7775097600))
    pure={};pure_rows=[];pathvector=[]
    for direction,(recipient,donor) in c.DIRECTIONS.items():
        w=dict(refs[recipient]);w.update({g:refs[donor][g] for g in c.PATHS});pure[direction]=w
        p,f=oracle(w)
        pure_rows.append(dict(direction=direction,recipient=recipient,donor=donor,path_coverage=13,changed_path_ids=";".join(c.PATHS),
                             **c.metrics(p,f),signature=c.signature(order,w),evidence="ORIGINAL_COUNTER_VERIFIED"))
        for g in order:pathvector.append(dict(direction=direction,group_id=g,recipient_width=refs[recipient][g],donor_width=refs[donor][g],
                pure_joint_width=w[g],is_joint_path_gene=g in c.PATHS,donor_width_verified=w[g]==refs[donor][g] if g in c.PATHS else None))
    c.csvwrite(c.O/"05_STRICT_JOINT_13_GENE_DONOR_RECIPIENT_VECTORS.csv",pathvector)
    c.csvwrite(c.O/"06_TWO_DIRECTIONS_UNCOMPENSATED_ORIGINAL_COUNTER_RESULTS.csv",pure_rows)
    print("BASELINE PARITY PASS. Pure joints:",[(r["direction"],r["params"],r["raw_flops"]) for r in pure_rows],flush=True)
    all_rows=[];start=time.monotonic();bounds_hit=False
    def make_row(direction,comp,w,level=1):
        recipient,donor=c.DIRECTIONS[direction];p,f=oracle(w)
        m=c.metrics(p,f)
        return dict(candidate_id=direction+"-"+c.signature(order,w)[:16],direction=direction,recipient=recipient,donor=donor,level=level,
             path_coverage=13,P3_coverage="5/5",P4_coverage="4/4",P5_coverage="4/4",compensator_count=len(comp),
             compensator_ids=";".join(comp),normalized_offtarget_edit=str(sum(Fraction(abs(w[g]-refs[recipient][g]),next(x["original_width"] for x in groups if x["group_id"]==g)) for g in comp)),
             **m,signature=c.signature(order,w),width_vector=[w[g] for g in order],
             identical_to_recipient=w==refs[recipient],identical_to_donor=w==refs[donor])
    for count in range(len(back)+1):
        for combo in itertools.combinations(back,count):
            for direction,(recipient,donor) in c.DIRECTIONS.items():
                w=dict(pure[direction]);w.update({g:refs[donor][g] for g in combo})
                all_rows.append(make_row(direction,list(combo),w))
        if time.monotonic()-start>900:
            bounds_hit=True;break
    exhaustive=not bounds_hit and len(all_rows)==8192
    promising=[r for r in all_rows if r["HARD_CAP_PASS"] and r["ORIGINAL_B6_BAND_PASS"] and r["matching_tier"]==0 and not r["identical_to_recipient"] and not r["identical_to_donor"]]
    # Conditional Level2 is not used to improve a successful Level1 solution.
    level2_ran=False
    if not promising:
        level2_ran=True
        beam=[make_row(d,[],dict(pure[d]),2) for d in c.DIRECTIONS]
        seen={r["signature"] for r in all_rows}
        unique2=0;t2=time.monotonic()
        for depth in range(1,len(back)+1):
            next_rows=[]
            for row in sorted(beam,key=c.rank):
                direction=row["direction"];recipient,donor=c.DIRECTIONS[direction];w=dict(zip(order,row["width_vector"]))
                previous=row["compensator_ids"].split(";") if row["compensator_ids"] else []
                for gid in back:
                    if gid in previous:continue
                    for width in r1.parse_allowed(next(g["allowed_widths"] for g in groups if g["group_id"]==gid)):
                        if width==refs[recipient][gid]:continue
                        ww=dict(w);ww[gid]=width;sig=c.signature(order,ww)
                        if sig in seen:continue
                        seen.add(sig);unique2+=1
                        rr=make_row(direction,sorted(previous+[gid]),ww,2);all_rows.append(rr);next_rows.append(rr)
                        if unique2>=100000 or time.monotonic()-t2>900:break
                    if unique2>=100000 or time.monotonic()-t2>900:break
                if unique2>=100000 or time.monotonic()-t2>900:break
            beam=sorted(next_rows,key=c.rank)[:256]
            if unique2>=100000 or time.monotonic()-t2>900 or not beam:break
        promising=[r for r in all_rows if r["HARD_CAP_PASS"] and not r["identical_to_donor"] and not r["identical_to_recipient"] and r["matching_tier"]<=2]
    ranked=sorted([r for r in all_rows if r["HARD_CAP_PASS"] and not r["identical_to_donor"] and not r["identical_to_recipient"]],key=c.rank)
    assert ranked,"NO_FULL_JOINT_PROMISING_SOLUTION_REDUCED_COVERAGE_REQUIRED"
    selected=ranked[0]
    leading={d:next(r for r in ranked if r["direction"]==d) for d in c.DIRECTIONS}
    top=[r for d in c.DIRECTIONS for r in sorted([r for r in ranked if r["direction"]==d],key=c.rank)[:10]]
    def flat(row):
        rr=dict(row);rr["width_vector"]=";".join(map(str,rr["width_vector"]));return rr
    c.csvwrite(c.O/"09_ALL_EXPLORED_CANDIDATES_HASHED_RESOURCE_LEDGER.csv",[flat(r) for r in all_rows])
    c.csvwrite(c.O/"10_TOP_MATCHED_FULL_JOINT_CANDIDATES_BOTH_DIRECTIONS.csv",[flat(r) for r in top])
    c.csvwrite(c.O/"11_BUDGET_CAP_BAND_AND_MATCH_QUALITY_CERTIFICATES.csv",pure_rows+[dict(direction=r["direction"],recipient=r["recipient"],donor=r["donor"],
          path_coverage=13,changed_path_ids=";".join(c.PATHS),**c.metrics(r["params"],r["raw_flops"]),signature=r["signature"],evidence="ORIGINAL_COUNTER_VERIFIED") for r in top])
    controls=[];control_designs=[]
    for direction,best in leading.items():
        recipient,donor=c.DIRECTIONS[direction];pc=dict(zip(order,best["width_vector"]));comp=best["compensator_ids"].split(";")
        wc=dict(refs[recipient]);wc.update({g:pc[g] for g in comp})
        for role,w in (("R0",refs[recipient]),("RP",pure[direction]),("RC",wc),("RPC",pc)):
            p,f=oracle(w);controls.append(dict(direction=direction,role=role,paired_RPC=best["candidate_id"],path_donor_coverage=13 if role in ("RP","RPC") else 0,
                compensator_ids=";".join(comp) if role in ("RC","RPC") else "",**c.metrics(p,f),signature=c.signature(order,w),width_vector=";".join(str(w[g]) for g in order),
                physical_status="NOT_EXECUTED_CONTROL_DESIGN_ONLY"))
        exact=c.metrics(*oracle(wc))
        if exact["ORIGINAL_B6_BAND_PASS"] and exact["matching_tier"]==0:
            control_designs.append(dict(direction=direction,type="EXACT_SAME_COMPENSATOR_VECTOR",signature=c.signature(order,wc),widths=wc,**exact,
                same_compensators=True,new_control_design_needed=False))
        else:
            # Level1 states in the opposite joint direction span all recipient-path backbone-only controls.
            pool=[]
            opposite="R" if direction=="D" else "D"
            for rr in all_rows:
                if rr["direction"]!=opposite:continue
                ww=dict(zip(order,rr["width_vector"]))
                assert all(ww[g]==refs[recipient][g] for g in c.PATHS)
                edits=[g for g in back if ww[g]!=refs[recipient][g]]
                if not edits:continue
                rs=c.ratios(rr["params"],rr["raw_flops"])+[Fraction(abs(rr["params"]-best["params"]),best["params"]),Fraction(abs(rr["raw_flops"]-best["raw_flops"]),best["raw_flops"])]
                non_equiv=sum(ww[g]!=wc[g] for g in back)
                key=(not rr["HARD_CAP_PASS"],not rr["ORIGINAL_B6_BAND_PASS"],c.tier(rs),non_equiv,max(rs),sum(rs),len(edits),tuple(rr["width_vector"]))
                pool.append((key,rr,edits,non_equiv,rs))
            key,rr,edits,non_equiv,rs=min(pool,key=lambda x:x[0])
            control_designs.append(dict(direction=direction,type="ALTERNATIVE_BACKBONE_ONLY_RESOURCE_CONTROL",signature=rr["signature"],
                widths=dict(zip(order,rr["width_vector"])),**c.metrics(rr["params"],rr["raw_flops"]),same_compensators=non_equiv==0,
                off_target_gene_non_equivalence=non_equiv,changed_backbone_genes=edits,
                max_relative_to_refs_and_RPC=str(max(rs)),new_control_design_needed=True,exact_RC_metrics=exact))
    c.csvwrite(c.O/"14_R0_RP_RC_RPC_FACTORIAL_STRUCTURAL_RESOURCE_MATRIX.csv",controls)
    c.write(c.O/"aux/COMPENSATOR_CONTROL_DESIGNS.json",control_designs)
    c.write(c.O/"aux/SELECTED_AND_LEADING.json",dict(selected=selected,leading=leading,top=top,groups=order,backbone=back,refs=refs,pure=pure))
    c.csvwrite(c.O/"16_REDUCED_COVERAGE_SECONDARY_EXPLORATORY_IF_NEEDED.csv",[dict(status="NOT_APPLICABLE",
        reason="Full13/13 source-legal cap/band/tier1 designs found. No reduced coverage search authorized by trigger.",actual_secondary_evaluations=0)])
    summary=dict(start=started,end=c.now(),level1_exhaustive=exhaustive,level1_expected_states=8192,level1_explored_states=sum(r["level"]==1 for r in all_rows),
        level2_executed=level2_ran,level2_reason="NOT_TRIGGERED_LEVEL1_COMPARABLY_MATCHED_FULL_JOINT" if not level2_ran else "LEVEL1_NO_COMPARABLE_NEW_HYBRID",
        global_legal_R1_optimality_proven=False,binary_Level1_ranking_optimality_proven=exhaustive,explored=len(all_rows),
        oracle_invocations=oracle_calls,oracle_cache_hits=cache_hits,unique_cached_width_vectors=len(cache),search_elapsed_seconds=time.monotonic()-start,
        bounds_hit=bounds_hit,unexplored_space="Other R1 legal backbone widths not examined because Level2 trigger false" if not level2_ran else "Unfinished bounded beam space",
        selected=selected,caps_pass=sum(r["HARD_CAP_PASS"] for r in all_rows),bands_pass=sum(r["ORIGINAL_B6_BAND_PASS"] for r in all_rows),
        band_tier1_nonreplica=len(promising),training=0,optimizer_steps=0,backward=0,dataset_inference=0,GPU_work=0,HA_restarted=False)
    c.write(c.O/"SEARCH_SUMMARY.json",summary)
    c.write(c.O/"aux/ORACLE_RESOURCE_CACHE.json",{k:list(v) for k,v in cache.items()})
    # Per-layer original-counter capacity accounting is saved only for references, pure joints and top/control definitions.
    capacity=[];allmodels={"Meta":refs["Meta"],"Manual":refs["Manual"],"Pure_D":pure["D"],"Pure_R":pure["R"]}
    for d,r in leading.items():allmodels["Best_"+d]=dict(zip(order,r["width_vector"]))
    for d in control_designs:allmodels["Control_"+d["direction"]]=d["widths"]
    for r in controls:
        if r["role"]=="RC":allmodels["Exact_RC_"+r["direction"]]=dict(zip(order,map(int,r["width_vector"].split(";"))))
    detailed_calls=0
    for label,w in allmodels.items():
        a=counter.calculate(w);detailed_calls+=1
        assert (a["params"],2*a["macs"])==oracle(w)
        for layer in a["layers"]:
            i=layer["block_index"];reg="BACKBONE_ORIGIN" if i<=9 else "P3_INGRESS_SHARED" if i<=17 else "P4_P5" if i<=20 else "LATE_P5" if i<=23 else "DETECT_THREE_SEPARATE_BRANCHES_AGGREGATED"
            capacity.append(dict(model=label,region=reg,**layer,raw_flops=2*layer["thop_macs"],
                 attribution="SHARED_P3_P4_P5" if reg in ("BACKBONE_ORIGIN","P3_INGRESS_SHARED") else "SHARED_P4_P5" if reg=="P4_P5" else reg,
                 group_widths=";".join(f"{g['group_id']}={w[g['group_id']]}" for g in groups if g["block_index"]==i)))
    c.csvwrite(c.O/"12_PATH_AND_BACKBONE_BEFORE_AFTER_CAPACITY_ACCOUNTING.csv",capacity)
    c.write(c.O/"aux/DETAILED_ACCOUNTING_CALLS.json",dict(extra_original_calculate_calls=detailed_calls,resource_oracle_calls_after_reporting=oracle_calls,cache_hits_after_reporting=cache_hits))
    for r in inventory.values():assert c.sha(r["path"])==r["sha256"]
    print("FULL_JOINT_LEVEL1_EXHAUSTIVE",exhaustive,"states",len(all_rows),"selected",selected["candidate_id"],selected["params"],selected["raw_flops"],"comp",selected["compensator_ids"],flush=True)

if __name__=="__main__":main()
