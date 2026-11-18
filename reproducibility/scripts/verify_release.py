"""Hash check and complete scientific-container checks for the staged bytes."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from payloads import replay_external,restore_plane,raw_tensor
from archive_io import names,require,digest,stream_hash
import table14_replay
import replay_tables
import replay_bootstrap

SCRIPTS=Path(__file__).resolve().parent
TOP=SCRIPTS.parents[2]


def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def verify(root,full):
    a=root/"GITHUB_RELEASE_ASSETS"; p=root/"GITHUB_REPOSITORY_PATCH"; d=p/"reproducibility"
    inv=rows(d/"manifests/files.csv")
    check={}
    for r in inv:
        rel=r["path"]; f=(root/rel).resolve()
        require(root in f.parents and rel not in check and f.is_file() and f.stat().st_size==int(r["bytes"]), "Manifest reference: "+rel)
        if full or r["role"]=="repository":
            with f.open("rb") as s: require(stream_hash(s)==r["sha256"],"Final bytes mismatch: "+rel)
        check[rel]=r["sha256"]
    reserved={d/"manifests/files.csv",a/"SHA256SUMS.txt"}
    actual={f.relative_to(root).as_posix() for x in (a,p) for f in x.rglob("*") if f.is_file() and f not in reserved}
    require(actual==set(check),"Untracked or absent public files")
    sums=dict((n,h.upper()) for h,n in (l.split("  ",1) for l in (a/"SHA256SUMS.txt").read_text().splitlines()))
    require(sums=={k.split("/",1)[1]:v for k,v in check.items() if k.startswith(a.name+"/")},"Asset checksum list mismatch")
    ext=rows(d/"manifests/external-payloads.csv")
    tbl=rows(d/"manifests/table14-member-index.csv")
    weights=rows(d/"checkpoints/checkpoint-index.csv")
    require(len(ext)==90 and len({(r["dataset"],r["model"],r["seed"]) for r in ext})==30,"External run identities")
    require(sum(bool(r["logical_tensor_sha256"]) for r in tbl)==2158,"TRT raw capture census")
    require(len(weights)==15 and Counter(r["seed"] for r in weights)=={"42":5,"43":5,"44":5},"Checkpoint census")
    widths=rows(d/"allocations/retained-widths.csv")
    require(len(widths)==10 and all(sum(k.startswith("DG-") for k in w)==35 for w in widths),"10 width vectors")
    g=json.loads((d/"architecture/dependency-groups-r1.json").read_bytes())["groups"]
    require(len(g)==35 and sum(x["allowed_width_count"]==1 for x in g)==10,"35 dependency scopes")
    require(len(list((d/"precision").glob("*-assignment.json")))==5 and
            len(list((d/"search").glob("fidelity-*-recall-lock.json")))==3,"Recall and precision locks")
    require(len(rows(d/"analysis/supplementary-seed-ap.csv"))==36,"H3 census")
    if not full:
        for f in sorted(a.glob("*.zip")):
            with ZipFile(f) as z: names(z)
        return {"mode":"smoke","repo_and_paths":"PASS","external":30,"raw_tensor_captures":2158,"weights":15}
    groups={}
    for r in ext: groups.setdefault((r["canonical_asset"],r["canonical_member"]),[]).append(r)
    tp={}
    for r in tbl: tp.setdefault(r["canonical_asset"],{})[r["canonical_member"]]=r
    raw=0; members=0
    for n in sums:
        with ZipFile(a/n) as z:
            names(z)
            for m in z.namelist():
                members+=1
                if (n,m) in groups:
                    g=groups[n,m]
                    with z.open(m) as f: out=replay_external(f)
                    require(out==([r["public_payload_sha256"] for r in g],[int(r["payload_bytes"]) for r in g],int(g[0]["split_images"])),
                            "Exact external replay: "+n+"!"+m)
                    with z.open(m) as f:
                        require(stream_hash(f)==g[0]["canonical_member_sha256"],"Prediction pack member hash")
                elif n in tp:
                    t=tp[n].get(m)
                    require(t is not None,"Unindexed Table14 canonical member: "+m)
                    b=z.read(m)
                    require(digest(b)==t["canonical_sha256"],"Table14 canonical bytes")
                    if t["logical_tensor_sha256"]:
                        q=restore_plane(b)
                        require(digest(q)==t["original_sha256"] and digest(raw_tensor(q).tobytes())==t["logical_tensor_sha256"],
                                "Final raw TRT bit pattern")
                        raw+=1
                    elif m.endswith(".npz"):
                        with ZipFile(io.BytesIO(b)) as npa:
                            names(npa)
                            require(set(npa.namelist())==set("image_names.npy detections.npy detection_offsets.npy correct.npy checkpoint_sha256.npy engine_sha256.npy".split()),"Label-free NumPy container")
                            for nn in npa.namelist():
                                require(not np.load(io.BytesIO(npa.read(nn)),allow_pickle=False).dtype.hasobject,"Non-numeric data")
                else:
                    with z.open(m) as f: stream_hash(f)
            if n in tp: require(set(tp[n])==set(z.namelist()),"Unreachable or extra Table14 member")
            if n.startswith("external-"): require(set(z.namelist())=={m for f,m in groups if f==n},"External member census")
    with ZipFile(a/"terminal-ema-15-safetensors.zip") as z:
        from safetensors.numpy import load
        inv=list(csv.DictReader(io.StringIO(z.read("tensor-sha256.csv").decode())))
        checks={}
        for r in inv: checks.setdefault((r["model"],r["seed"]),{})[r["tensor"]]=r
        for r in weights:
            b=z.read(r["sanitized_member"]); g=z.read(r["graph_member"])
            require(digest(b)==r["sanitized_sha256"] and digest(g)==r["graph_sha256"],"Weight/graph digest")
            ts=load(b)
            inv=checks[r["model"],r["seed"]]
            require(set(ts)==set(inv),"Tensor key census")
            for k,t in ts.items():
                e=inv[k]
                require(list(t.shape)==json.loads(e["shape"]) and t.dtype.str=={"F16":"<f2","F32":"<f4","I64":"<i8"}[e["dtype"]]
                        and digest(t.tobytes())==e["original_tensor_sha256"]==e["sanitized_tensor_sha256"],"Original tensor bytes")
    c=table14_replay.replay(p,a)
    e,idle=replay_tables.energy(a/"raw-energy-originals.zip")
    replay_tables.search_and_cost(p)
    boot=replay_bootstrap.check(d/"analysis")
    return {"mode":"full","status":"PASS","repo_release_files":len(check),"assets":len(sums),
            "zip_members_crc_read":members,"external_runs_exact_byte_replay":len(groups),"external_payloads":90,"raw_trt_tensor_roundtrips":raw,
            "table14_evaluator_txt_exact_replays":c,"energy_reintegrated":e,"pre_campaign_idle_w":idle,"tensor_equal_checkpoints":15,
            "exploratory_intervals_replayed":boot,"width_vectors":10,"recall_floors":3,"qdq_assignments":5,"H3_cells":36}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("root",type=Path,nargs="?",default=TOP)
    p.add_argument("--full",action="store_true"); p.add_argument("--write-json",type=Path)
    a=p.parse_args()
    r=verify(a.root.resolve(),a.full)
    s=json.dumps(r,indent=2)+"\n"; print(s,end="")
    if a.write_json: a.write_json.write_text(s,encoding="ascii")


if __name__=="__main__": main()
