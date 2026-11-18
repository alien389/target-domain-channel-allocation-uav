"""Reconstruct frozen VisDrone evaluator inputs and check all retained comparison rows."""
import argparse
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import numpy as np


def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def replay(repo,assets,dest=None):
    d=repo/"reproducibility"
    count=0
    ids=set()
    with ZipFile(assets/"table14-full-split-predictions-and-capture-metadata.zip") as z:
        for r in rows(d/"manifests/table14-evaluator-index.csv"):
            path=r["evaluator_relative_path"]
            p=PurePosixPath(path)
            if p.is_absolute() or ".." in p.parts or path in ids:
                raise ValueError("Unsafe or duplicate evaluator path: "+path)
            ids.add(path)
            b=z.read(r["canonical_member"])
            if r["encoding"]!="native":
                a=np.load(io.BytesIO(b),allow_pickle=False)
                if a.ndim!=2 or a.shape[1]!=6:
                    raise ValueError("Invalid numerical detections")
                ending="\r\n" if r["encoding"]=="CRLF" else "\n"
                b="".join(f"{x:.10f},{y:.10f},{x2-x:.10f},{y2-y:.10f},{s:.10f},{int(c)+1},-1,-1"+ending
                          for x,y,x2,y2,s,c in a.tolist()).encode("ascii")
            if hashlib.sha256(b).hexdigest().upper()!=r["original_evaluator_sha256"]:
                raise ValueError("Mismatched archived evaluator bytes: "+path)
            if dest:
                out=dest/path
                out.parent.mkdir(parents=True,exist_ok=True)
                with out.open("xb") as f: f.write(b)
            count+=1
    assert count==43160
    data=rows(d/"evaluation/backend-parity/matched-backend-comparisons.csv")
    seen=set()
    for r in data:
        a,b,c,e=[float(r[k]) for k in ("historical_PT_rect","new_laptop_PT_square","new_Jetson_TRT_square","historical_Jetson_TRT_square")]
        for k,v in (("B_geometry_plus_laptop_protocol_minus_historical_PT",b-a),
                    ("C_cross_device_backend_residual_TRT_minus_PT_same_tensor",c-b),
                    ("D_new_common_TRT_minus_historical_TRT",c-e),("original_TRT_minus_PT_opposite_sign_observation",e-a)):
            if abs(float(r[k])-v)>1e-15:
                raise ValueError("Comparison arithmetic "+k)
        seen.add((r["split"],r["model_id"]))
    assert len(seen)==10
    print(f"PASS: {count} historical evaluator inputs and {len(seen)} model/split comparisons")
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("repository",type=Path); p.add_argument("assets",type=Path)
    p.add_argument("--write",type=Path)
    a=p.parse_args(); replay(a.repository,a.assets,a.write)


if __name__=="__main__": main()
