"""Recreate the archived CARPK/UAVDT COCO annotation projection from local provider text."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re


def read_table(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def compile_gt(dataset, manifest, labels):
    obs=read_table(manifest)
    n=459 if dataset=="carpk" else 16592
    if len(obs)!=n or [int(r["order"]) for r in obs]!=list(range(1,n+1)):
        raise ValueError("Expected the exact released ordered split")
    groups=defaultdict(list)
    if dataset=="uavdt":
        for seq in sorted({r["image_id"].split("/")[0] for r in obs}):
            for line in (labels/(seq+"_gt_whole.txt")).read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():continue
                frame,track,l,t,w,h,view,occ,cat=[int(float(x)) for x in line.strip().split(",")]
                if w<=0 or h<=0 or cat not in (1,2,3) or view not in (1,2,3) or occ not in (1,2,3,4):
                    raise ValueError("Invalid UAVDT ground truth")
                groups[seq,frame].append((cat,l,t,w,h,view,occ))
    images=[]; anns=[]
    order=list(enumerate(obs,1))
    if dataset=="carpk":
        # The original prep enumerated filenames, assigning COCO IDs from the split list.
        order.sort(key=lambda pair:pair[1]["image_id"])
    for i,r in order:
        ident=r["image_id"]; w=int(r["width"]); h=int(r["height"])
        images.append(dict(id=i,file_name=ident+(".png" if dataset=="carpk" else ""),width=w,height=h))
        if dataset=="carpk":
            gt=[]
            for line in (labels/(ident+".txt")).read_text(encoding="utf-8-sig").splitlines():
                if not line.strip(): continue
                x,y,x2,y2,c=map(int,line.split())
                if c!=1 or not (0<=x<=x2<w and 0<=y<=y2<h):
                    raise ValueError("Invalid CARPK box")
                gt.append((c,x,y,x2-x+1,y2-y+1,None,None))
        else:
            seq,im=ident.split("/")
            frame=int(re.fullmatch(r"img(\d+)\.jpg",im)[1])
            gt=groups.pop((seq,frame),[])
        for c,x,y,bw,bh,view,occ in gt:
            a=dict(id=len(anns)+1,image_id=i,category_id=c,bbox=[x,y,bw,bh],area=bw*bh,iscrowd=0)
            if dataset=="uavdt": a.update(out_of_view=view,occlusion=occ)
            anns.append(a)
    if groups or len(anns)!=(47500 if dataset=="carpk" else 375884):
        raise ValueError("Annotation count differs from the archived evaluator input")
    classes=["car"] if dataset=="carpk" else ["car","truck","bus"]
    return dict(info={},licenses=[],images=images,annotations=anns,
                categories=[dict(id=i,name=t) for i,t in enumerate(classes,1)])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("dataset",choices=["carpk","uavdt"])
    p.add_argument("manifest",type=Path);p.add_argument("provider_annotation_directory",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args()
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(compile_gt(a.dataset,a.manifest,a.provider_annotation_directory),f)
