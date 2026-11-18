"""Verify recorded percentile intervals from frozen replicate differences."""
import argparse
import csv
from pathlib import Path
import numpy as np


def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def check(root):
    matched=0
    cases=(
      ("carpk-replicates.csv","carpk-paired-bootstrap.csv","ci_low","ci_high",
       ("meta_minus_manual","meta_minus_randomsearch")),
      ("uavdt-replicates.csv","uavdt-cluster-bootstrap.csv","CI95_lower","CI95_upper",
       ("meta_minus_matched_manual","meta_minus_random_search")),
      ("bnscale-replicates.csv","bn-scale-paired-bootstrap.csv","CI95_lower","CI95_upper",
       ("mean_paired_delta","mean_paired_delta")),
    )
    for df,table,l,h,columns in cases:
        data=rows(root/df)
        for t,col in zip(rows(root/table),columns):
            x=[float(r[col]) for r in data if "split" not in t or r["split"]==t["split"]]
            target=5000 if df.startswith("uavdt") else 10000
            assert len(x)==target
            ci=np.quantile(x,[.025,.975],method="linear")
            assert np.allclose(ci,[float(t[l]),float(t[h])],rtol=0,atol=2e-12),(df,ci)
            print(df,t.get("split",col),*ci); matched+=1
    assert matched==6
    return matched


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("analysis",type=Path)
    check(p.parse_args().analysis)
