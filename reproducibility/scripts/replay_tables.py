"""Rebuild numerical audit tables from the retained ratios, observations and clocks."""
import argparse
from bisect import bisect_left
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import io
import itertools
import json
import math
from pathlib import Path
from zipfile import ZipFile


def csv_bytes(b):
    return list(csv.DictReader(io.StringIO(b.decode("utf-8-sig"))))


def integrate(samples,t0,t1):
    x=[int(r["host_monotonic_ns"]) for r in samples]
    p=[float(r["vdd_in_mw"])/1000 for r in samples]
    assert t0<t1 and x[0]<=t0<t1<=x[-1] and all(a<b for a,b in zip(x,x[1:]))
    def value(t):
        j=bisect_left(x,t)
        if x[j]==t: return p[j]
        return p[j-1]+(p[j]-p[j-1])*(t-x[j-1])/(x[j]-x[j-1])
    points=[(t0,value(t0))]+[(t,w) for t,w in zip(x,p) if t0<t<t1]+[(t1,value(t1))]
    return math.fsum((b-a)*1e-9*(v+w)*0.5 for (a,v),(b,w) in zip(points,points[1:]))


def energy(asset, emit=False):
    with ZipFile(asset) as z:
        root="Original_Evidence/"
        win=json.loads(z.read(root+"RAW_IDLE_TRACES/E01_pre_attempt1.window.json"))
        idle=integrate(csv_bytes(z.read(root+"RAW_IDLE_TRACES/E01_pre_attempt1.csv")),win["t0_ns"],win["t1_ns"])/300
        joined=[]
        for r in csv_bytes(z.read(root+"A1_ALL_160_ACCEPTED_RUNS.csv")):
            x,y=(int(r[a]) for a in ("t0_ns","t1_ns"))
            j=integrate(csv_bytes(z.read(root+"RAW_POWER_TRACES/"+r["name"]+".csv")),x,y)
            n=int(r["items"])
            gross=j/n
            assert math.isclose(j,float(r["joules"]),rel_tol=1e-10,abs_tol=1e-10)
            assert math.isclose(gross,float(r["gross_j_item"]),rel_tol=1e-10,abs_tol=1e-10)
            s=(y-x)*1e-9
            joined.append(dict(engine_id=r["engine_id"],run=r["name"],mode=r["mode"],gross_j_item=gross,
                               mean_vdd_in_w=j/s,pre_campaign_idle_w=idle,common_baseline_net_j_item=(j-idle*s)/n,
                               timed_duration_s=s,logger_duration_s=r["logger_duration_s"]))
        assert len(joined)==160 and len({r["run"] for r in joined})==160
        if emit:
            import sys
            w=csv.DictWriter(sys.stdout,list(joined[0]),lineterminator="\n"); w.writeheader(); w.writerows(joined)
        return len(joined),idle


def search_and_cost(repo):
    data=repo/"reproducibility"
    read=lambda p:csv_bytes((data/p).read_bytes())
    match=defaultdict(list)
    for r in read("search/kendall-exact-signature-matches.csv"):
        if r["included"]=="True":
            match[r["pair"]].append((Decimal(r["left_AP"]),Decimal(r["right_AP"])))
    for a in read("search/kendall-tau-b-audit.csv"):
        if not a["tau_b"] or not int(a["n_exact"]): continue
        if a["metric"]!="AP":
            c,d,x,y=(int(a[k]) for k in ("concordant","discordant","tie_x_only","tie_y_only"))
            tau=(c-d)/math.sqrt((c+d+x)*(c+d+y))
            assert math.isclose(tau,float(a["tau_b"]),abs_tol=1e-15)
            print(a["pair"],a["metric"],"archived-pair-count replay",tau)
            continue
        vals=match[a["pair"]]; assert len(vals)==int(a["n_exact"])
        c=Counter()
        for (x,y),(u,v) in itertools.combinations(vals,2):
            if x==u and y==v: c["both"]+=1
            elif x==u: c["x"]+=1
            elif y==v: c["y"]+=1
            elif (x-u)*(y-v)>0: c["C"]+=1
            else: c["D"]+=1
        n=c["C"]+c["D"]
        tau=(c["C"]-c["D"])/math.sqrt((n+c["x"])*(n+c["y"]))
        assert math.isclose(tau,float(a["tau_b"]),abs_tol=1e-15)
        print(a["pair"],len(vals),tau)
    stages=defaultdict(Decimal)
    jobs=read("cost-accounting/process-time-ledger.csv")
    for r in jobs:
        if r["seconds"]:
            stages[(r["stage"],r["component"])]+=Decimal(r["seconds"])
    assert len(jobs)==1049
    for k,v in sorted(stages.items()): print(*k,str(v/3600))
    slots=read("search/f1-logical-slots.csv")
    assert len(slots)==720 and len(read("search/f1-physical-evaluations.csv"))==500
    assert Counter((r["record_origin"],r["cache_hit"]) for r in slots)=={
        ("STAGE05_NEW","False"):465,("STAGE05_NEW","True"):111,("STAGE04_P0_REUSE","True"):144}
    assert len({r["source_sha256"] for r in read("cost-accounting/process-input-links.csv")})==522


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest="op",required=True)
    e=s.add_parser("energy"); e.add_argument("asset",type=Path)
    e=s.add_parser("search-cost"); e.add_argument("repository",type=Path)
    a=p.parse_args()
    if a.op=="energy": energy(a.asset,True)
    else: search_and_cost(a.repository)


if __name__=="__main__": main()
