"""Release adapter for authentic source interfaces, not a new implementation."""
import argparse,ast,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGETS={'stage10': 'code/study/10_final_fulltrain_300e_matched/scripts/stage10_run_training.py', 'scratch': 'code/baselines/scratch/b2_runner.py', 'joint-transplant': 'code/baselines/transplant/training_runner.py', 'kd15': 'code/baselines/kd/run_ft15.py', 'bn-gamma': 'code/baselines/bn_gamma/training/training_runner.py'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=sorted(TARGETS),required=True);p.add_argument('--source-interface',action='store_true',help='Read authentic parser declarations without importing or executing source')
    a,rest=p.parse_known_args();target=ROOT/TARGETS[a.stage]
    if a.source_interface:
        tree=ast.parse(target.read_text(encoding='utf-8-sig'));flags=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
                flags.extend(v.value for v in n.args if isinstance(v,ast.Constant) and isinstance(v.value,str))
        print(json.dumps({'source':TARGETS[a.stage],'literal_source_arguments':flags,'execution':'NOT_EXECUTED'},indent=2));return
    # Explicit invocation beyond the interface check executes the authentic source.
    sys.path[:0]=[str(target.parent),str(ROOT),str(ROOT/'code'),str(ROOT/'code/yolov5')]
    sys.argv=[str(target),*rest];runpy.run_path(str(target),run_name='__main__')
if __name__=='__main__':main()
