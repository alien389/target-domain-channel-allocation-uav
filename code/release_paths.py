"""Explicit portable roots. No model imports, downloads or experiment actions."""
import os,re
from pathlib import Path
REPOSITORY=Path(__file__).resolve().parents[1]
DEFAULTS={'TRANSPLANT_CONSTRUCTION_ROOT': 'code/baselines/transplant/construction', 'DEPGRAPH_ROOT': 'code/baselines/depgraph', 'SCRATCH_ROOT': 'code/baselines/scratch', 'TRANSPLANT_ROOT': 'code/baselines/transplant', 'KD_ROOT': 'code/baselines/kd', 'BN_ROOT': 'code/baselines/bn_gamma', 'BN_TRAINING_ROOT': 'code/baselines/bn_gamma/training', 'STUDY_ROOT': 'code/study', 'YOLOV5_ROOT': 'code/yolov5', 'PRUNING_RESEARCH_ROOT': 'code/pruning_research', 'EXTERNAL_GENERALIZATION_ROOT': 'code/external', 'BASELINES_ROOT': 'code/baselines', 'CODE_ROOT': 'code', 'SOURCE_PARENT_HYP': 'configs/source_parent_hyp.yaml', 'DEPGRAPH_ARCHITECTURE_CONFIG': 'configs/baselines/depgraph/architecture.yaml', 'TRANSPLANT_ARCHITECTURE_CONFIG': 'configs/baselines/transplant/architecture.yaml', 'BN_GAMMA_ARCHITECTURE_CONFIG': 'configs/baselines/bn_gamma/architecture.yaml'}
def expand(value):
    def one(match):
        key=match.group(1)
        if os.environ.get(key):return os.environ[key]
        if key in DEFAULTS:return str(REPOSITORY/DEFAULTS[key])
        raise RuntimeError('Set required environment root: '+key)
    resolved = re.sub(r'\$\{([A-Z0-9_]+)\}',one,value)
    # Recorded Windows paths use backslashes even when opened on Linux.
    return resolved if os.name == 'nt' else resolved.replace(chr(92), '/')
