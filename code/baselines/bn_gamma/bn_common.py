"""CPU-only structural audit plumbing; immutable historical code is imported read-only."""
import release_paths as _release_paths  # Reviewed release-only path relocation.
import contextlib
import copy
import csv
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys

OUT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(OUT / 'mplconfig')
P = Path(_release_paths.expand('${STUDY_ROOT}'))
B = Path(_release_paths.expand('${BASELINES_ROOT}'))
V3 = B / 'depgraph'
sys.path.insert(0, str(V3))
import scope_adapter as a
a.OUT = OUT
import independent_validator as original_check
import torch
import torch_pruning as tp
from torch_pruning.pruner import function
from thop import profile
from utils.general import LOGGER
torch.set_num_threads(1)
torch.manual_seed(42)

PMIN, PMAX = 2440209, 2515679
FMIN, FMAX = 7618383104, 7854003200
STAGE10 = P / '10_final_fulltrain_300e_matched/scripts/stage10_run_training.py'
PROTOCOL = P / '10_final_fulltrain_300e_matched/stage10_protocol_lock.json'
PRUNER = Path(_release_paths.expand('${PRUNING_RESEARCH_ROOT}\\pruning\\structured_pruner.py'))
IMPORTANCE = Path(inspect.getfile(tp.importance.BNScaleImportance))
CANDIDATE = V3 / 'b1_v3_scope_complete_joint_resource_raw.pt'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def guard():
    if shutil.disk_usage(OUT).free < 200_000_000:
        raise RuntimeError('BOUNDED_DISK_GUARD_BEFORE_WRITE')


def dump(name, obj):
    guard()
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write(name, text):
    guard()
    (OUT / name).write_text(text, encoding='utf-8')


def table(name, rows, fields=None):
    guard()
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':')) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def legal(p, f):
    return PMIN <= p <= PMAX and FMIN <= f <= FMAX


def margins(p, f):
    return {'P_minus_Pmin': p-PMIN, 'Pmax_minus_P': PMAX-p, 'F_minus_Fmin': f-FMIN, 'Fmax_minus_F': FMAX-f}


def initial_lineages(genes):
    return {g['group_id']: list(range(g['original_width'])) for g in genes}


def refs(model):
    return model.model[-1].anchors.clone(), model.stride.clone(), copy.deepcopy(model.names)


def independent_counter(model, size=32):
    # Separately written reproduction of the archived structural_preflight expression.
    saved = LOGGER.level
    LOGGER.setLevel('ERROR')
    try:
        with contextlib.redirect_stdout(None), torch.no_grad():
            cloned = copy.deepcopy(model).to('cpu').to(torch.float32).eval()
            cloned.fuse()
            n = sum(p.nelement() for p in cloned.parameters())
            s = int(max(int(x) for x in cloned.stride))
            macs, _ = profile(cloned, inputs=(torch.zeros((1, 3, size, size)),), verbose=False)
            raw = int(round((float(macs) * 2.0) * ((640 / size) ** 2)))
        assert s == 32
        return int(n), raw
    finally:
        LOGGER.setLevel(saved)


def bn_evidence(model, group, width):
    """Independent FP64 scalar reconstruction, BN terms only, no scorer helper reuse."""
    names = {module: name for name, module in model.named_modules()}
    entries = []
    total = [0.0] * width
    for item in group:
        dep, indices = item
        layer = dep.layer
        if not isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            continue
        if dep.pruning_fn != function.prune_batchnorm_out_channels or not layer.affine:
            continue
        assert item.root_idxs is not None and all(i is not None for i in item.root_idxs)
        assert len(indices) == len(item.root_idxs)
        vals = layer.weight.detach().cpu().tolist()
        for i, r in zip(indices, item.root_idxs):
            total[r] += abs(float(vals[i]))
        entries.append({'module': names[layer], 'physical_indices': list(indices), 'root_indices': list(item.root_idxs),
                        'gamma': [float(vals[i]) for i in indices], 'affine': True,
                        'handler': dep.pruning_fn.__qualname__})
    if not entries:
        raise RuntimeError('NO_REAL_BN_GAMMA_MAPPING')
    reduced = [t / len(entries) for t in total]
    mean = math.fsum(reduced) / len(reduced)
    if not math.isfinite(mean) or mean <= 0:
        raise RuntimeError('ZERO_OR_NONFINITE_BN_NORMALIZATION')
    manual = torch.tensor([v / mean for v in reduced], dtype=torch.float64)
    return manual, entries


def architecture(model, genes, lineages):
    modules = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            modules.append([n, m.in_channels, m.out_channels, m.groups, list(m.kernel_size)])
        elif isinstance(m, torch.nn.BatchNorm2d):
            modules.append([n, m.num_features])
    widths = [len(lineages[g['group_id']]) for g in genes]
    h = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest().upper()
    return {'architecture_sha256': h(modules), 'width_vector_sha256': h(widths),
            'original_index_sha256': h(lineages), 'widths': widths}


def used_sources():
    files = [a.SOURCE, a.GENES, PROTOCOL, P/'10_final_fulltrain_300e_matched/pretraining_lock_hashes.json',
             STAGE10, PRUNER,
             Path(a.__file__), Path(original_check.__file__), IMPORTANCE,
             Path(_release_paths.expand('${YOLOV5_ROOT}\\models\\common.py')), Path(_release_paths.expand('${YOLOV5_ROOT}\\models\\yolo.py')),
             CANDIDATE]
    for pattern in ['*.json', '*.py', '*.md', '*.csv']:
        files.extend(sorted(V3.glob(pattern)))
    u1 = B / 'B1_U1_UNIFORM_B6_STRUCTURAL_FEASIBILITY'
    files.extend([u1/'06_U1_ALL_VISITED_WIDTH_STATES_AND_PHYSICAL_RESOURCES.csv', u1/'10_U1_FINAL_SCIENTIFIC_M03_S07_REPORT.md'])
    files.extend(sorted((B/'M03_U1C_SINGLE_ALPHA_CANONICAL_UPROUND_STRUCTURAL_ONLY').glob('run_*/17_FINAL_MACHINE_READABLE_DECISION.json')))
    u2 = B/'B1_U2_TWO_ZONE_WIDTH_SCALING_B6_STRUCTURAL_ONLY'
    files.extend([u2/'10_U2_FINAL_SCIENTIFIC_REPORT.md',u2/'11_U2_FINAL_GATE_DECISION.json'])
    # Protect prior arm decisions without reading their model solutions into the generator.
    for path in B.glob('B2*'):
        if path.is_dir():
            files.extend(sorted(path.glob('*GATE*.json')))
    return list(dict.fromkeys(p.resolve() for p in files if p.is_file()))
