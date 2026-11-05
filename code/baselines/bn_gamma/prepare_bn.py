"""Pretrajectory source identity, physical counter reproduction and real BN mapping."""
from bn_common import *
import platform
import time


def main():
    started, wall, cpu = now(), time.perf_counter(), time.process_time()
    records = [{'path': str(p), 'sha256_before': a.sha(p), 'size_bytes': p.stat().st_size} for p in used_sources()]
    table('01_ORIGINAL_SOURCE_AND_HASH_AUDIT.csv', records)
    assert a.sha(a.SOURCE) == a.SOURCE_SHA
    assert a.sha(a.GENES) == a.GENES_SHA
    assert a.sha(CANDIDATE) == 'E7204DF94F17326D2D80CA84D78671F546AAFBCE7E77024A0C926E66E963F00F'
    actual_protocol = a.sha(PROTOCOL)
    assert actual_protocol == '98167179238D1078CFFAA3A7B8546C5AE3786A1550B6E3128C2365663048FC35'
    archived_hashes = json.loads((P/'10_final_fulltrain_300e_matched/pretraining_lock_hashes.json').read_text(encoding='utf-8-sig'))
    assert actual_protocol.lower() in json.dumps(archived_hashes).lower()
    model, genes = a.load_source()
    lines = initial_lineages(genes)
    assert len(model.state_dict()) == 319
    parent_count = a.counter(model)
    parent_independent = independent_counter(model)
    assert parent_count == parent_independent == (3985767, 10825984000)
    checkpoint = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    b1 = checkpoint['model'].cpu().float().eval()
    b1_count = a.counter(b1)
    b1_independent = independent_counter(b1)
    assert b1_count == b1_independent == (2508151, 7810252800)
    criterion = tp.importance.BNScaleImportance(group_reduction='mean', normalizer='mean')
    assert criterion.p == 1 and criterion.bias is False
    assert criterion.target_types == (torch.nn.modules.batchnorm._BatchNorm,)
    graph = a.graph(model)
    all_scores, coverage, component_map = {}, [], {}
    for gene in genes:
        gid, width = gene['group_id'], gene['original_width']
        group, closure = a.composite_group(model, graph, gene, list(range(width)))
        native = criterion(group)
        assert native is not None and len(native) == width and torch.isfinite(native).all()
        manual, components = bn_evidence(model, group, width)
        error = float((manual - native.double()).abs().max())
        assert torch.allclose(native.double(), manual, rtol=2e-6, atol=2e-7), (gid, error)
        distinct = len(set(float(v) for v in native))
        if gene['allowed_width_count'] > 1:
            assert distinct > 1, ('INDISTINGUISHABLE_BN_SCORES', gid)
        all_scores[gid] = [float(v) for v in native]
        component_map[gid] = {'components': components, 'closure': closure, 'manual_max_abs_error': error}
        coverage.append({'group_id': gid, 'root': gene['source_modules'], 'scope': gene['scope_name'],
                         'variable': gene['allowed_width_count'] > 1, 'original_width': width,
                         'allowed_widths': gene['allowed_widths'], 'real_affine_BN_count': len(components),
                         'BN_modules': [c['module'] for c in components], 'BN_original_index_mappings': components,
                         'finite': True, 'distinct_score_count': distinct, 'score_min': float(native.min()),
                         'score_max': float(native.max()), 'independent_max_abs_error': error,
                         'status': 'PASS', 'source_sha256': a.SOURCE_SHA})
    table('03_R1_35_GROUP_BN_GAMMA_COVERAGE.csv', coverage)
    dump('source_bn_scores_and_component_mappings.json', {'scores': all_scores, 'mapping': component_map,
                                                        'source_sha256': a.SOURCE_SHA})
    # The original gamma and the original API-normalized scores are frozen, not relearned or renormalized after pruning.
    environment = {'created_utc': now(), 'python': sys.version, 'executable': sys.executable,
                   'torch': torch.__version__, 'torch_pruning_distribution': importlib.metadata.version('torch-pruning'),
                   'torch_pruning_module': tp.__version__, 'thop': importlib.metadata.version('thop'),
                   'os': platform.platform(), 'device': 'CPU', 'torch_threads': torch.get_num_threads(),
                   'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'cuda_initialized': torch.cuda.is_initialized(),
                   'importance_source': str(IMPORTANCE), 'importance_sha256': a.sha(IMPORTANCE)}
    dump('environment_and_counter_prechecks.json', {'environment': environment, 'parent_counter': parent_count,
        'parent_independent_counter': parent_independent, 'archived_B1_counter': b1_count,
        'archived_B1_independent_counter': b1_independent, 'groups': len(genes), 'variable_groups': 25,
        'mapped_groups': len(coverage), 'state_keys': len(model.state_dict()),
        'protocol_sha256_clerical_correction': {'supplied': '98167179238D1078CFFAA3A7B854C5AE3786A1550B6E3128C2365663048FC35',
            'actual': actual_protocol, 'corroboration': str(P/'10_final_fulltrain_300e_matched/pretraining_lock_hashes.json')},
        'started_utc': started, 'finished_utc': now(), 'wall_seconds': time.perf_counter()-wall,
        'process_CPU_seconds': time.process_time()-cpu, 'status': 'PASS'})
    print('PREPARATION_PASS: parent and archived B1 exact; 35 BN groups mapped; independent scores match', flush=True)


if __name__ == '__main__':
    main()
