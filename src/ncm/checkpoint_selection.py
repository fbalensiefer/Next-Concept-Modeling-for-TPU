"""Rank real-text RVQ checkpoints and validate their codec trees strictly.

Run with the project JAX environment:
    python -m ncm.checkpoint_selection
"""
import json
import math
from pathlib import Path


COLLAPSE_PERPLEXITY = 2.0
MAX_LONG_CONTEXT_BER_REGRESSION = 0.05


def score_candidate(record, metrics_by_length):
    """Return transparent reconstruction/usage scores for one matching length."""
    length = record['length_bytes']
    row = metrics_by_length[length]
    stages = row['codebook_perplexity']
    book_size = record['codebook_size']
    collapsed = [i for i, value in enumerate(stages) if value < COLLAPSE_PERPLEXITY]
    all_rows = [metrics_by_length[x] for x in sorted(metrics_by_length) if x <= length]
    bers = [item['byte_error_rate'] for item in all_rows]
    regressions = [max(0.0, b - a) for a, b in zip(bers, bers[1:])]
    max_regression = max(regressions, default=0.0)
    stable = max_regression <= MAX_LONG_CONTEXT_BER_REGRESSION
    utilization = sum(stages) / (len(stages) * book_size)
    ber = row['byte_error_rate']
    stability_score = max(0.0, 1.0 - max_regression / MAX_LONG_CONTEXT_BER_REGRESSION)
    combined = 0.55 * (1.0 - ber) + 0.25 * stability_score + 0.20 * utilization
    return {
        'byte_error_rate': ber,
        'exact_match_rate': row['exact_match_rate'],
        'stage_perplexity': stages,
        'collapsed_stages': collapsed,
        'codebook_healthy': not collapsed,
        'ber_scaling_stable': stable,
        'maximum_longer_window_ber_regression': max_regression,
        'normalized_codebook_utilization': utilization,
        'combined_score': combined,
        'valid_loss': None,
    }


def _engine_config(run_config):
    return {
        'rvq': {'lengths': run_config['lengths'],
                'codebook_size': run_config['codebook_size'], 'dim': run_config['dim']},
        'transformer': {'width': run_config['width'], 'heads': run_config['heads'],
                        'layers': run_config['layers']},
        'losses': {'beta': 0.25, 'entropy': run_config.get('entropy', 0.0),
                   'concept': 0.0, 'prior': 0.0},
        'learning_rate': run_config['learning_rate'],
        'init_batch_size': 1,
        'token_vocab_size': 256,
    }


def validate_checkpoint_with_worker(config, seed, checkpoint):
    """Initialize the matching engine template and invoke the strict codec loader."""
    from . import engine_rpc

    handle = engine_rpc.dispatch('init_model', {'config': _engine_config(config), 'seed': seed})
    engine_id = handle['engine_id']
    try:
        result = engine_rpc.dispatch('load_codec', {'engine_id': engine_id, 'path': str(checkpoint)})
        return {'valid': True, 'codec_sha256': result.get('codec_sha256')}
    finally:
        engine_rpc._models.pop(engine_id, None)


def select(root: Path):
    root = Path(root).resolve()
    metrics_path = root / 'reports/reconstruction-pilot/reconstruction_metrics.json'
    metrics_doc = json.loads(metrics_path.read_text())
    metrics_by_length = {row['length_bytes']: row for row in metrics_doc['results']}
    expected_lengths = {32, 128, 512}
    if set(metrics_by_length) != expected_lengths:
        raise ValueError(f'reconstruction report must contain exactly {sorted(expected_lengths)}')
    report_rows, scored = [], []
    runs = root / 'runs/real-text-pilot'
    for config_path in sorted(runs.glob('length-*-seed-*/config.json')):
        config = json.loads(config_path.read_text())
        length = max(config['lengths'])
        checkpoint = config_path.parent / 'checkpoint.msgpack'
        if length not in metrics_by_length or not checkpoint.is_file():
            continue
        seed = int(config['seed'])
        row = {'length_bytes': length, 'seed': seed,
               'checkpoint': str(checkpoint.relative_to(root)),
               'source_config': str(config_path.relative_to(root))}
        try:
            validation = validate_checkpoint_with_worker(config, seed, checkpoint)
            row['strict_loader'] = validation
        except Exception as error:
            row['strict_loader'] = {'valid': False, 'error': str(error)}
        # The committed reconstruction metrics were generated only from seed zero.
        # Do not assign those measurements to other seed checkpoints.
        if seed == 0 and row['strict_loader']['valid']:
            row['scores'] = score_candidate(
                {'length_bytes': length, 'codebook_size': config['codebook_size']},
                metrics_by_length)
            scored.append(row)
        else:
            row['scores'] = None
            row['score_status'] = 'no matching per-seed reconstruction metrics'
        report_rows.append(row)
    if not scored:
        raise ValueError('no seed-zero checkpoint passed strict loading with matching metrics')
    scored.sort(key=lambda row: (-row['scores']['combined_score'],
                                 row['scores']['byte_error_rate'], row['length_bytes']))
    winner = scored[0]
    return {
        'protocol': 'real-text reconstruction checkpoint screening; no generation-quality evaluation',
        'reconstruction_report': str(metrics_path.relative_to(root)),
        'criteria': {
            'collapse_perplexity_threshold': COLLAPSE_PERPLEXITY,
            'maximum_longer_window_ber_regression': MAX_LONG_CONTEXT_BER_REGRESSION,
            'score_weights': {'inverse_ber': 0.55, 'ber_stability': 0.25,
                              'normalized_codebook_utilization': 0.20},
            'loss_metric': 'unavailable in reconstruction_metrics.json; excluded',
        },
        'candidates': report_rows,
        'ranked_scored_candidates': scored,
        'selected_codec_checkpoint': winner['checkpoint'],
        'selected_length_bytes': winner['length_bytes'],
        'selected_seed': winner['seed'],
        'generation_eligible': False,
        'generation_blocker': (
            'This selector evaluates codec reconstruction and codebook health only; it does not '
            'evaluate generation quality. Use the selected Phase 1 codec with a separately trained '
            'Phase 2 prior, and treat generation quality as unmeasured until a held-out benchmark.'
        ),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    result = select(root)
    output = root / 'reports/generation-pilot/checkpoint_selection_report.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'selected_codec_checkpoint': result['selected_codec_checkpoint'],
                      'generation_eligible': result['generation_eligible'],
                      'report': str(output.relative_to(root))}, indent=2))


if __name__ == '__main__':
    main()
