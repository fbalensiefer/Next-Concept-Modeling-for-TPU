"""Generate shared manuscript evidence directly from committed experiment reports."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[4]
STUDIES = ('milestone4-scale-verified', 'scale-coverage-followup', 'scale-balanced-coverage', 'coarse-codebook-stability')
ARMS = ('reconstruction', 'alignment', 'dependency', 'alignment_dependency')


def generate():
    facts = {'implementation_commit': '1d600d7f4598ddb7e2808b29ee8f6eefe01cfb78', 'studies': {}, 'source_sha256': {}}
    macros = []
    def macro(name, value):
        macros.append('\\newcommand{\\' + name + '}{' + str(value) + '}')
    for name in STUDIES + ('milestone3-benchmark',):
        path = ROOT / 'reports' / name / 'report.json'
        facts['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    shared_teacher = None
    for name in STUDIES:
        r = json.loads((ROOT / 'reports' / name / 'report.json').read_text())
        identity = (r['teacher']['checkpoint_sha256'], r['teacher']['split_id_sha256'])
        if shared_teacher is None:
            shared_teacher = identity
            facts['shared_teacher'] = {'checkpoint_sha256': identity[0], 'split_id_sha256': identity[1]}
        assert identity == shared_teacher, (name, 'teacher or sampled splits changed')
        cfg = r['protocol']['config']
        seeds = cfg.get('confirmation_seeds', cfg['seeds'])
        means = {arm: statistics.mean(x['total_variation'] for x in r['results'] if x['arm'] == arm and x['seed'] in seeds)
                 for arm in ARMS if any(x['arm'] == arm for x in r['results'])}
        facts['studies'][name] = {'seeds': seeds, 'mean_output_tv': means,
                                 'followup_decision': r.get('followup_decision'),
                                 'stability_decision': r.get('coarse_stability_decision')}
    last = json.loads((ROOT / 'reports/coarse-codebook-stability/report.json').read_text())
    validation = json.loads((ROOT / 'reports/coarse-codebook-stability/validation.json').read_text())
    bench = json.loads((ROOT / 'reports/milestone3-benchmark/report.json').read_text())
    assert validation['passed'] == 44 and validation['failed'] == 0
    assert last['coarse_stability_decision']['evaluated_models'] == 20
    assert last['coarse_stability_decision']['passed'] and not last['followup_decision']['teacher_alignment_gate_passed']
    for source, expected in last['protocol']['implementation_sha256'].items():
        content=subprocess.check_output(['git', 'show', facts['implementation_commit'] + ':src/ncm/' + source], cwd=ROOT)
        # The sole change to historical modules is the opt-in maintenance hook.
        if source=='scale_experiment.py':
            content=content.replace(b'def make_codec_step(config, optimizer, *, maintenance=maintain_coarse):',b'def make_codec_step(config, optimizer):').replace(b'books, coarse, diagnostics = maintenance(',b'books, coarse, diagnostics = maintain_coarse(')
        assert hashlib.sha256(content).hexdigest() == expected, source
    # Pin both the recorded suite and the newly archived historical text logs.
    record = ROOT / 'reports/coarse-codebook-stability/validation.json'
    facts['source_sha256'][str(record.relative_to(ROOT))] = hashlib.sha256(record.read_bytes()).hexdigest()
    facts['historical_text'] = {}
    for stage in ('codec', 'prior'):
        folder = ROOT / 'reports/text-experiments' / stage
        for path in sorted(folder.iterdir()):
            facts['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = json.loads((folder / 'provenance.json').read_text())
        assert provenance['historical_record_not_a_new_run']
        assert provenance['checkpoint_sha256'] == bench['config'][stage + '_sha256']
        rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        facts['historical_text'][stage] = {'checkpoint_sha256': provenance['checkpoint_sha256'], 'observations': rows}
    codec_rows = facts['historical_text']['codec']['observations']
    assert codec_rows[0]['step'] == 100 and codec_rows[-1]['step'] == 500
    assert codec_rows[0]['train']['byte_accuracy'] == 1
    assert round(codec_rows[0]['validation']['byte_cross_entropy'], 3) == 1.949
    assert round(codec_rows[-1]['validation']['byte_cross_entropy'], 3) == 3.550
    assert (ROOT / 'examples/text/train.txt').stat().st_size == 26
    assert (ROOT / 'examples/text/validation.txt').stat().st_size == 13
    facts['final_protocol'] = last['protocol']['config']
    seeds = last['protocol']['config']['confirmation_seeds']
    new_runs = [x for x in last['results'] if x['seed'] in seeds]
    assert len(new_runs) == 20
    assert all(x['zero_support_teacher_mass'] == 0 for x in new_runs)
    assert all(x['population_coarse_active_codes'] == 2 for x in new_runs)
    assert min(x['population_sequence_reconstruction'] for x in new_runs) >= .98
    assert round(min(x['population_sequence_reconstruction'] for x in new_runs if x['arm'] == 'alignment_dependency') * 100, 2) == 99.07
    assert set((x['codec_step_flops_estimate'], x['prior_step_flops_estimate'], x['decoder_repair']['step_flops_estimate']) for x in last['results']) == {(10564359., 6996102., 5837030.)}
    facts['test_cases_passed'] = validation['passed']
    facts['environment'] = {k: bench[k] for k in ('platform', 'python', 'versions', 'dtype', 'device_kind')}
    facts['stability'] = last['coarse_stability_decision']
    facts['budget'] = last['budget']
    facts['benchmark_rows'] = []
    macro('TestCases', 44)
    macro('StabilityModels', 20)
    macro('StabilityPP', f"{facts['stability']['minimum_population_perplexity']:.4f}")
    macro('StabilityCollapsed', '0')
    macro('SourceRevision', '1d600d7')
    pilot = json.loads((ROOT / 'reports/milestone4-scale-verified/report.json').read_text())
    macro('PilotTCAlignment', f"{statistics.mean(x['hard_conditional_tc_sum'] for x in pilot['results'] if x['arm']=='alignment'):.4f}")
    macro('PilotTCCombined', f"{statistics.mean(x['hard_conditional_tc_sum'] for x in pilot['results'] if x['arm']=='alignment_dependency'):.4f}")
    macro('FinalAlignmentGain', f"{last['followup_decision']['mean_alignment_tv_improvement_over_dependency_only']:.5f}")
    macro('FinalAlignmentWins', last['followup_decision']['alignment_winning_seeds'])
    rows = []
    labels = ('Initial pilot', 'Decoder replay', 'Balanced codec', 'Coarse maintenance')
    for label, name in zip(labels, STUDIES):
        study = facts['studies'][name]
        numbers = [f"{study['mean_output_tv'][a]:.4f}" if a in study['mean_output_tv'] else '--' for a in ARMS]
        rows.append(label + ' & ' + ','.join(map(str, study['seeds'])) + ' & ' + ' & '.join(numbers) + r'\\')
    macros.append(r'''\newcommand{\StudyResultsTable}{%
\begin{table}[t]\centering\small
\caption{Mean exact output total variation (TV; lower is better). Columns R, A, D, and AD denote reconstruction, alignment, dependence, and their combination. Only the listed student seeds enter each row; the first row has no decoder replay, later rows include 200 replay updates. Different rows use different seed sets and interventions, so row-to-row means are not causal quality comparisons.}
\label{tab:studies}
\begin{tabularx}{\linewidth}{@{}Ylrrrr@{}}\toprule
Study & Student seeds & R & A & D & AD\\\midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabularx}\end{table}}''')
    rows = []
    for length in (8, 128, 512):
        source = bench['trained_results'] if length == 8 else next(x['results'] for x in bench['synthetic_results'] if x['length'] == length)
        cases = {x['model']: x for x in source if x['batch_size'] == 16}
        values = [cases[k]['median_ms'] for k in ('ncm_padded', 'ncm_actual_lengths', 'ar_kv_cache')]
        facts['benchmark_rows'].append({'length': length, 'batch_size': 16, 'median_ms': values, 'trained_tiny_fixture': length == 8})
        rows.append(f"{length} & {'Trained toy' if length == 8 else 'Untrained shape'} & " + ' & '.join(f'{v:.3f}' for v in values) + r'\\')
    macro('PaddedLatency', f"{facts['benchmark_rows'][-1]['median_ms'][0]:.1f}")
    macro('ActualLatency', f"{facts['benchmark_rows'][-1]['median_ms'][1]:.1f}")
    macros.append(r'''\newcommand{\CPUBenchmarkTable}{%
\begin{table}[t]\centering\small
\caption{Synchronized single-device CPU median generation latency (ms), batch size 16, float32, five warmups and 30 repetitions. NCM includes byte decoding; AR uses a KV cache. Length 8 uses trained toy models of width 16 and one block; lengths 128/512 use untrained width-64, two-block shape checks. These are not quality-matched speedups or TPU measurements.}
\label{tab:cpu}
\begin{tabularx}{\linewidth}{@{}rYrrr@{}}\toprule
Bytes & Scope & Padded NCM & Actual-length NCM & Cached AR\\\midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabularx}\end{table}}''')
    protocol_rows = []
    for label, name in zip(labels, STUDIES):
        cfg = json.loads((ROOT / 'reports' / name / 'report.json').read_text())['protocol']['config']
        known = [s for s in cfg['seeds'] if s not in cfg.get('confirmation_seeds', cfg['seeds'])]
        protocol_rows.append(label + ' & ' + (','.join(map(str, known)) or '--') + ' & ' + ','.join(map(str, cfg.get('confirmation_seeds',cfg['seeds']))) + ' & ' + str(len(cfg['seeds']) * len(cfg.get('arms', ['R','A','AD']))) + r'\\')
    macros.append(r'''\newcommand{\ProtocolSeedsTable}{%
\begin{table}[t]\centering\small
\caption{Completed finite-domain protocols. Known seeds are repeated initializations, not new independent confirmations. Every four-arm follow-up fits 32 student models, of which 20 use newly reserved student seeds; the teacher, domain and sampled datasets remain fixed.}
\label{tab:protocols}
\begin{tabularx}{\linewidth}{@{}Yllr@{}}\toprule
Protocol & Known-seed checks & Evaluation seeds & Models\\\midrule
''' + '\n'.join(protocol_rows) + r'''
\bottomrule\end{tabularx}\end{table}}''')
    return '% Generated by generate_results.py; do not hand-edit.\n' + '\n'.join(macros) + '\n', json.dumps(facts, indent=2) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    tex, data = generate()
    for name, text in [('observed_results.tex', tex), ('observed_results.json', data)]:
        path = ROOT / 'paper/generated' / name
        if args.check:
            if path.read_text() != text:
                raise SystemExit(f'Stale generated evidence: {path}')
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
    print('Shared manuscript evidence matches committed reports.')


if __name__ == '__main__':
    main()
