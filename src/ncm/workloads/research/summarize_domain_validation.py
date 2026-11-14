"""Summarize the complete frozen fresh-domain experiment without selecting runs."""
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from ncm.domain_validation import DOMAINS, SEEDS, TREATMENTS
from ncm.scale_experiment import ALL_ARMS

import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--from-reports',action='store_true')
args=parser.parse_args()
root = Path('reports/fresh-domain-validation' if args.from_reports else 'runs/fresh-domain-validation')
output = Path('reports/fresh-domain-validation')
rows = json.loads((root/'results.json').read_text())
expected = {(d, s, a, t) for d, _, _ in DOMAINS for s in SEEDS
            for a in ALL_ARMS for t in TREATMENTS}
keys = [(r['domain'], r['seed'], r['arm'], r['treatment']) for r in rows]
assert len(keys) == len(expected) and set(keys) == expected
for name, _, _ in DOMAINS:
    for seed in SEEDS:
        assert len({r['initial_state_sha256'] for r in rows
                    if r['domain'] == name and r['seed'] == seed}) == 1

summary = {}
paired = {}
by_arm = {}
bootstrap_indices = np.asarray(list(itertools.product(range(len(SEEDS)), repeat=len(SEEDS))))
for domain, _, _ in DOMAINS:
    summary[domain] = {}
    for treatment in TREATMENTS:
        group = [r for r in rows if r['domain'] == domain and r['treatment'] == treatment]
        summary[domain][treatment] = dict(
            models=len(group), fidelity_failures=sum(bool(r['failures']) for r in group),
            reconstruction_failures=sum('reconstruction' in r['failures'] for r in group),
            coverage_failures=sum('missing_mass' in r['failures'] for r in group),
            collision_ceiling_failures=sum('collision_ceiling' in r['failures'] for r in group),
            any_collision_models=sum(bool(r['after']['code_collision_groups']) for r in group),
            final_coarse_collapse=sum(r['after']['active_coarse_codes'] < 2 for r in group),
            usage_window_failures=sum(r['collapsed_fraction'] > .05 for r in group),
            mean={metric: float(np.mean([r['after'][metric] for r in group])) for metric in (
                'sequence_reconstruction', 'factorized_oracle_tv', 'factorized_oracle_kl_nats',
                'factorized_missing_teacher_mass', 'factorized_outside_mass',
                'max_sequence_reconstruction_from_codes', 'coarse_perplexity')},
            min_reconstruction=min(r['after']['sequence_reconstruction'] for r in group),
            max_missing_mass=max(r['after']['factorized_missing_teacher_mass'] for r in group),
            median_codec_seconds=float(np.median([r['codec_seconds'] for r in group if r['codec_seconds'] is not None])),
            missing_codec_timings=sum(r['codec_seconds'] is None for r in group),
            codec_step_flops=sorted(set(r['codec_step_flops'] for r in group)))
    by_arm[domain] = {}
    for arm in ALL_ARMS:
        by_arm[domain][arm] = {}
        for treatment in TREATMENTS:
            group = [r for r in rows if r['domain'] == domain and r['arm'] == arm and r['treatment'] == treatment]
            by_arm[domain][arm][treatment] = dict(fidelity_failures=sum(bool(r['failures']) for r in group),
                mean_tv=float(np.mean([r['after']['factorized_oracle_tv'] for r in group])),
                mean_reconstruction=float(np.mean([r['after']['sequence_reconstruction'] for r in group])))
    paired[domain] = {}
    for candidate in ('ema_recovery', 'fixed_binary'):
        differences = []
        for seed in SEEDS:
            def mean_tv(treatment):
                return np.mean([r['after']['factorized_oracle_tv'] for r in rows
                                if r['domain'] == domain and r['seed'] == seed and r['treatment'] == treatment])
            differences.append(float(mean_tv(candidate)-mean_tv('reference')))
        diffs = np.asarray(differences)
        bootstrap_means = diffs[bootstrap_indices].mean(axis=1)
        paired[domain][candidate] = dict(seed_mean_tv_differences=differences,
            mean=float(diffs.mean()), wins=int((diffs < 0).sum()),
            exact_seed_bootstrap_percentile_95=np.percentile(bootstrap_means,[2.5,97.5]).tolist(),
            descriptive_mean_within_tolerance=bool(diffs.mean() <= .02))

output.mkdir(parents=True, exist_ok=True)
for filename in ('protocol.json', 'results.json', 'teachers.json', 'source_diagnostics.json', 'source_baselines.json', 'numeric_incident.json', 'continuation.json'):
    if not (root/filename).exists():
        if filename in ('protocol.json','results.json','teachers.json'):raise FileNotFoundError(root/filename)
        continue
    if (root/filename).resolve() != (output/filename).resolve():
        shutil.copyfile(root/filename, output/filename)
analysis = dict(summary=summary, by_objective_arm=by_arm, paired_vs_reference=paired,
    bootstrap_note='All 5^5 resamples of five seed-level, objective-arm-averaged differences. Descriptive percentile intervals; no multiplicity-adjusted noninferiority claim.',
    all_initial_hashes_matched=True, complete_models=len(rows))
(output/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
for domain in summary:
    print(domain)
    for treatment, item in summary[domain].items():
        print(treatment, 'TV', round(item['mean']['factorized_oracle_tv'],4),
              'recon', round(item['mean']['sequence_reconstruction'],4),
              'fail',item['fidelity_failures'],'collapse',item['final_coarse_collapse'])
