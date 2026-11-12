import hashlib,json,shutil
from pathlib import Path
import numpy as np
root=Path('reports/dependency-validation');out=Path('reports/dependency-validation');out.mkdir(parents=True,exist_ok=True)
rows=json.loads((root/'results.json').read_text());assert len(rows)==480
# Read committed evidence; no ignored training artifacts are needed.
models=['independent','adjacent_pairs','conditional_tree','joint_fine'];domains=list(dict.fromkeys(r['domain'] for r in rows));base=[r for r in rows if r['model']=='independent'];summary={}
for d in domains:
 summary[d]={}
 for m in models:
  rr=[r for r in rows if r['domain']==d and r['model']==m];safe=[r for r in rr if r['codec_eligible']]
  summary[d][m]=dict(codecs=len(rr),eligible=len(safe),all_mean_tv=float(np.mean([r['output_tv'] for r in rr])),eligible_mean_tv=float(np.mean([r['output_tv'] for r in safe])),eligible_mean_validation_nll=float(np.mean([r['validation_code_nll'] for r in safe])),min_rounds=min(r['complexity']['serial_rounds'] for r in safe),max_rounds=max(r['complexity']['serial_rounds'] for r in safe))
misses=[dict(domain=r['domain'],seed=r['seed'],arm=r['arm'],validation_sequence_accuracy=r['validation_sequence_accuracy'],population_failures=r['population_codec_failures']) for r in base if r['codec_eligible'] and r['population_codec_failures']]
(out/'analysis.json').write_text(json.dumps(dict(summary=summary,eligible=sum(r['codec_eligible'] for r in base),rejected=sum(not r['codec_eligible'] for r in base),population_gate_misses=misses),indent=2)+'\n')
labels={'independent':'Independent','adjacent_pairs':'Adjacent pairs','conditional_tree':'Conditional tree','joint_fine':'Joint fine level'}
lines=['# Reconstruction-first dependency comparison','',
'Production defaults remain unchanged. This experiment freezes all 120 reference codecs from the six-domain study and fits four explicit code priors from their original 512 training observations. No codec, codebook, feature teacher or neural prior is retrained. Every result, including unsafe codecs, is retained.','',
'## Reconstruction protection and protocol','',
'An experimental eligibility check flags codecs whose validation sequence accuracy is below 98% or whose training collision-based reconstruction ceiling is below 98%. It uses sampled training/validation observations, not population probabilities. It accepts 104 codecs and flags 16. The exact population audit finds one additional accepted codec below the reconstruction threshold: persistent_markov, seed 34, alignment arm, validation accuracy 98.633%. Thus this gate is a useful filter, not a population guarantee or production admission certificate.','',
'Freezing the codec ensures that changing the prior cannot change its autoencoding fidelity; checkpoint hashes were checked before and after each comparison. It does not ensure that generated samples are accurate or that a prior can repair collisions already present. Validation NLL and population output quality are evaluation metrics only; neither selects a smoothing value, model topology or training update.','',
'The protocol was saved before fitting. All conditional tables use total symmetric Dirichlet pseudo-count one per conditional row, distributing the mass uniformly over its output categories. The coarse-bit table uses the same rule. The tree topology maximizes empirical conditional mutual information on training codes, using one shared topology across coarse states, deterministic tie breaking and a center root. Conditional tables are then smoothed; this regularized fit is not claimed to retain the unsmoothed Chow–Liu optimality theorem.','',
'## Results on codecs passing the sampled fidelity check','',
'Cells are mean exact decoded output TV; lower is better. All four objective arms remain represented, and eligibility is identical across the four priors for each codec. This selected subset is not evidence for all codecs.','',
'| Source | Eligible codecs /20 | Independent | Adjacent pairs | Conditional tree | Joint fine level |','|---|---:|---:|---:|---:|---:|']
for d,s in summary.items():lines.append('| '+d+' | '+str(s['independent']['eligible'])+' | '+' | '.join(f"{s[m]['eligible_mean_tv']:.4f}" for m in models)+' |')
lines+=['','## All codecs, including flagged failures','',
'| Source | Independent | Adjacent pairs | Conditional tree | Joint fine level |','|---|---:|---:|---:|---:|']
for d,s in summary.items():lines.append('| '+d+' | '+' | '.join(f"{s[m]['all_mean_tv']:.4f}" for m in models)+' |')
lines+=['','The empirical joint tables improve decoded TV over independent tables on every source in these averages. The Markov source is the important capacity counterexample: conditional trees outperform the larger joint tables at the fixed 512-observation budget. This is consistent with a useful sparsity bias and reduced estimation error; it is not a causal proof of that explanation. No model dominates for both dependency capacity and serial depth.','',
'## Complexity and practical limits','',
'| Prior | Free parameters at length eight | Sampling rounds |','|---|---:|---:|',
'| Independent | 17 | 2 |','| Adjacent pairs | 25 | 2 |','| Conditional tree | 31 | 3–6 observed |','| Joint fine level | 511 | 2 |','',
'Counts include the coarse distribution and all conditional probability rows, including unobserved contexts. Tree rounds include the coarse bit and the center-rooted fine tree. Block distributions can sample their categories in parallel; trees must respect parent dependencies. The joint model scales exponentially with fine-level length and is an oracle/finite-domain control, not a proposed full-sequence language architecture.','',
'The implementation enumerates codewords to evaluate the exact output distribution. Its fit_seconds measure local table construction and enumeration, not accelerator inference speed or end-to-end generation latency. The reported parameter counts describe conditional models; the enumeration arrays used for diagnosis are larger. No compiler FLOP or TPU speed claim is made.','',
'## Decision','',
'1. Retain production defaults and the reference quantizer.',
'2. Require fidelity checks before interpreting improvements in dependence or usage. Strengthen the sampled gate with prespecified uncertainty handling before treating it as an admission rule; the observed false acceptance shows why point estimates alone are insufficient.',
'3. Use joint fine-level tables as a finite-domain upper-capacity control. Use sparse conditional trees as the first tractable dependency-aware baseline, with explicit measurement of the extra serial rounds.',
'4. For longer sequences, test small joint blocks linked by sparse dependencies rather than assuming a globally factorized fine level or a full exponentially sized joint table.',
'5. Address codec information loss separately through fidelity-constrained training. The present experiment protects frozen codec fidelity; it does not implement a reconstruction-constrained training algorithm.','',
'These results support redesigning dependency handling before adopting a new quantizer. They do not establish language quality, a practical decoder replacement, or cross-domain generalization: each prior is fitted separately to the training observations of its source. Five student seeds and one feature teacher per source remain the evidence envelope. No population probabilities enter fitting or gate eligibility.','',
'## Validation and reproduction','',
'Five tests passed for positive normalized laws in every model family and a closed-form joint posterior predictive check. All 480 fitted laws normalize, every source codec checkpoint hash is unchanged, and all 120 codecs are retained. protocol.json records design and implementation identity; results.json preserves every model-level metric and conditional structure. The new code is experimental and has no production call path.','',
'```bash','PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m ncm.dependency_validation','PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m pytest -q tests/test_dependency_validation.py','```','',
'The runner requires the locally available fresh-domain checkpoints and data under runs/fresh-domain-validation and refuses an existing output protocol. This experiment adds no codec training or quantizer changes.','']
(out/'report.md').write_text('\n'.join(lines))
# Verify that the complete source set is represented and hashes match the local checkpoints.
keys={(r['domain'],r['seed'],r['arm']) for r in base};assert len(keys)==120
for key in keys:
 group=[r for r in rows if (r['domain'],r['seed'],r['arm'])==key];assert {r['model'] for r in group}==set(models)
 assert len({r['codec_sha256'] for r in group})==1
(out/'validation.json').write_text(json.dumps(dict(fitted_priors=480,frozen_codecs=120,all_four_models_per_codec=True,all_codec_hashes_unchanged=True,tests_passed=5,eligible_codecs=104,population_gate_misses=misses),indent=2)+'\n')
print('Verified 480 fits and 120 unchanged reference checkpoints. Report:',out/'report.md')
