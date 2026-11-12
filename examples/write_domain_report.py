import json,hashlib
from pathlib import Path
import numpy as np
out=Path('reports/fresh-domain-validation');analysis=json.loads((out/'analysis.json').read_text());teachers=json.loads((out/'teachers.json').read_text());rows=json.loads((out/'results.json').read_text());baseline=json.loads((out/'source_baselines.json').read_text())
labels={'ema_recovery':'EMA + recovery','reference':'Reference','fixed_binary':'Fixed binary'}
lines=['# Fresh-seed, broader-distribution validation','',
'Completed 360 paired model runs on six prespecified full-support binary source distributions. The fixed binary coarse quantizer does not validate as a general replacement. All three conditions encounter failures outside the original domain, and factorized generation can remain poor despite perfect reconstruction. Production defaults were not changed.','',
'## Protocol and scope','',
'Student seeds 31–35 were not present in the earlier archived experiments. Six source laws, two sequence lengths, all four objective arms, and three stable quantizers were fixed before training. Each model received 400 codec updates and 200 decoder-only replay updates. Six independently initialized width-16 causal feature teachers received 400 updates each. Data budgets remained 512 training, 512 validation and 256 calibration draws; examples can repeat across splits, and this is not unseen-support testing.','',
'The paired treatments share data, parameters, K-means initialization, minibatch order, neural architecture, objectives, regularizers and schedules. Initial-state hashes match within every domain/seed across all arms and treatments. Fixed binary means fixed two-vector coarse endpoints with a trainable encoder and fine-book EMA; it is not full FSQ. No learned prior was trained: generation metrics are exact population-law oracles for the frozen codec.','',
'The source law itself is the sampling/evaluation target. Feature teachers supply only PCA alignment targets, preventing teacher fitting limitations from replacing the intended source dependence. This differs from the older pilot, which targeted a fitted teacher law; do not read changes from old TV values as a causal improvement. The original causal feature-alignment convention is intentionally retained to isolate maintenance.','',
'| Source | Length | Definition | Distinct training sequences | Source-to-feature-teacher KL |',
'|---|---:|---|---:|---:|']
definitions={'original_mixture':'0.75 constant modes + 0.25 uniform','biased_independent':'Independent Bernoulli(0.2) symbols','persistent_markov':'Fair initial bit; 0.85 probability of staying','noisy_parity':'0.90 uniform even parity + 0.10 uniform','skewed_modes':'0.80 all A + 0.10 all B + 0.10 uniform','alternating_modes':'0.80 alternating modes + 0.20 uniform'}
for domain in analysis['summary']:
 length=next(r['length'] for r in rows if r['domain']==domain)
 lines.append(f"| {domain} | {length} | {definitions[domain]} | {teachers[domain]['observed_training_sequences']} | {teachers[domain]['source_to_teacher_kl']:.5f} |")
lines += ['', '## All results after matched decoder replay','',
'Fidelity failure means any prespecified criterion fails: sequence reconstruction >=98%, missing source mass <=1%, outside-domain mass <=1%, or the collision-based reconstruction ceiling >=98%. All are population-weighted. Coarse balance is diagnostic rather than a universal criterion: a skewed source need not require balanced coarse codes. An encoding collision is also reported even when its probability mass is below the failure tolerance.','',
'| Source | Condition | Mean reconstruction | Fidelity failures /20 | Models with any collision | Mean oracle TV | Mean oracle KL, nats | Final coarse collapse /20 |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for domain,conditions in analysis['summary'].items():
 for treatment,item in conditions.items():
  m=item['mean'];lines.append(f"| {domain} | {labels[treatment]} | {100*m['sequence_reconstruction']:.3f}% | {item['fidelity_failures']} | {item['any_collision_models']} | {m['factorized_oracle_tv']:.5f} | {m['factorized_oracle_kl_nats']:.5f} | {item['final_coarse_collapse']} |")
lines+=['','## Failures by objective arm','','R = reconstruction, A = feature alignment, D = dependence, AD = both. Counts are failed fidelity checks out of five seeds. These rows prevent an average over objective arms from hiding a problematic training objective.','',
'| Source | Arm | EMA + recovery | Reference | Fixed binary |','|---|---|---:|---:|---:|']
armlabels={'reconstruction':'R','alignment':'A','dependency':'D','alignment_dependency':'AD'}
for domain,arms in analysis['by_objective_arm'].items():
 for arm,conditions in arms.items():
  if any(c['fidelity_failures'] for c in conditions.values()):
   counts=[conditions[t]['fidelity_failures'] for t in labels];lines.append(f"| {domain} | {armlabels[arm]} | {counts[0]} | {counts[1]} | {counts[2]} |")
lines+=['','## Paired comparisons','','Differences are candidate minus reference oracle TV, averaging objective arms within each seed first. Negative favors the candidate. Intervals enumerate all 5^5 bootstrap resamples of five seed means. They are descriptive, not multiplicity-adjusted noninferiority certificates. The prespecified 0.02 mean tolerance does not override fidelity failures.','',
'| Source | Candidate | Mean TV difference | Descriptive 95% interval |','|---|---|---:|---|']
for domain,conditions in analysis['paired_vs_reference'].items():
 for treatment,item in conditions.items():
  lo,hi=item['exact_seed_bootstrap_percentile_95'];lines.append(f"| {domain} | {labels[treatment]} | {item['mean']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
lines+=['','## Mathematical controls and interpretation','',
'The source-space controls use exact known conditionals, not learned networks, and are not matched-compute empirical baselines. They identify avoidable statistical restrictions:','',
'- Biased independent symbols admit exact one-round generation without a codec. Nonzero learned-code oracle error here is introduced by the representation/factorization, not an intrinsic dependence of the data.',
'- The persistent Markov source admits exact midpoint-bridge sampling in four rounds: positions [3], [1,5], [0,2,4,6], [7], using zero-based positions. The selected points separate independent remaining intervals.',
'- Noisy parity admits exact two-round generation: seven independent fair bits, followed by the final bit conditioned on them. A schedule with one coarse bit followed by eight independent codes is a different restriction, despite also using two rounds.','',
'Each construction was enumerated against the source law; residual TV is at floating-point roundoff. This demonstrates that dependence structure and the ordering of conditional decisions matter, rather than simply codebook balance or number of rounds. It does not establish that an unknown language distribution admits such simple exact schedules.','',
'The clearest failure example is noisy-parity seed 31, AD: fixed binary reconstructs only 25% of source mass and misses 75% under its factorized oracle. Recovery and reference also show objective-dependent failures. Thus the earlier binary simplification succeeded in a limited envelope, and neither centroid learning nor freezing alone protects information against the shared weighted objectives. Keep fidelity and collision checks ahead of TC/utilization improvements. Nominal code capacity is sufficient: 32 codewords for 16 inputs or 512 for 256 inputs. Learned collisions are not forced by cardinality.','',
'## Timing and resource record','',
'Each result records synchronized codec/replay training windows, compiler-estimated step FLOPs, compilation, shared initialization and shared data preparation. Other task-dispatched tests were not run concurrently. Times exclude evaluation and checkpoint I/O. Background system load remains uncontrolled; these measurements do not establish a speed advantage or TPU economics. One interrupted codec window has no persisted timing and is recorded as null, not imputed.','',
'| Length | Condition | Timed models | Median seconds /400 codec updates | Step FLOP estimates |','|---|---|---:|---:|---|']
for length in [4,8]:
 for treatment in labels:
  group=[r for r in rows if r['length']==length and r['treatment']==treatment];timed=[r['codec_seconds'] for r in group if r['codec_seconds'] is not None]
  lines.append(f"| {length} | {labels[treatment]} | {len(timed)}/{len(group)} | {np.median(timed):.5f} | {sorted(set(r['codec_step_flops'] for r in group))} |")
lines+=['','## Numerical incident and continuation','',
'After 268 completed models, an evaluation assertion compared direct code KL with a float32 entropy identity and stopped. On the saved affected checkpoint, float32 entropy cancellation produced a 7.52394e-6 nat discrepancy; float64 entropy and direct float64 KL agree within 1.11e-15. The correction changes evaluation verification only. Training precision, loss, data, seeds and maintenance functions were unchanged. Function-AST checks preserve the maintenance, teacher preparation and source-law helpers.','',
'The first 268 results were loaded unchanged, the affected model resumed from its completed trained checkpoint, and all remaining planned models were completed. Original protocol/source snapshots, continuation source versions, incident values and hashes are preserved. No failing seed was replaced and no result excluded.','',
'## Recommendation and limits','',
'Keep the current production default. Do not adopt fixed binary coarse endpoints as a generally validated replacement. Recovery remains a reasonable learned-codebook safeguard, but it does not solve the broader source/objective failures. The next experiment should protect reconstruction and collision fidelity while changing the dependency objective or sampling schedule; increasing maintenance complexity is not supported by these results.','',
'This broadens synthetic dependency patterns, sequence length and feature teachers. It does not validate natural-language corpora, larger alphabets or codebooks, long contexts, cross-domain transfer, external pretrained teachers, learned-prior quality, or TPU performance. Each source has one feature teacher and five student seeds; statistical generality remains limited. Models train separately on each distribution, rather than transferring to an unseen domain.','',
'## Artifacts and reproduction','',
'protocol.json freezes the original design; results.json contains all 360 records; analysis.json includes domain, objective-arm and paired-seed summaries. teachers.json records feature-teacher fidelity. source_baselines.json contains exact source-space controls. numeric_incident.json and continuation.json document the evaluation correction. Ignored runs/fresh-domain-validation contains all model checkpoints, teacher-domain arrays, original source snapshots and continuation source versions.','',
'```bash',
'PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m ncm.domain_validation',
'# If interrupted after protocol creation:',
'PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m ncm.domain_validation --resume',
'PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python examples/domain_source_baselines.py',
'PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python examples/summarize_domain_validation.py',
'```','',
'The runner starts from the committed example configuration and generates its own analytic domains and feature teachers; unlike the preceding ablation, it does not require an ignored historical teacher checkpoint. The report-writing step is presentation only.','']
(out/'report.md').write_text('\n'.join(lines))
print('Report written:',out/'report.md')
