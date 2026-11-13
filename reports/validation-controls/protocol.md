# Frozen validation protocol

git_commit: 8f1fddc (implementation baseline; this protocol is committed before execution).
No network. No seeds, cases, thresholds or failed observations may be removed.
All generated evidence must include the execution commit hash; dirty source hashes
are also recorded. Existing regression tests retain their original seeds.

## Randomness and fixed thresholds

New root seeds: 200, 201, 202. Streams use fold-in study/case indices; these are
not replacement seeds. Power: speculative root 200, direct-floor root 201,
negative controls root 200, greedy prompts root 202. N=200000 sequences,
vocabulary 3, output length 4, draft length 3, latent cardinality 2; 20 direct
multinomial floor repetitions. The fixed pass threshold is the empirical 95th
percentile of the direct-TV floor, with no multiple-testing relaxation. Report
sqrt(support/(2*N)) as a descriptive conservative scale. A negative control is
rejected only when its TV strictly exceeds that same floor. Controls: q-law
mismatch (temperature 1.2), clipped residual (epsilon 0.05 before normalization;
simply omitting normalization in log-categorical cannot change its distribution),
future-token leakage, and proposal-distributed bonus. Statistical failures remain
failures; no rerun with a replacement seed. Greedy: 50 prompts, draft lengths
1,2,3,4,7, output length 11. Point-mass adapters define greedy teacher/draft laws.
Bookkeeping: prefixes 0,1,D-1,D, D=4, output lengths 1,3,5,9.
Cache/logit tolerance float32: atol=rtol=1e-5. Sharded deterministic tokens must
match exactly. Float64 likelihood-bound tolerance: 1e-9.

## Cache and padding

Reference remains full-prefix parallel teacher verification and an independently
KV-cached teacher baseline. No speculative teacher rollback claim is made.
Coarse cache is opt-in until identical codes and <=1e-5 logits are verified.
Uncached hybrid latency rows carry `pessimistic: no coarse KV cache`.
Cross-bucket equality is tested only for geometries with identical active grids
and transferred parameter prefixes; area resampling that changes active grids is
not assumed invariant. Unsupported geometries are explicitly reported.

## TC gate (run before any text TC measurement)

Seeds 200,201,202; L=8,10; lossless contiguous partitions (L//2,L-L//2).
Sources: symmetric Markov flip .15, noisy parity .10, copied bit, two-state HMM
(state persistence .9, emission correctness .85), constant-mode mixture (.5
mixture of Bernoulli .1 and .9), uniform. Float64 exact conditional TC by full
enumeration. Independent training/evaluation draw streams, training sizes
256,512,2048; evaluation=4096. Also report overlap of sequence identities: literal
identity-disjoint train/test sets are impossible for copied-bit's two-point
support; any affected case fails the strict disjoint-identity requirement and
cannot qualify the real-text gate. No silent reinterpretation of that condition.
Probe and parallel reference use shared CausalBlock primitives, width16,
heads2,layers1, Adam .003, 200 steps, batch64, no tuning/early stopping.
Error gate: abs(estimated-exact) <= max(.02 nats,.1*exact) for >=90% of all
(source,length,level,seed,training-size) cases, plus disjointness requirement.
Failing either closes the real-text gate. Decision threshold: coarse captures
>=30% of estimated total TC. No additional hybrid training based on the decision.

## Model experiments

Root seeds 200,201,202, paired data-order/initialization streams.
Local real text: examples/text/train.txt and examples/text/validation.txt;
raw hashes recorded before reading. Fixed continuation windows L=128 with
64 bytes available prefix. All prompt buckets use the SAME continuation windows
and masks; prompts are last P bytes of that prefix. Inference has no target input.
P sweep 0,16,64. Draft sweep 2,4,8,16. B sweep 1,16,128.
Architecture: NCM width8,heads2,layers1,dim4,codes4; teacher width16,layers2;
small AR nearest parameter count including codebook embeddings. Adam .0003.
Codec training 8 steps; teacher and priors/AR 8 steps; batch4. Checkpoints 2,4,8.
These are first local real-text feasibility runs, NOT converged language models.
Smoke: L=8,P=2,seed200,1 training step, same sweeps that fit the bucket.
Teacher-generated ablation: 32 sequences per seed/prompt bucket, frozen teacher,
train prior AND decoder for 8 steps on those sequences; real-text codec encoder
and codebooks frozen. Charge teacher generation plus ablation training.
Latency: five warmups, thirty synchronized observations, independently split
keys. Teacher reference is cached incremental AR with the full prefix.
Teacher warning if abs(teacher BPB - scratch-small-AR BPB) <=.3.

## Accounting/statistics/claim rules

All compiler FLOPs explicitly labeled compiler estimates. Analytic counts are
separate operation-model estimates (MAC=2), including training backward and
optimizer estimates with stated conventions. Speculative totals use measured
rounds; report per-round count and rounds separately. Include prefill, codec,
teacher distillation forward, teacher pretraining, teacher-sample generation.
Every result row has git_commit, likelihood kind, compiler estimate label and
analytic estimate label. NCM BPB always `upper_bound`, never exact likelihood.
Continuation denominators and available context are shared across arms.
Per-seed results and paired bootstrap 95% intervals (10000 resamples, seed202)
for b-a, hybrid-b, parallel-hybrid; three-seed intervals labeled exploratory.
BPB-vs-total-training-FLOPs CSV includes all prerequisite costs, without double
counting teacher forwards already present in compiled distillation steps.
Quality matching uses checkpoint curves only within overlapping BPB ranges;
no extrapolation. Exact AR BPB and NCM upper bounds are NOT commensurate exact
quality and do not support a quality-matched speedup claim. Report that limitation
alongside any numerical upper-bound intersection. No TPU or compute-saving claim.

Pre-execution clarification: output length is four, with draft blocks of three.
A three-token output cannot observe the first all-accepted bonus, so it cannot
identify the required bonus negative control. No statistical result existed
when this clarification was committed. All controls use the same four-token law.
