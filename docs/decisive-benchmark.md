# Hybrid NCM and teacher verification

`NCMConfig(ar_coarse_levels=1, prompt_len=16)` enables autoregression on the
first RVQ grid and a static prompt bucket. `SharedLevelGenerator` accepts
`prompt_ids` (byte IDs, with 256 reserved for BOS), or `prompt_embeddings` of
Transformer width, and an optional `prompt_mask`. `Batch` also accepts prompt
IDs and a mask for the standard training API. Prompt inputs must contain only
observed prefix information. Codec targets encode the continuation, independently
of the prompt. The default configuration retains the historical scanned path.

Coarse training uses shifted code IDs and causal attention at the actual grid
resolution. Sampling fills that grid sequentially; fine grids remain parallel.
Attention and MLP projections share parameters across grids, and use the same
`CausalBlock` implementation as the AR baseline. Inference currently recomputes
coarse attention at each position; only the AR baseline has a KV cache.

## Run the experiment

Supply separate UTF-8 `.txt` files or JSONL documents with unique `id` and `text`
fields. Exact content and JSONL identity overlap are rejected. All document tails
are retained with explicit masks. Use published corpus SHA256 values to pin the
inputs; the report records actual hashes even when expected hashes are omitted.

```sh
python scripts/run_decisive_benchmark.py \
  --train /data/train.jsonl --heldout /data/test.jsonl \
  --train-sha256 TRAIN_DIGEST --heldout-sha256 TEST_DIGEST \
  --output runs/decisive-128 --length 128 \
  --steps 10000 --codec-steps 10000
```

Repeat with `--length 512` for the second bucket. Defaults measure batches
1, 16, and 128 with five warmups and thirty synchronized repetitions. Checkpoint
files, `results.json`, and `results.csv` are written to the output directory.
Use a separate output directory for each experiment. Fixed seeds and corpus
hashes make data ordering and initialization reproducible on a fixed backend.
The loader also supports an explicit subword tokenizer with `encode`/`decode`;
the experiment itself uses bytes so its BPB denominator is unambiguous.

The script trains a larger teacher, scratch and logit-distilled small AR models,
a codec, and two frozen-codec priors (parallel and hybrid). Prior training adds
teacher-logit KL through a differentiable expected-code decoder. This is a
teacher-forced training surrogate, not a claim of exact marginal distillation.
The small AR width is selected to approximate total deployed hybrid NCM
parameters, including codebook embeddings and excluding EMA statistics; reports
include actual counts and mismatch fractions. The teacher
is trained for the same number of steps, without an assumption that it has
converged. Training budgets and model sizes must be increased for substantive
quality conclusions.

AR BPB is causal held-out likelihood. NCM BPB is the joint code-and-reconstruction
negative log likelihood of deterministic encoder codes, an upper bound on
marginal BPB. These numbers must retain their labels in comparisons. Descriptive
free-running diagnostics include byte-bigram TV, distinctness, UTF-8 validity,
and raw sample bytes. They do not measure semantic quality.

Compiler FLOPs come directly from `lowered.cost_analysis()`. They describe the
lowered computation; loop-body estimates are not multiplied by data-dependent
iteration counts, and are not a hardware FLOP counter. Training costs include
separate codec and teacher entries. Timing includes codec decoding and full
teacher-prefix verification. A separate cached-teacher reference measures the
same target distribution; speculative rows report its median latency divided by
speculative median latency. No speedup or TPU performance is implied by a CPU
smoke test.

## Exact speculative sampling

`speculative_generate` takes two `DistributionAdapter` objects containing dynamic
parameters and static pure apply functions. Their shape contracts are documented
in `ncm.speculative`. A draft adapter returns independent token logits given
prefix and sampled latent auxiliary variables. The verifier samples those token
probabilities itself; passing logits that differ from the actual proposal law
would invalidate the guarantee. Correlated token proposals need their true
sequential conditional probabilities and are outside this adapter contract.

The teacher evaluates every proposed prefix in one causal pass, including the
bonus conditional. Rejection samples from normalized `(p-q)+`; complete
acceptance samples a bonus from the teacher. Draft latent variables may be
sampled anew each round: conditional on any such variables, the correction
still returns the teacher law. The benchmark adapter uses the frozen decoder
logits conditional on sampled hierarchical codes, so the token draft is parallel.
The teacher sees the full prefix; the student sees its most recent prompt bucket.
The returned continuation has exactly `max_new_tokens` entries. Acceptance rate
counts accepted draft positions divided by eligible proposed positions; tokens
per pass counts committed tokens, including correction/bonus tokens and final
block truncation. There is no EOS stopping in this fixed-length benchmark.

## Verification

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  python -m pytest tests/test_hybrid_and_speculative.py -q
```

Tests cover future-code masking, prompt sensitivity, inference target invariance,
zero-mass draft proposals, a history-dependent teacher's joint distribution,
all-acceptance bonus/truncation, gradients, dp/fsdp agreement, hash verification,
and padded document tails.

## Frozen validation controls

The protocol and all root seeds are frozen in
`reports/validation-controls/protocol.md`. New study seeds are 200–202; existing
regression tests retain their historical seeds. No network is needed by the
studies. The real-text control runner requires the already-local processed
Gutenberg books under `runs/real-text-pilot/`, pinned to the protocol's hashes.

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  python -m pytest -q
python scripts/run_tc_probe.py
python scripts/run_drafter_comparison.py --profile smoke \
  --output reports/validation-controls/smoke
python scripts/run_drafter_comparison.py --profile real128 \
  --output reports/validation-controls/real128
```

CI runs the full test suite, including the powered 200,000-sequence verifier
study and four-device tests, on every push and pull request. The full model
sweeps and TC training study are explicit experiments, not CI jobs.

`DistributionAdapter.draft_fn`, when supplied, returns sampled tokens and their
true sequential conditional logits. It is used by the KV-cached distilled AR
control. The independent decoded-logit adapter remains unchanged. The verifier
reports accepted-prefix counts by position and actual teacher round counts.
Both AR draft and teacher prefixes are rebuilt per speculative round; no teacher
cache rollback is claimed. `ar_adapter` and the direct cached-teacher reference
use exactly the same Transformer projections.

Coarse inference accepts `use_cache=True`. This is opt-in to preserve existing
sampling paths. Cache equivalence tests cover two coarse grids, prompt masks,
padding holes, greedy sampling and seeds 200–202. The new drafter comparison
measures the cache path. The original benchmark's uncached hybrid rows are
explicitly labeled `pessimistic: no coarse KV cache`.

The comparison outputs `drafter_comparison.csv`, `bpb_vs_train_flops.csv`,
`checkpoint_curves.csv`, `latency_at_matched_bpb.csv`, `paired_statistics.json`,
per-seed checkpoints with commit metadata, and detailed results JSON. The draft
grid is 2/4/8/16; prompt grid is 0/16/64; batches are 1/16/128. A teacher-sampled
sequence distillation ablation trains a prior and decoder with encoder/codebooks
frozen. Its 32 teacher generations and prerequisite teacher/codec training are
charged. Ordinary logit distillation teacher forwards are already inside the
compiled training step and are not charged twice.

Every reported compiler FLOP value is a **compiler estimate**. Analytic estimates
use MAC=2, dense stored attention capacity, an approximate backward cost of twice
forward cost, and an explicit Adam operation allowance. They exclude memory,
communication, RNG and indexing. Loop inference reports per-round estimates and
measured rounds. The per-sequence round average is a compute diagnostic; vmapped
batch latency follows the slowest sequence. These estimates are not hardware
counters. NCM BPB is always an **upper bound**. A numerical intersection with an
AR likelihood curve does not establish equal quality. The matched table reports
non-overlap without extrapolation and disallows exact-quality claims for bounds.
Three-seed bootstrap intervals are explicitly exploratory.

The learned TC probe compares a trained causal within-level model with a frozen
parallel reference, using the same block primitives. Enumeration uses float64
source laws and lossless contiguous partitions. The real-text gate is closed if
either the fixed numerical threshold or strict sequence-identity disjointness
fails. Copied-bit's two-point support makes literal disjoint identity sets
incompatible with evaluating the same nondegenerate source law; this is recorded
as a failed condition, not silently reinterpreted. Consult `FINDINGS.md`,
`claims.md`, and `tc_probe_decision.md` before making any TC or performance claim.

For the zero-prompt fairness control, the runner directly selects the existing
HybridGenerator core for both parallel and hybrid priors. It does not select the
legacy auto(0,0) padded path. Empty prompt embeddings avoid counting an unused
lookup table at P=0. No public generator default is changed. Discarded timings
from overlapping validation jobs are retained in `real128-contended/`; the clean
run starts only after all 118 four-device tests complete. Use
`python scripts/summarize_validation_controls.py` after all three seeds finish
to assemble the claim ledger, seed outcomes, paired intervals, and final tables.


Final analytic audits distinguish a cache-only prefill from a complete decode
step: the last layer contributes K/V projections, without its unused attention,
MLP, or head. RVQ distance estimates count the stored padded K*L grid. Historical
estimates from the frozen a1ed801 execution remain untouched in its study
folder. `scripts/revise_validation_estimates.py` derives corrected tables and
retains raw-estimate columns and the analytic revision. Measured timings, BPB,
acceptance and round counts are never rewritten. Compiler estimates retain
the original execution revision. Fused QKV and elementwise counts remain
nominal analytic estimates, not hardware counters.

The fixed width grid gives approximate total-parameter matching, not exact
matching: AR counts exceed the NCM target by about 14.6%, 7.3%, and 2.5% for
initial prompt lengths 0, 16, and 64. No architecture is retuned after observing
results. Initial prompt bytes are shared; subsequent NCM rounds condition on
the most recent prompt bucket while the AR drafter uses the full committed
prefix. These limitations are explicit in the final tables and prohibit a
general student-advantage claim. Bootstrap intervals for sequence distillation
versus real-text NCM are separate from the NCM-versus-AR drafter contrast.

CI uses `.github/validation-constraints.txt` to reproduce the locally tested
package versions. The final test count is recorded in
`reports/validation-controls/tests.json`; earlier test counts in this guide
refer to earlier validation revisions.
