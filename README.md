# Next-Concept Modeling for TPU

A JAX/Flax implementation of [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) for TPU-oriented execution. A multiscale residual vector quantizer encodes text into hierarchical codes, and a shared Transformer predicts each level's positions in parallel.

Project by Fabian Balensiefer. The underlying NCM method is due to Kumbong et al.

**[Research paper (PDF)](output/pdf/ncm_systems_paper.pdf)**

## SCALE framework and completed mechanism studies

**[SCALE: Dependency-Aware Representation Compilation from Pretrained Transformers to Hierarchical Generative Models (PDF)](output/pdf/SCALE_research_paper.pdf)** — Fabian Balensiefer, November 12, 2026.

SCALE proposes transferring pretrained Transformer capabilities into hierarchical NCM generators through representation alignment, dependency-aware residual quantization, selective parameter reuse, and constrained distillation. The revised paper retains its conditional theory and integrates the completed finite-domain controls, coverage repair and coarse-codebook stabilization studies. All 20 new-seed models passed the final hard-stability gate; the reliable teacher-alignment gate still failed. The full pretrained, cross-size SCALE compiler remains a research proposal. Both papers share report-derived results, notation and bibliographic metadata; see [sources, build instructions and consistency audit](paper/README.md).

## Implementation

The core includes mask-aware quantization, functional EMA codebooks, combined reconstruction and prediction losses, runtime codebook freezing, byte fallback routing, and explicit `dp`/`fsdp` sharding. See the paper for the formulation and limitations.

- [Model and training code](src/ncm/)
- [Training example](examples/train.py)
- [Tests](tests/)

## Run

Python 3.12 is the tested development environment. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/train.py
```

## Status

CPU tests cover numerical correctness, checkpoint replay, sampling, exact dependence diagnostics and virtual four-device sharding. Local CPU benchmarks and finite-domain experiments are available; TPU performance and useful language-model quality remain unverified. Rust data tooling and a complete byte-stream codec are planned.

[Validation details](paper/validation.json) · [Citation](CITATION.cff) · [MIT license](LICENSE)

## Reproducible text reconstruction experiment

```bash
python -m ncm.experiment --config examples/text/config.json --output runs/text-demo
python -m ncm.experiment --config examples/text/config.json --output runs/text-demo --resume
```

The included tiny text files are an overfitting fixture, not a language-quality
benchmark. For a meaningful experiment, replace the two file paths in a copied
configuration with separate training and held-out UTF-8 text corpora. Paths are
relative to the current working directory. Byte IDs form a fixed 256-symbol
vocabulary; consecutive non-overlapping chunks are padded to the finest grid.
No byte fallback bypasses quantization. Splits must be prepared before running;
keep duplicates and overlapping documents out of held-out data.

This experiment optimizes tokenizer reconstruction with commitment and entropy
regularization; prior loss has weight zero. It does not evaluate generation.
Each checkpoint interval evaluates every valid byte of both datasets, without
dropout, reporting byte cross-entropy in nats and reconstruction accuracy.
Evaluation covers reconstruction of the input, not next-token prediction.

The output directory contains the configuration and input SHA-256 hashes,
append-only `metrics.jsonl`, and an atomically replaced `checkpoint.msgpack`
containing parameters, optimizer moments, EMA codebooks and step. Random batches
and dropout keys are derived from seed and step, so resume restores the same
trajectory on the same software/hardware. Resume requires the same configuration
and corpus hashes; `steps` is the total target step count. An interrupted run
restarts from its most recent checkpoint. Metrics can repeat a step after a
replayed checkpoint; select the latest record per step. Use a fresh directory
for a changed configuration. Checkpoints are intended for the same code and
dependency versions; cross-version compatibility is not guaranteed.

The fixture reached 100% training-byte reconstruction by step 100 in a 500-step
CPU run. Held-out cross-entropy worsened after that point, demonstrating
overfitting. Stable held-out quality on a larger corpus remains an experimental
acceptance gate. The test suite checks exact final-state and metric equivalence
between an uninterrupted run and a run resumed from its midpoint.

## Frozen-codec prior and sampling

After training a codec with the reconstruction command above:

```bash
python -m ncm.prior_experiment --config examples/text/prior.json --output runs/prior-demo
python -m ncm.prior_experiment --config examples/text/prior.json --output runs/prior-demo --resume
python -m ncm.sample --run runs/prior-demo --count 16 --seed 42 --temperature 1.0 --top-k 0
```

`codec_run` points to the reconstruction run. The prior optimizer contains only
prior parameters: encoder, decoder, projections and EMA codebooks remain fixed.
Encoding runs with dropout disabled. Prior checkpoints contain prior weights,
optimizer state and step; retain the source codec run as well. Its checkpoint
hash is recorded and verified on resume and sampling. As in the codec experiment,
randomness is step-addressed and a changed run configuration is rejected.

The prior evaluates each level under two contexts:

- `teacher_code_ce` / `teacher_code_accuracy`: true earlier codes.
- `free_code_ce` / `free_code_accuracy`: greedy earlier predictions.
- `ce_degradation`: free-prefix CE minus teacher-forced CE, per level, in nats.
- `codec_reconstruction_accuracy`: decoding true codes, an oracle reconstruction
  reference; this does not measure prior predictions.
- `free_reference_byte_accuracy`: greedy generated bytes compared with the
  paired input, using its length. This is an exposure diagnostic, not a quality
  score for unconditional generation. A valid alternative sample can mismatch.

Code counts are aggregated over the corpus before normalization. Free-prefix CE
is not a marginal sequence likelihood. First-level CE must agree between paths,
since neither has an earlier-code context. Evaluation geometry includes reference
lengths; standalone sampling instead accepts a caller-selected `--length` within
the trained bucket, defaulting to its maximum. There is no prompt conditioning,
learned length distribution, EOS mechanism, or fallback side channel in these
byte experiments. This milestone supplies unconditional fixed-length generation.

Sampling draws categorical codes at each level with an explicit seed, optional
top-k filtering (`0` means all codes), and temperature (`0` means greedy).
The frozen decoder selects bytes by argmax. JSON includes raw byte arrays and
codes; padded positions have ID -1. Display text replaces invalid UTF-8, so raw
bytes are authoritative. `samples.json` records samples after training; the
standalone sampling command prints JSON for any requested sampling settings.

For programmatic use, `ncm.generation.generate(model, codec_params, prior_params,
codebooks, key, batch_size=..., length=..., temperature=..., top_k=...)` returns
byte IDs and hierarchical codes. `bytes_to_text` provides display strings.
Sampling settings are Python values (static when wrapping generation in JIT).

On the deliberately tiny fixture, a 500-step prior trained against the 500-step
codec produced 12 distinct outputs from 16 draws, including fragments such as
` cat sat`. Later-level training code accuracy reached 100% with true prefixes,
while greedy-prefix code accuracy was about 38% and 42%. Held-out greedy-prefix
CE increased by about 6.00 and 5.88 nats at those levels. These are diagnostics
from a tiny overfit experiment, not evidence of useful language quality. A larger
held-out corpus and quality evaluation remain necessary before performance
comparisons. Tests cover seeded sampling, padding, frozen codec state, empty
batches, first-level metric agreement and exact prior checkpoint-resume parity.

## Inference benchmark (milestone 3)

```bash
python -m ncm.benchmark --config examples/text/benchmark.json --output runs/benchmark-demo --require-backend cpu
```

Set `prior_run` to a completed frozen-codec prior run. The benchmark trains and
checkpoints a small causal byte Transformer on the same corpus and chunking,
then compares three complete byte-generation paths: padded NCM, NCM using actual
level lengths with sparse resampling, and autoregressive generation with a KV
cache. The NCM paths reuse identical trained weights. Greedy bytes and codes are
checked for equality before timing. The optimized path is also available as
`ncm.fast_generation.generate_fast` with the same arguments as `generate`.
It remains opt-in; smaller shapes or another accelerator may favor the padded
scan. Sampling uses a different random stream, so same-seed stochastic outputs
need not match across implementations.

`benchmark.json` controls baseline training steps, batch sizes, warmups,
repetitions, sample count and temperature. `ar_width`, `ar_heads` and `ar_layers`
optionally override baseline dimensions. Longer `synthetic_lengths` create
untrained models for shape/performance checks; these results are separate from
the trained fixture and cannot support language-quality claims. The benchmark
uses one device (device zero), float32 and one process, including on a TPU host.
It does not benchmark multi-device scaling or training throughput.

The output directory contains:

- `report.json`: backend/device/software versions, source and checkpoint hashes,
  parameter sizes, raw timing observations, compilation times, compiler memory
  estimates, sample diagnostics and stated limitations.
- `report.md`: a readable latency/throughput table.
- `ar_checkpoint.msgpack`: baseline parameters, optimizer state and step.

Compilation and lowering are timed separately from warm inference. Timed inputs
are on-device; each run waits for completion with `jax.block_until_ready`, so
asynchronous dispatch cannot masquerade as completed inference. Throughput is
**bytes/second**, not tokenizer-dependent tokens/second. All paths return bytes;
NCM timing includes the decoder and AR timing includes cache initialization and
all serial steps. Host text conversion, data loading and checkpoint I/O are
outside inference timing. Compilation times are observed within the process,
not guaranteed cold-cache timings. Avoid other workloads during measurement.

Common sample diagnostics include byte-bigram total variation against held-out
chunks, distinctness and valid UTF-8 fraction. These are descriptive, not semantic
quality metrics. AR held-out next-byte CE is reported separately from NCM code
CE; these objectives cannot be compared as perplexities. The baseline and NCM
are not automatically matched for parameter count, training compute or quality.
Inspect those differences before interpreting latency ratios as speedups.

Compiler live-buffer estimates include arguments, outputs and temporaries minus
aliases. They are **not measured peak device memory**. Available device memory
counters are recorded as process-lifetime observations, not per-case peaks; an
unsupported measurement is null. Add `--trace` for separate profiling executions
outside timed repeats. Trace capture errors are recorded in the report.

No TPU or TPU VM was available for this milestone. CPU timings do not establish
TPU performance. On an existing TPU machine, install a compatible TPU-enabled
JAX within this project's dependency range following the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
copy the source codec/prior runs and corpora preserving configuration paths, and
run:

```bash
python -m ncm.benchmark --config examples/text/benchmark.json --output runs/tpu-benchmark --require-backend tpu --trace
```

The backend guard fails before training or benchmarking if TPU is unavailable.
Before a publishable TPU comparison, additionally measure isolated per-case
peak device memory, select representative trained sequence lengths, match quality
and resource budgets, and repeat across seeds and runs. The harness follows the
[JAX timing guidance](https://docs.jax.dev/en/latest/benchmarking.html) and uses
[JAX profiling](https://docs.jax.dev/en/latest/profiling.html) for optional traces.

## Minimal SCALE mechanism pilot (milestone 4)

```bash
python -m ncm.scale_experiment --config examples/scale/config.json --output runs/scale-pilot
python -m ncm.scale_experiment --config examples/scale/config.json --output runs/scale-pilot --resume
```

This is a CPU-sized, one-teacher test of representation alignment and
hierarchy dependence. A tiny causal Transformer learns a synthetic distribution
over four A/B bytes. Its output law is explicitly normalized over A/B at each
prefix. The teacher is then frozen. Independent random draws form training,
validation and representation-calibration samples; they share the same finite
support, so this is not a held-out-domain language experiment. Calibration fits
one PCA feature interface using teacher causal features before the byte head.
It provides empirical representation targets, not an exact operator-transfer
certificate. No external pretrained LLM is downloaded or converted.

For each student seed, three arms start from identical codec parameters and
codebooks and use the same minibatches, architecture, nominal code capacity,
optimizer settings and update counts:

- `reconstruction`: reconstruction, commitment and utilization objectives.
- `alignment`: the same objectives plus teacher representation alignment.
- `alignment_dependency`: alignment plus a soft conditional-dependence penalty.

Every arm then freezes its complete codec and trains the same-size level prior.
The dependence term enumerates the mixture of soft code assignments, conditional
on preceding levels. Its gradient can help design a hierarchy, but it can also
reward information loss. Evaluation therefore measures **hard-code** conditional
total correlation and requires reconstruction fidelity. It verifies the entropy
identity against a separately implemented factorized-prior KL calculation.
Tests calibrate these calculations on independent bits, a shared bit, and a
shared bit explained by a preceding level.

Both code and output domains are small enough for exact enumeration. The pilot
sums probabilities of every hierarchical codeword through the frozen deterministic
byte decoder, retaining an additional bin for out-of-domain outputs. The primary
output error is exact total variation against the teacher distribution, not
sample distinctness or teacher-forced code accuracy. Reports also include
Jensen–Shannon divergence and forward KL. If the student assigns zero mass to
positive teacher support, exact forward KL is infinite: JSON records it as null
alongside the missing teacher mass, rather than hiding the failure by smoothing.
Enumeration is capped at 4096 codewords and is unsuitable for real text scales.

`protocol.json` freezes configuration, gates, implementation hashes and software
versions before training. The default three-seed pilot requires all of:

- At least 95% population byte reconstruction in each dependence-aware run.
- At most a one-percentage-point reconstruction drop versus alignment alone.
- Mean paired output-TV improvement of at least 0.02 versus alignment alone.
- Mean paired hard conditional-TC reduction of at least 0.01 nats.
- Identical compiler-estimated student step FLOPs across arms and seeds.

The weights and gates are fixed before the experiment, not selected from its
population evaluation. This local protocol is not an external preregistration.
The baseline also computes alignment and dependence with dynamic zero
coefficients, preserving the same student operation graph. Teacher training,
feature extraction and calibration are shared setup costs charged equally to
all arms. This matches student updates and estimated operations; it does not
establish reduced total training compute or hardware-time equivalence.

The output directory stores the teacher checkpoint and enumerable distribution,
canonical map, sample split indices, codec/prior checkpoints for each seed and
arm, and `report.json` / `report.md` with the fixed-gate decision. Checkpoints use
step-addressed minibatch randomness. Resume restores incomplete stages and skips
completed arms; configuration or implementation changes require a new directory.
Tests check exact checkpoint replay, protocol protection, frozen-codec training,
output probability normalization and the independent TC identity.

This implements a minimal SCALE mechanism test. Cross-size teacher transfer,
selective operator transplantation, realistic teacher-logit distillation,
prompt-conditioned language quality and TPU validation remain future work.
The full compiler described in the proposal is not implemented by this pilot.
The dependence rationale follows the existing
[parallel-sampling analysis](https://arxiv.org/abs/2608.25505), while the
codec/prior staging follows [NCM](https://arxiv.org/abs/2610.14162).

The default three-seed CPU pilot passed its fixed gates. Mean exact output TV
was 0.332 for reconstruction, 0.334 for alignment alone and 0.174 for alignment
plus dependence. Mean hard conditional TC fell from 0.455 to 0.137 nats versus
alignment alone; mean byte reconstruction was 99.24% in the dependence-aware arm.
All arms used the same estimated student step FLOPs. A same-seed verification
run reproduced checkpoints and numerical results exactly; it is not an
additional independent trial.

These results support the dependence mechanism in this small setting. Alignment
alone did not improve output error, so teacher-transfer benefit is not established.
About 91% of the summed paired TV gain came from one seed. Another dependence-aware
run could not generate outcomes covering 5.12% of teacher probability, giving
infinite forward KL despite passing the byte-reconstruction gate. A stronger
follow-up should add a dependence-only control, address rare-output coverage,
and use more seeds and independent domains before testing larger teachers.

## Four-arm and rare-output coverage follow-up

```bash
python -m ncm.scale_experiment --config examples/scale/followup.json --output runs/scale-followup
python -m ncm.scale_experiment --config examples/scale/followup.json --output runs/scale-followup --resume
```

This separate protocol adds the missing `dependency` control: a dependence penalty
without teacher-feature alignment. The four arms form a two-factor comparison of
alignment and dependence. All receive the same 400 codec updates, 400 prior
updates and 200 additional decoder-replay updates. Seeds 0–2 replicate previously
observed runs; **only new student seeds 3–7 determine follow-up gates**. The
teacher and synthetic domain are unchanged, so these are new initializations,
not a new-domain or cross-teacher confirmation.

The earlier seed-0 coverage failure had 16 distinct input encodings, but its
decoder reconstructed several rare inputs incorrectly. The follow-up tests
uniform replay over distinct **observed training sequences**, updating only the
decoder. It freezes encoder parameters, codebook state and prior parameters.
Checks enforce unchanged code assignments and hard-code dependence. Training
and evaluation distributions are not mixed into the repair loss; no probability
smoothing, fallback routing or external lookup table supplies missing outputs.
A decoder-only update cannot recover distinctions already lost by the encoder.

The protocol separates two questions:

- Coverage: on each new combined-arm seed, repaired output support must cover all
  positive teacher mass, sequence reconstruction must reach at least 98%, and
  exact output TV may increase by at most 0.01 versus before repair.
- Alignment contribution: in addition to coverage, alignment plus dependence must
  improve mean output TV over dependence alone by at least 0.02 and win on at
  least four of five new seeds. Every comparison uses equally repaired models.

The original pilot gates are still reported separately; their success does not
imply that either stronger follow-up gate passed. All student training and repair
step FLOP estimates must be available and identical for a matched-budget claim.
The output records metrics before and after repair and every paired new-seed
comparison, including negative results.

Each experiment now archives its exact package sources, project configuration
and input configuration under `source_snapshot/`, alongside protocol hashes.
`decoder_repair.msgpack` contains replay optimizer state for resume.
`codec_repaired.msgpack` is a frozen inference export containing only parameters
and codebooks; it does not carry obsolete joint-training optimizer moments.
The original `codec.msgpack` and `prior.msgpack` remain the pre-repair checkpoints.
Checkpoint hashes canonicalize dictionary order so template reloads preserve
content identity. Tests cover the four-arm coefficients, decoder learning,
unchanged frozen components and exact mid-replay resume.

### Balanced-codec follow-up after collision diagnosis

```bash
python -m ncm.scale_experiment --config examples/scale/balanced_followup.json --output runs/balanced-codec-followup
```

The decoder-replay study failed its stronger gates. It fixed the original
missing-output case, but new seed 4 still missed 6.63% of teacher probability.
Its encoder assigned the 16 input sequences to only 10 distinct codewords. Even
an ideal deterministic decoder could reconstruct at most 89.80% of teacher
sequence mass from those codes, below the fixed 98% threshold. The four-arm
comparison also gave alignment wins on only three of five new seeds. These
negative results are retained; more decoder updates cannot restore distinctions
that share the same discrete encoding.

The balanced-codec protocol changes the codec sampler to uniform over distinct
observed training sequences. Reconstruction and feature alignment therefore
emphasize rare training observations. Soft dependence uses self-normalized
importance weights derived from training counts, retaining the empirical
teacher-frequency target instead of confusing uniform sampling with a new
teacher distribution. This is a finite-minibatch surrogate, not an unbiased
exact TC estimator. Codebooks learn from the balanced codec batches; the prior
still trains on the original empirical teacher observations.

Architecture, coefficients, update counts and follow-up gates stay fixed.
Known cases 0, 4 and 7 are replications; new student seeds 8–12 determine the
new protocol's gates. All four arms receive the same sampler and decoder replay.
Each result now includes unique population encodings, collision groups and the
maximum sequence reconstruction possible from its deterministic code mapping.
No validation examples or population teacher probabilities select training
examples, importance weights or repair targets.

The balanced-codec study passed its coverage gate on new seeds 8–12: every
combined-arm run retained 16 encodings, reconstructed 100% of sequences and
covered all positive teacher output mass. Coverage was already restored before
extra decoder replay. The known failing seed 4 also recovered to 100% sequence
reconstruction with no missing output support.

The stronger alignment claim still failed: alignment beat dependence alone on
only three of five new seeds. Mean output TV was 0.426 for reconstruction, 0.237
for alignment, 0.523 for dependence only and 0.247 for the combination. Thus the
original dependence benefit did not hold uniformly across sampler choices and
seeds. Post-run analysis associated the worst generation with collapsed coarse
codebooks, despite retained reconstruction information. A controlled codebook
initialization/usage study is the next target; larger-teacher transfer remains
unproven. Both failed and passing coverage protocols and their source snapshots
are retained under their separate run directories.

## Coarse-codebook stabilization

The new maintained preset uses deterministic K-means++ on initial encoder
outputs, independent hard-usage EMA tracking and seeded dead-code replacement.
It adds optional coarse marginal entropy, temperature annealing, commitment
warmup and a separate EMA-centroid optimizer group. All buffers and RNG/schedule
counters are checkpointed; historical disabled configurations keep their schema.

```bash
python -m ncm.scale_experiment --config examples/scale/coarse_stability.json --output runs/coarse-stability
```

All 20 new-seed models across the four arms passed the coarse-stability gate:
both codes active, minimum population perplexity 1.973/2, zero post-annealing
collapsed steps and preserved output support. The broader teacher-transfer gate
still failed. See [configuration, optimizer semantics and resume details](docs/coarse-codebooks.md).

Compact [recorded experiment reports](reports/README.md), including negative
follow-up results and the coarse-codebook validation record, are versioned in
`reports/`. Checkpoints, source snapshots and profiler traces remain local
under the ignored `runs/` directory.
