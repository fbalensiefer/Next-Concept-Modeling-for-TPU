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
