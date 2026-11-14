# Causal-AR evaluation and drafter comparison

Both commands are native Rust runners. They use the JAX engine through the typed JSON-RPC methods in the [Rust/JAX protocol guide](rust-jax.md). Rust prepares batches and prompts, computes summary statistics, and writes the reports.

## Causal-AR evaluation

The decisive runner can optionally train one causal AR model, then evaluate held-out byte sequences in batches. Plain-text files are split into fixed-width byte rows; document JSONL files preserve document boundaries and carry padding masks. The report includes token count, sequence count, mean NLL, population variance, perplexity, and sequence NLL values.

```sh
cargo run --bin ncm-run -- run-decisive-benchmark \
  --model-config examples/text/ar_config.json \
  --train examples/text/train.txt \
  --heldout examples/text/validation.txt \
  --steps 10 --batch-size 2 --prompt-len 1 --seed 7 \
  --output runs/decisive-ar
```

`--train` is required when `--steps` is greater than zero. Outputs are `results.json` and `summary.md`.

## Speculative drafter comparison

The runner initializes one target and one or more causal AR drafters. For every prompt it records direct target generation, then repeatedly asks a drafter for proposal tokens and probabilities and calls `speculative_verify` on the target. Rust measures acceptance, rounds, throughput, latency, analytic FLOPs, and paired throughput differences versus direct target generation.

```sh
cargo run --bin ncm-run -- run-drafter-comparison \
  --target-config examples/text/ar_config.json \
  --drafter small-ar=examples/text/ar_config.json \
  --prompts examples/text/prompts.txt \
  --max-new-tokens 8 --draft-len 2 --temperature 1 --top-k 0 --seed 7 \
  --output runs/drafter-ar
```

Prompts may be newline-delimited text or JSONL records with `prompt` and optional `id` fields. Output files are `results.json`, `drafter_comparison.csv`, `drafter_summary.csv`, `paired_statistics.json`, and `summary.md`.

The FLOP values are analytical estimates based on configured AR shapes; they are not compiler or hardware counters. Timing is a single run and may include first-use compilation. BPB is omitted because this prompt comparison has no held-out continuation corpus. For research claims, use repeated, warmed measurements and report the full protocol and model/corpus provenance.

These native commands implement the current AR evaluation and speculative comparison interfaces. They do not reproduce the historical multi-arm NCM-versus-AR training sweep or its decoder quality measurements. Historical paper results remain documented in the archived reports and manuscripts.
