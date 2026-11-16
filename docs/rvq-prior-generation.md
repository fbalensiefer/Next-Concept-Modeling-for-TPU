# Causal RVQ prior and generation pilot

The text generation path has two separately trained stages:

1. **Phase 1 codec:** a multiscale RVQ encoder/decoder maps byte windows to discrete indices and reconstructs bytes. The reconstruction-only checkpoint has its prior loss disabled.
2. **Phase 2 prior:** `src/ncm/prior_training.py` strictly restores that codec, keeps its encoder, decoder and codebooks frozen, and trains a causal categorical Transformer on valid RVQ indices in level-major order. The prior predicts one code index at each autoregressive position.

The shared-level NCM predictor remains a different model: it predicts positions in a codebook level in parallel. The categorical Phase 2 model is sequential over the serialized index streams, which supplies the causal ordering needed for sampling. Neither model's architecture implies useful text generation by itself.

## Phase 1 checkpoint gate

Run the reconstruction screening utility in the project Python environment:

```sh
source .venv311/bin/activate
python -m ncm.checkpoint_selection
```

The utility checks checkpoint structure and tensor shapes with the strict worker loader, then ranks candidates with matching per-length reconstruction results. It flags stage perplexity below 2.0 as possible collapse and checks that longer-window byte error does not regress by more than 0.05. Its score weights are 0.55 for inverse byte error, 0.25 for scaling stability and 0.20 for normalized codebook utilization. Loss is unavailable in the reconstruction report and is excluded.

The committed screening report selects `runs/real-text-pilot/length-512-seed-0/checkpoint.msgpack` (seed 0). Its byte error rate is `0.16011568099171897`, exact-window match is `0.0`, stage perplexities are `[9.906031520677761, 15.14685696072266, 14.976882163869677]`, normalized utilization is `0.8339535551097937`, and combined score is `0.8787270864765133`. The strict loader accepted the checkpoint with codec SHA-256 `679b8c6d85a8e2db5cf42204d3608c006f0a15d0b911fd99f6ff1fe51d93a82d`.

This is a codec reconstruction screen, not a generation-quality gate. `generation_eligible` remains false in that report because it contains no prior-quality evaluation. Pair the selected codec with a separately trained prior; assess quality using held-out generation experiments.

## Phase 2 prior training

The reproducible three-step pilot used the Phase 1 training split, seed 0, batch size 1, learning rate `0.003`, codebook size 16, and level lengths `[128, 256, 512]`. It froze the codec and used `prior_loss_weight=1.0`. The level-major prior sequence has `128 + 256 + 512 = 896` positions. The final recorded values are:

| Metric | Value |
|---|---:|
| Step | 3 |
| Prior cross-entropy | `2.9945807456970215` |
| Reconstruction loss | `0.5021113753318787` |
| Weighted commitment loss | `0.8349403738975525` |
| Total loss | `4.331632614135742` |
| Gradient norm | `0.9893022775650024` |

The total is reconstruction loss plus weighted commitment loss plus weighted prior loss. These metrics describe the final minibatch and do not establish convergence or held-out prior quality.

To reproduce the run with the recorded configuration:

```sh
source .venv311/bin/activate
python -m ncm.prior_training \
  --config reports/generation-training/checkpoints/training_config.json \
  --output reports/generation-training/checkpoints
```

The run writes `prior_checkpoint.msgpack` and `prior_metrics.json`. The binary checkpoint is ignored by Git; the JSON training config and metrics are tracked. Inputs under `runs/` are local experiment data and checkpoints and are not included in a fresh clone.

## Rust generation CLI

Use the selected codec, a trained categorical-prior checkpoint, and the Phase 1 JSON config:

```sh
source .venv311/bin/activate
export NCM_PYTHON="$PWD/.venv311/bin/python"
cargo run --release --bin ncm-run -- generate-pilot \
  --mode rvq-prior \
  --config reports/generation-training/checkpoints/training_config.json \
  --codec-checkpoint runs/real-text-pilot/length-512-seed-0/checkpoint.msgpack \
  --prior-checkpoint reports/generation-training/checkpoints/prior_checkpoint.msgpack \
  --prompt "Evaluation prompt sequence for autoregressive generation" \
  --max-length 32 --temperature 0.8 --top-k 50 --top-p 0.95 --seed 42 \
  --output reports/generation-pilot
```

`--max-length` is the number of continuation bytes; prompt and continuation must fit the codec's byte window. Prompt bytes are encoded to RVQ indices and supplied as known code positions while the remaining level-major positions are sampled. The returned byte stream retains the literal prompt prefix; its continuation comes from the codec decoder. `--output` defaults to `reports/generation-pilot`. The existing byte-level AR mode remains available with `--mode byte-ar`, `--model-config` and `--checkpoint`.

The pilot used temperature `0.800000011920929`, top-k 50, top-p `0.949999988079071`, and seed 42. Since the RVQ vocabulary has 16 entries, top-k 50 retains all code categories; nucleus filtering still applies. It sampled 896 RVQ codes, preserving the 56-byte prompt and returning 32 continuation bytes (88 bytes total). The observed mean code log probability was `-2.5274094066449573`, mean entropy was `2.4040825506672263` nats, and measured prior-plus-decoder time was `0.6235438750009052` seconds (`0.31300058303168043` prior and `0.3105432919692248` decode). Log probabilities are from the raw prior distribution; entropy is measured after sampling filters. The timing is a single first-call measurement that includes prior JIT compilation, not steady-state latency. These are one-run telemetry values, not quality statistics.

The recorded decoded string (JSON-escaped) was:

```text
Evaluation prompt sequence for autoregressive generationwowo asim nosl \n\nolunrlSlltoi\non
```

The generated sample is stored in `reports/generation-pilot/generated_samples.json`; run settings and telemetry are in `reports/generation-pilot/generation_metrics.json`. The short output is not fluent, and this three-step, single-prompt run is strictly an end-to-end pipeline and finite-metric verification pilot—not a generation-quality benchmark. No held-out prompts, multiple seeds, uncertainty intervals, or quality-matched baselines were evaluated.
