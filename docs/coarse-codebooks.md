# Coarse-codebook stabilization

The maintained codec initializes its first RVQ level from encoder activations,
tracks hard assignments independently of centroid learning, and recovers dead
codes using current training activations. Fine codebooks retain the existing EMA
algorithm. Neural parameters keep their existing optimizer.

## Enable and run

```bash
python -m ncm.scale_experiment --config examples/scale/coarse_stability.json --output runs/coarse-stability
python -m ncm.scale_experiment --config examples/scale/coarse_stability.json --output runs/coarse-stability --resume
python -m pytest -q
```

The preset applies the same maintenance configuration to `reconstruction`,
`alignment`, `dependency`, and `alignment_dependency`. It uses known-seed checks
4, 8, and 10 and new confirmation seeds 13–17. It does not overwrite earlier
experiments or their thresholds.

For direct model construction:

```python
from ncm import CoarseConfig, RVQConfig

rvq = RVQConfig((1, 4), 2, 8, coarse=CoarseConfig(enabled=True))
```

Use `initialize` and `make_train_step` to route the two optimizer groups. An
optional `warmup_batches` list passed to `initialize` supplies initial training
batches; without it, direct initialization uses the supplied example batch.
The text and SCALE experiment runners collect their configured number of actual
initial training batches with the experiment's fixed batch-order RNG. Parameters
remain frozen during this prepass.

Historical configurations default to `coarse.enabled=false`, preserving their
optimizer structure. New maintained runs opt in explicitly. `experiment.py` and
`prior_experiment.py` also load the optional nested `coarse` configuration.

## Configuration

All keys below live under the experiment's `coarse` object:

| Setting | Default | Meaning |
|---|---:|---|
| `enabled` | false | Enable initialization, maintenance and schedules |
| `warmup_batches` | 2 | Frozen-encoder training batches for initialization |
| `kmeans_iterations` | 20 | Fixed Lloyd iteration count after seeded K-means++ |
| `initialization_noise` | 0.0001 | Small deterministic centroid jitter |
| `usage_decay` | 0.9 | Decay for hard cluster assignment counts |
| `dead_code_threshold` | 1.0 | Minimum EMA assignments per global batch |
| `dead_code_grace_steps` | 25 | Delay before checking for dead codes |
| `dead_code_check_every` | 1 | Replacement-check interval |
| `reset_count` | 2.0 | Reset pseudo-count, raised to at least twice the threshold |
| `replacement_noise` | 0.01 | Gaussian perturbation of a valid pooled encoder activation |
| `entropy_weight` | 0.05 | Additional coarse marginal soft-entropy coefficient; 0 disables it |
| `temp_start`, `temp_end` | 2.0, 0.25 | Coarse auxiliary posterior temperatures |
| `temp_anneal_steps` | 200 | Linear temperature interpolation duration |
| `beta_start`, `beta_end` | 0.0, 0.25 | Coarse commitment coefficients |
| `beta_warmup_steps` | 200 | Linear commitment warmup duration |
| `coarse_codebook_lr_scale` | 0.25 | Coarse centroid optimizer step relative to the ordinary EMA centroid proposal |

A zero schedule duration uses the end value immediately. Schedules use the saved
coarse training-step counter. Nearest-neighbor code selection stays hard at every
step; temperature smooths the coarse entropy/TC posteriors only. Fine-level
posterior temperatures and commitment coefficients are unchanged. No
Gumbel-Softmax sampling is introduced.

Hard-count thresholds are in global-batch assignment units, not fractions or
code probabilities. Choose thresholds appropriate to batch size, coarse grid
length and vocabulary size. Genuine constant/undersized activation data cannot
support arbitrary hard-code diversity; centroid jitter does not create missing
information. Population hard usage and reconstruction must still be checked.

## Initialization and recovery

1. Extract encoder outputs, excluding padding and byte-fallback positions.
2. Pool onto the same coarse grid used by RVQ.
3. Run seeded, masked K-means++ and a fixed number of Lloyd iterations.
4. Initialize coarse embeddings and their EMA counts/sums; leave fine levels
   unchanged. Record the initialized flag and valid sample count.
5. Track a separate hard-usage EMA. If a code falls below threshold after the
   grace interval, sample a valid coarse activation from the current batch, add
   seeded Gaussian noise, and reset its vector, EMA count/sum and usage count.

Candidate sampling and noise generation run every maintained step, including
when there are no dead codes. A mask selects replacements. This avoids an
arm-dependent replacement branch and preserves the common operation graph.
Completely padded batches skip all optimizer/state changes. Runtime phases that
freeze the coarse level also freeze its vectors, EMA statistics, usage and
replacement counts. The schedule clock continues with valid training steps.

Initialization on an all-padding/all-fallback warmup leaves the flag false. The
first later batch with learned-path activations initializes once before its
objective is evaluated. In SCALE runs, successful warmup is checkpointed at step
zero; resume loads the saved state before considering another warmup. Interrupted
warmup itself can be deterministically recomputed before any training update.

## Separate optimizer group

Codebooks are EMA-managed state rather than parameters differentiated through
hard assignments. The implementation therefore keeps that architecture and adds
a separate Optax `coarse_codebook` group for tracking EMA centroid proposals:

```text
proposal = EMA_sums / (EMA_counts + epsilon)
pseudo_gradient = old_coarse_embedding - proposal
new_coarse_embedding = old_coarse_embedding - lr_scale * pseudo_gradient
```

The coarse group uses stateless SGD, so replacement has no stale momentum to
reset. Fine books take the original full EMA proposal. Neural encoder/decoder
and fine-network parameters retain the supplied neural optimizer and learning
rate. `coarse_codebook_lr_scale=1` takes the full centroid proposal; `0.25` takes a
quarter of that movement. It is a scale on EMA centroid movement, **not** a
numerical multiplier on the neural Adam learning rate, whose gradient units
are different. Coarse EMA sums continue tracking actual assignment statistics,
so a slowed embedding may lag their normalized centroid intentionally.

## Persisted state and compatibility

`CodebookState` still has exactly `embeddings`, `counts`, and `sums`.
`TrainState` still has `params`, `opt_state`, `codebooks`, and `step`.
Maintained runs store:

```text
opt_state.neural                  existing neural optimizer state
opt_state.coarse.initialized      successful data-driven initialization flag
opt_state.coarse.initial_samples  valid pooled warmup sample count
opt_state.coarse.hard_counts      independent hard-usage EMA
opt_state.coarse.replacements     per-code lifetime replacement counts
opt_state.coarse.base_key         serializable uint32[2] RNG key
opt_state.coarse.step             schedule and replacement RNG counter
opt_state.coarse.optimizer_state  separate coarse centroid optimizer group
opt_state.coarse.monitored_steps  post-warmup hard-usage observations
opt_state.coarse.collapsed_steps  observations with fewer than V active codes
opt_state.coarse.min_hard_perplexity
```

All are functional Flax/Optax checkpoint state. There is no hidden module RNG,
Python random generator, or external annealing counter. Dictionary order is
canonicalized by the existing checkpoint writer. To construct a load template,
use `initialize(..., initialize_codebooks=False)`; restoration fills centroids,
flags, counters, keys and optimizer groups. Historical disabled-state checkpoints
still load; a real earlier 500-step codec checkpoint was checked as well.

Changing maintenance settings is a new experiment, not a bit-exact continuation
of an old configuration. The experiment protocol guard continues to enforce
that distinction. Exact continuation is checked on the same software/backend and
device layout; cross-hardware numerical identity is not claimed.

## Budget and validation

Initialization is shared per student seed, and its fixed encoder-batch and
K-means budgets are charged equally to all four arms. All arms receive identical
initial vectors, all scheduled losses and the same maintenance graph, with only
the original arm coefficients differing. Compiler-estimated step FLOPs must
match. The additional logic costs work relative to disabled maintenance; it is
not claimed to be free or faster.

The controlled run passed the fixed gate on all 20 new-seed models (five seeds
by four arms): both coarse codes were used, minimum population perplexity was
1.973/2, and no collapsed steps occurred in the 200 post-annealing observations
per model. Teacher output support was preserved. The combined arm's minimum
sequence reconstruction was 99.07%; the other arms reached 100% on these new
seeds. The teacher-alignment superiority gate still failed, so this is evidence
for coarse stability in the finite domain, not a general SCALE transfer result.
The package was tested together; this study does not attribute gains to any
single stabilization component.

Tests cover masked/degenerate K-means, no repeated initialization on resume,
dead-code recovery and hard perplexity, frozen-phase/empty-batch behavior,
coarse-only learning-rate and commitment changes, soft-temperature behavior,
canonical bit-exact checkpoint continuation, all four arms' FLOP parity, and
seeded replacements on a virtual four-device CPU mesh. TPU execution remains
unmeasured.

## Subsequent validation boundary

The stability package's original success does not establish general safety. The matched maintenance ablation found that unguarded EMA can collapse despite perfect reconstruction; recovery helped within that domain. Fixed binary coarse endpoints were competitive there but failed some subsequent source-law cases. Across 360 fresh-seed runs on six distributions, no condition passed every fidelity check. Balanced usage alone should not select a quantizer.

See `reports/maintenance-ablation/report.md` and `reports/fresh-domain-validation/report.md`. Their protocols, population versus sampled checks, corrected historical support gate, numerical continuation and timing limitations are retained. Production configuration defaults have not changed. The next research priority is reconstruction protection and dependency modeling, not a quantizer replacement.
