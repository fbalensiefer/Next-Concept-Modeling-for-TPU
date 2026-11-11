"""End-to-end functional training: parameters, optimizer and EMA travel together."""
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn, struct

from .config import LossConfig, RVQConfig, TransformerConfig
from .generator import SharedLevelGenerator
from .losses import combined_loss
from .rvq import CodebookState, ema_update, init_codebooks
from .tokenizer import MultiscaleTokenizer


@struct.dataclass
class Batch:
    tokens: jax.Array  # int32 [B,L], vocabulary IDs
    mask: jax.Array  # bool [B,L], True = valid
    fallback_bytes: jax.Array  # int32 [B,L], IDs in 0..255 at fallback slots
    fallback_mask: jax.Array  # bool [B,L]


@struct.dataclass
class TrainState:
    params: object
    opt_state: object
    codebooks: CodebookState
    step: jax.Array


class NCM(nn.Module):
    rvq: RVQConfig
    transformer: TransformerConfig
    token_vocab_size: int
    losses: LossConfig = LossConfig()

    @nn.compact
    def __call__(self, batch: Batch, codebooks, phase=0, *, training=False):
        tokenizer = MultiscaleTokenizer(self.rvq, self.token_vocab_size)(
            batch.tokens, batch.mask, batch.fallback_bytes, batch.fallback_mask,
            codebooks, phase=phase, training=training)
        prior = SharedLevelGenerator(self.rvq, self.transformer)(
            codebooks, tokenizer.rvq.mask, tokenizer.rvq.indices,
            teacher_forcing=True, deterministic=not training,
            source_mask=batch.mask & ~batch.fallback_mask)
        loss, metrics = combined_loss(
            tokenizer.logits, batch.tokens, batch.mask, tokenizer.rvq,
            prior.logits, self.losses, coarse=tokenizer.coarse,
            projected_fine=tokenizer.projected_fine, concept_mask=tokenizer.concept_mask)
        return loss, (metrics, tokenizer.rvq)


def initialize(model: NCM, optimizer: optax.GradientTransformation, key, batch: Batch):
    key_codebook, key_params, key_dropout = jax.random.split(key, 3)
    codebooks = init_codebooks(key_codebook, model.rvq)
    params = model.init({"params": key_params, "dropout": key_dropout},
                        batch, codebooks, training=True)["params"]
    return TrainState(params, optimizer.init(params), codebooks, jnp.asarray(0, jnp.int32))


def make_train_step(model: NCM, optimizer: optax.GradientTransformation):
    """Return an uncompiled global-array step; use sharding.compile_train_step.

    The phase is a dynamic scalar integer, never a static_argnum. Codebooks
    used to create targets are the old state; EMA is applied once afterward.
    All-padding global batches skip *both* optimizer and EMA updates.
    """
    def step(state: TrainState, batch: Batch, phase, key):
        def objective(params):
            return model.apply({"params": params}, batch, state.codebooks, phase,
                               training=True, rngs={"dropout": key})
        (loss, (metrics, output)), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        codebooks = ema_update(state.codebooks, output.counts, output.sums,
                               model.rvq, phase=phase)
        candidate = TrainState(params, opt_state, codebooks, state.step + 1)
        new_state = jax.lax.cond(jnp.any(batch.mask), lambda: candidate, lambda: state)
        metrics = dict(metrics, grad_norm=optax.tree.norm(grads))
        return new_state, metrics
    return step
