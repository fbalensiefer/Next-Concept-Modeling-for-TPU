"""End-to-end functional training: parameters, optimizer and EMA travel together."""
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn, struct

from .config import LossConfig, RVQConfig, TransformerConfig
from .coarse import ensure_initialized, maintain_coarse, optimizer_state, schedules, warmup_codebooks
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
    def __call__(self, batch: Batch, codebooks, phase=0, *, training=False,
                 coarse_beta=None, coarse_temperature=None, return_latent=False):
        tokenizer = MultiscaleTokenizer(self.rvq, self.token_vocab_size)(
            batch.tokens, batch.mask, batch.fallback_bytes, batch.fallback_mask,
            codebooks, phase=phase, training=training, coarse_temperature=coarse_temperature)
        prior = SharedLevelGenerator(self.rvq, self.transformer)(
            codebooks, tokenizer.rvq.mask, tokenizer.rvq.indices,
            teacher_forcing=True, deterministic=not training,
            source_mask=batch.mask & ~batch.fallback_mask)
        loss, metrics = combined_loss(
            tokenizer.logits, batch.tokens, batch.mask, tokenizer.rvq,
            prior.logits, self.losses, coarse=tokenizer.coarse,
            projected_fine=tokenizer.projected_fine, concept_mask=tokenizer.concept_mask,
            coarse_beta=coarse_beta, coarse_entropy_weight=self.rvq.coarse.entropy_weight if self.rvq.coarse.enabled else 0.)
        auxiliary = (tokenizer.rvq, tokenizer.latent) if return_latent else tokenizer.rvq
        return loss, (metrics, auxiliary)


def initialize(model: NCM, optimizer: optax.GradientTransformation, key, batch: Batch, *,
               warmup_batches=None, initialize_codebooks=True):
    key_codebook, key_params, key_dropout = jax.random.split(key, 3)
    codebooks = init_codebooks(key_codebook, model.rvq)
    params = model.init({"params": key_params, "dropout": key_dropout},
                        batch, codebooks, training=True)["params"]
    opt_state = optimizer_state(optimizer.init(params), codebooks, model.rvq, key_codebook)
    if model.rvq.coarse.enabled and initialize_codebooks:
        tokenizer = MultiscaleTokenizer(model.rvq, model.token_vocab_size)
        codebooks, coarse = warmup_codebooks(tokenizer, params['MultiscaleTokenizer_0'], codebooks,
                                            opt_state['coarse'], [batch] if warmup_batches is None else warmup_batches)
        opt_state = dict(opt_state, coarse=coarse)
    return TrainState(params, opt_state, codebooks, jnp.asarray(0, jnp.int32))


def make_train_step(model: NCM, optimizer: optax.GradientTransformation):
    """Return an uncompiled global-array step; use sharding.compile_train_step.

    The phase is a dynamic scalar integer, never a static_argnum. Codebooks
    used to create targets are the old state; EMA is applied once afterward.
    All-padding global batches skip *both* optimizer and EMA updates.
    """
    def step(state: TrainState, batch: Batch, phase, key):
        original = state
        temperature = beta = None
        if model.rvq.coarse.enabled:
            tokenizer = MultiscaleTokenizer(model.rvq, model.token_vocab_size)
            books, coarse = ensure_initialized(tokenizer, state.params['MultiscaleTokenizer_0'],
                                               state.codebooks, state.opt_state['coarse'], batch)
            state = state.replace(codebooks=books, opt_state=dict(state.opt_state, coarse=coarse))
            temperature, beta = schedules(model.rvq.coarse, coarse)
        def objective(params):
            return model.apply({"params": params}, batch, state.codebooks, phase,
                               training=True, rngs={"dropout": key}, coarse_beta=beta,
                               coarse_temperature=temperature, return_latent=model.rvq.coarse.enabled)
        (loss, (metrics, output)), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        neural_state = state.opt_state['neural'] if model.rvq.coarse.enabled else state.opt_state
        updates, neural_state = optimizer.update(grads, neural_state, state.params)
        params = optax.apply_updates(state.params, updates)
        if model.rvq.coarse.enabled:
            output, latent = output
            codebooks, coarse, diagnostics = maintain_coarse(
                state.codebooks, state.opt_state['coarse'], output.counts, output.sums,
                latent, batch.mask & ~batch.fallback_mask, model.rvq, phase=phase)
            opt_state = {'neural': neural_state, 'coarse': coarse}
            metrics = dict(metrics, **diagnostics, coarse_temperature=temperature, coarse_beta=beta)
        else:
            opt_state = neural_state
            codebooks = ema_update(state.codebooks, output.counts, output.sums, model.rvq, phase=phase)
        candidate = TrainState(params, opt_state, codebooks, state.step + 1)
        new_state = jax.lax.cond(jnp.any(batch.mask), lambda: candidate, lambda: original)
        metrics = dict(metrics, grad_norm=optax.tree.norm(grads))
        return new_state, metrics
    return step
