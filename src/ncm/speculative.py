"""Exact speculative sampling with explicit, normalized proposal probabilities.

Adapters are Flax structs with a static ``apply_fn`` and dynamic ``params``.
Draft apply_fn(params, buffer, prefix_length, draft_len, key) returns [D,V]
logits, conditional on any sampled latent auxiliary variables. Given those
variables the draft tokens MUST be independent. Teacher apply_fn(params,
buffer, prefix_length, draft_len) returns [D+1,V] next-token logits along the
proposed prefix, including the bonus distribution. Buffers have static capacity.
"""
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class DistributionAdapter:
    params: object
    apply_fn: object = struct.field(pytree_node=False)
    draft_fn: object = struct.field(pytree_node=False, default=None)


@struct.dataclass
class SpeculativeOutput:
    tokens: jax.Array
    acceptance_rate: jax.Array
    tokens_per_forward_pass: jax.Array
    teacher_forward_passes: jax.Array
    accepted_by_position: object = None
    proposed_by_position: object = None
    mean_prefix_length: object = None
    round_prefix_lengths: object = None


def speculative_generate(ncm_params, ar_params, prompt_ids, draft_len, max_new_tokens, key):
    """Single-sequence static-bucket sampler, safe under jit and vmap.

    Returns exactly max_new_tokens, without EOS stopping. The final block is
    truncated. Acceptance counts exclude draft slots beyond the output budget.
    """
    if prompt_ids.ndim != 1 or draft_len < 1 or max_new_tokens < 1:
        raise ValueError('expected a 1D prompt and positive static lengths')
    prompt_len = prompt_ids.shape[0]
    capacity = prompt_len + max_new_tokens + draft_len + 1
    buffer = jnp.zeros((capacity,), jnp.int32).at[:prompt_len].set(prompt_ids)
    end = prompt_len + max_new_tokens

    def condition(state):
        return state[1] < end

    def step(state):
        buffer, position, accepted_total, proposed_total, passes, rng, accepted_positions, proposed_positions, prefix_sum, prefix_trace = state
        rng, latent_key, token_key, accept_key, bonus_key = jax.random.split(rng, 5)
        if ncm_params.draft_fn is None:
            q = jax.nn.softmax(ncm_params.apply_fn(ncm_params.params, buffer, position, draft_len, latent_key), -1)
            draft = jax.random.categorical(token_key, jnp.log(q)).astype(jnp.int32)
        else:
            draft, logits = ncm_params.draft_fn(ncm_params.params, buffer, position, draft_len, token_key)
            q = jax.nn.softmax(logits, -1)
        if q.ndim != 2 or q.shape[0] != draft_len:
            raise ValueError('draft logits must have shape [draft_len,vocab]')
        proposed = jax.lax.dynamic_update_slice(buffer, draft, (position,))
        p = jax.nn.softmax(ar_params.apply_fn(ar_params.params, proposed, position, draft_len), -1)
        if p.shape != (draft_len + 1, q.shape[-1]):
            raise ValueError('teacher logits must have shape [draft_len+1,vocab]')
        rows = jnp.arange(draft_len)
        qx, px = q[rows, draft], p[rows, draft]
        accept = jax.random.uniform(accept_key, (draft_len,)) < jnp.minimum(1., px / qx)
        prefix = jnp.sum(jnp.cumprod(accept.astype(jnp.int32)))
        # At a rejection use (p-q)+, otherwise sample from the extra teacher row.
        index = jnp.minimum(prefix, draft_len - 1)
        residual = jnp.maximum(p[index] - q[index], 0)
        residual = residual / jnp.maximum(residual.sum(), jnp.finfo(p.dtype).tiny)
        bonus_p = jnp.where(prefix == draft_len, p[-1], residual)
        bonus = jax.random.categorical(bonus_key, jnp.log(bonus_p)).astype(jnp.int32)
        slots = jnp.arange(capacity)
        keep = (slots >= position) & (slots < position + prefix)
        updated = jnp.where(keep, proposed, buffer).at[position + prefix].set(bonus)
        remaining = end - position
        accepted_total += jnp.minimum(prefix, remaining)
        proposed_total += jnp.minimum(draft_len, remaining)
        return (updated, jnp.minimum(position + prefix + 1, end), accepted_total,
                proposed_total, passes + 1, rng,
                accepted_positions + ((jnp.arange(draft_len) < prefix) & (jnp.arange(draft_len) < remaining)),
                proposed_positions + (jnp.arange(draft_len) < remaining), prefix_sum + position,
                prefix_trace.at[passes].set(position))

    buffer, _, accepted, proposed, passes, _, accepted_positions, proposed_positions, prefix_sum, prefix_trace = jax.lax.while_loop(condition, step,
        (buffer, jnp.asarray(prompt_len, jnp.int32), jnp.asarray(0), jnp.asarray(0), jnp.asarray(0), key,
         jnp.zeros(draft_len,jnp.int32),jnp.zeros(draft_len,jnp.int32),jnp.asarray(0),
         jnp.full(max_new_tokens,-1,jnp.int32)))
    return SpeculativeOutput(buffer[prompt_len:end], accepted / jnp.maximum(proposed, 1),
                             max_new_tokens / passes, passes, accepted_positions, proposed_positions, prefix_sum / passes, prefix_trace)


def ar_adapter(model, params, *, draft=False):
    """Full-prefix teacher, or a true sequential KV-cached AR proposal.

    The optional draft_fn returns samples AND the logits that sampled them.
    q_t may depend on earlier draft tokens, but never later ones. Existing
    independent-logit adapters retain their original RNG and sampling behavior.
    Prefix caches are rebuilt each round; no rollback/reuse claim is made.
    """
    def forward(params, buffer, position, length):
        inputs=jnp.concatenate([jnp.array([256],jnp.int32),buffer[:-1]])[None]
        valid=jnp.arange(len(buffer))[None] <= position+length
        logits=model.apply({'params':params},inputs,valid)[0]
        return jax.lax.dynamic_slice(logits,(position,0),(length+1,256))
    def propose(params,buffer,position,length,key):
        shape=(1,model.max_length,model.config.heads,model.config.width//model.config.heads)
        cache=tuple((jnp.zeros(shape),jnp.zeros(shape)) for _ in range(model.config.layers))
        def prefill(i,cache):
            token=jnp.where(i==0,256,buffer[jnp.maximum(i-1,0)])[None]
            _,cache=model.apply({'params':params},token,i,cache,method=model.decode_step)
            return cache
        cache=jax.lax.fori_loop(0,position,prefill,cache)
        previous=jnp.where(position==0,256,buffer[jnp.maximum(position-1,0)])
        def step(carry,i):
            previous,cache=carry
            logits,cache=model.apply({'params':params},previous[None],position+i,cache,method=model.decode_step)
            logits=logits[0]
            token=jax.random.categorical(jax.random.fold_in(key,i),logits).astype(jnp.int32)
            return (token,cache),(token,logits)
        _,(tokens,logits)=jax.lax.scan(step,(previous,cache),jnp.arange(length))
        return tokens,logits
    return DistributionAdapter(params,forward,propose if draft else None)
