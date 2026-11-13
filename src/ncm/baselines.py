"""AR baseline using the same attention and MLP blocks as hybrid NCM."""
from .autoregressive import AutoregressiveBytes, CausalBlock, sample_ar, shifted_inputs

# Both arms share CausalBlock; cache inference uses the identical projections.
CausalAR = AutoregressiveBytes


def parameter_count(params):
    import jax
    return sum(x.size for x in jax.tree.leaves(params))


def matched_ar_config(target_parameters, *, heads=4, layers=2, max_length=512):
    """Select the nearest width (multiple of heads), including embeddings/head.

    Returns the actual mismatch; callers must report it rather than asserting
    exact equality when discrete Transformer widths cannot attain it.
    """
    from .config import TransformerConfig
    def count(w):
        # Per block: qkv/out, two MLP matrices, two norms and their biases.
        return 2 * 256 * w + w + max_length * w + layers * (12*w*w + 13*w) + 2*w + 256
    widths = range(heads, max(heads * 2, int(target_parameters**.5) + heads), heads)
    width = min(widths, key=lambda w: abs(count(w) - target_parameters))
    return TransformerConfig(width, heads, layers), count(width) - target_parameters


def sample_conditioned(model, params, prompt, length, key):
    """KV-cached continuation of a fixed-length, unpadded batch of prompts."""
    import jax
    import jax.numpy as jnp
    b, p = prompt.shape
    if p + length > model.max_length:
        raise ValueError('prompt and continuation exceed position bucket')
    shape = (b, model.max_length, model.config.heads, model.config.width // model.config.heads)
    cache = tuple((jnp.zeros(shape), jnp.zeros(shape)) for _ in range(model.config.layers))
    inputs = jnp.concatenate([jnp.full((b, 1), 256, jnp.int32), prompt], 1)
    def prefill(cache, i):
        _, cache = model.apply({'params': params}, inputs[:, i], i, cache, method=model.decode_step)
        return cache, None
    cache, _ = jax.lax.scan(prefill, cache, jnp.arange(p))
    previous = inputs[:, -1]
    def step(carry, i):
        previous, cache = carry
        logits, cache = model.apply({'params': params}, previous, p + i, cache, method=model.decode_step)
        token = jax.random.categorical(jax.random.fold_in(key, i), logits).astype(jnp.int32)
        return (token, cache), token
    _, tokens = jax.lax.scan(step, (previous, cache), jnp.arange(length))
    return tokens.T
