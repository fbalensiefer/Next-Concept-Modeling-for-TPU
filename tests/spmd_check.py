"""Executed in a fresh interpreter with four virtual CPU devices."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ncm import Batch, NCM, RVQConfig, TransformerConfig, initialize, make_train_step, ema_update, quantize
from ncm.sharding import compile_train_step, create_mesh


def main():
    assert len(jax.devices()) == 4
    config = RVQConfig((1, 2, 4), 4, 4)
    model = NCM(config, TransformerConfig(8, 2, 1), 16)
    # Unequal valid-token counts on each dp shard catch mean-of-means bugs.
    mask = jnp.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], bool)
    batch = Batch(jnp.arange(16).reshape(4, 4), mask,
                  jnp.zeros((4, 4), jnp.int32), jnp.zeros((4, 4), bool))
    # A non-tiny epsilon avoids amplifying roundoff in analytically zero
    # attention-bias gradients when different reduction orders are selected.
    optimizer = optax.adam(1e-3, eps=1e-5)
    initial = initialize(model, optimizer, jax.random.key(1), batch)
    fn = make_train_step(model, optimizer)
    reference, reference_metrics = jax.jit(fn)(initial, batch, jnp.array(0), jax.random.key(2))
    mesh = create_mesh(2, 2)
    compiled, state, sharded_batch = compile_train_step(fn, initial, batch, mesh)
    result, metrics = compiled(state, sharded_batch, jnp.array(0), jax.random.key(2))
    actual_leaves = jax.tree_util.tree_flatten_with_path((result, metrics))[0]
    expected_leaves = jax.tree.leaves((reference, reference_metrics))
    for (path, actual), expected in zip(actual_leaves, expected_leaves):
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5,
                                   err_msg=str(path))
    assert all(x.sharding.is_fully_replicated for x in jax.tree.leaves(result.codebooks))
    assert any(not x.sharding.is_fully_replicated for x in jax.tree.leaves(result.params))
    assert any(not x.sharding.is_fully_replicated for x in jax.tree.leaves(result.opt_state))
    # Manual local-array path: reduce sufficient statistics explicitly across dp.
    local_counts = jnp.arange(4 * 3 * 4, dtype=jnp.float32).reshape(4, 3, 4)
    local_sums = jnp.ones((4, 3, 4, 4))
    manual = jax.pmap(lambda c, s: ema_update(initial.codebooks, c, s, config, axis_name="dp"),
                      axis_name="dp")(local_counts, local_sums)
    expected = ema_update(initial.codebooks, local_counts.sum(0), local_sums.sum(0), config)
    for replicas, value in zip(jax.tree.leaves(manual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(replicas, jnp.broadcast_to(value, replicas.shape), atol=1e-6)
    latent = jnp.arange(64, dtype=jnp.float32).reshape(4, 4, 4) / 64
    global_rvq = quantize(latent, mask, initial.codebooks, config)
    local_rvq = jax.pmap(lambda x, m: quantize(x, m, initial.codebooks, config, axis_name="dp"),
                         axis_name="dp")(latent[:, None], mask[:, None])
    for name in ("counts", "sums", "soft_counts", "commitment_numerator", "commitment_denominator"):
        replicas, value = getattr(local_rvq, name), getattr(global_rvq, name)
        np.testing.assert_allclose(replicas, jnp.broadcast_to(value, replicas.shape), atol=1e-6)
    print("2x2 dp/fsdp training and manual collective EMA match global reference")


if __name__ == "__main__":
    main()
