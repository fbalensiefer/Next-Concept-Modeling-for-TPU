"""Exercise initialized buffers and random replacements on a virtual 2x2 mesh."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ncm import Batch, CoarseConfig, NCM, RVQConfig, TransformerConfig, initialize, make_train_step
from ncm.sharding import compile_train_step, create_mesh


def main():
    assert len(jax.devices()) == 4
    config = RVQConfig((1, 2, 4), 2, 4, coarse=CoarseConfig(
        enabled=True, warmup_batches=1, kmeans_iterations=3, dead_code_threshold=100.,
        dead_code_grace_steps=0, usage_decay=0.))
    model = NCM(config, TransformerConfig(8, 2, 1), 16)
    mask = jnp.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], bool)
    batch = Batch(jnp.arange(16).reshape(4, 4), mask, jnp.zeros((4, 4), jnp.int32), jnp.zeros((4, 4), bool))
    optimizer = optax.adam(1e-3, eps=1e-5)
    initial = initialize(model, optimizer, jax.random.key(1), batch)
    step = make_train_step(model, optimizer)
    reference, expected_metrics = jax.jit(step)(initial, batch, jnp.array(0), jax.random.key(2))
    compiled, placed, data = compile_train_step(step, initial, batch, create_mesh(2, 2))
    actual, metrics = compiled(placed, data, jnp.array(0), jax.random.key(2))
    for a, b in zip(jax.tree.leaves((actual, metrics)), jax.tree.leaves((reference, expected_metrics))):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
    assert metrics['coarse_dead_replacements'] == 2
    assert all(x.sharding.is_fully_replicated for x in jax.tree.leaves(actual.opt_state['coarse']))
    assert all(x.sharding.is_fully_replicated for x in jax.tree.leaves(actual.codebooks))
    print('Coarse initialization, usage buffers and seeded replacements match on a 2x2 mesh')


if __name__ == '__main__':
    main()
