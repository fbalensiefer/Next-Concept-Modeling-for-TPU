"""CPU smoke run; on TPU select mesh dimensions for the available slice."""
import jax
import jax.numpy as jnp
import optax

from ncm import Batch, LossConfig, NCM, RVQConfig, TransformerConfig, initialize, make_train_step
from ncm.sharding import compile_train_step, create_mesh


def main():
    key = jax.random.key(0)
    config = RVQConfig((2, 4, 8), codebook_size=16, dim=16)
    model = NCM(config, TransformerConfig(width=32, heads=4, layers=2),
                token_vocab_size=256, losses=LossConfig(concept=.1))
    batch = Batch(jnp.arange(16, dtype=jnp.int32).reshape(2, 8),
                  jnp.ones((2, 8), bool), jnp.zeros((2, 8), jnp.int32),
                  jnp.zeros((2, 8), bool))
    optimizer = optax.adamw(1e-3)
    state = initialize(model, optimizer, key, batch)
    mesh = create_mesh(dp=len(jax.devices()), fsdp=1)
    step, state, batch = compile_train_step(make_train_step(model, optimizer), state, batch, mesh)
    for i in range(3):
        key, subkey = jax.random.split(key)
        # The same executable handles both phases.
        state, metrics = step(state, batch, jnp.asarray(i >= 2, jnp.int32), subkey)
        print({k: float(v) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
