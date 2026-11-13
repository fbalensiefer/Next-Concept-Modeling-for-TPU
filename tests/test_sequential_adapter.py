import jax
import jax.numpy as jnp
import numpy as np
from ncm.autoregressive import AutoregressiveBytes,shifted_inputs
from ncm.config import TransformerConfig
from ncm.speculative import ar_adapter


def test_ar_draft_probabilities_are_its_true_sampling_law():
    m=AutoregressiveBytes(TransformerConfig(8,2,1),16)
    prompt=jnp.array([1,2,3]); buffer=jnp.zeros(16,jnp.int32).at[:3].set(prompt)
    params=m.init(jax.random.key(200),buffer[None],jnp.ones((1,16),bool))['params']
    adapter=ar_adapter(m,params,draft=True)
    tokens,logits=jax.jit(adapter.draft_fn,static_argnums=(3,))(params,buffer,jnp.asarray(3),4,jax.random.key(201))
    proposed=buffer.at[3:7].set(tokens)
    expected=m.apply({'params':params},shifted_inputs(proposed[None]),jnp.ones((1,16),bool))[0,3:7]
    np.testing.assert_allclose(logits,expected,atol=1e-5,rtol=1e-5)
    sampled=jax.vmap(lambda i,row:jax.random.categorical(jax.random.fold_in(jax.random.key(201),i),row))(jnp.arange(4),logits)
    np.testing.assert_array_equal(tokens,sampled)
