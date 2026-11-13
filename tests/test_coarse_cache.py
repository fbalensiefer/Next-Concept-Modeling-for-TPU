import jax
import jax.numpy as jnp
import numpy as np
from ncm.config import NCMConfig,RVQConfig,TransformerConfig
from ncm.generator import SharedLevelGenerator
from ncm.rvq import init_codebooks
from ncm.autoregressive import AutoregressiveBytes


def test_coarse_cache_matches_recompute():
    r=RVQConfig((3,5,8),4,4)
    g=SharedLevelGenerator(r,TransformerConfig(8,2,2),NCMConfig(2,3))
    books=init_codebooks(jax.random.key(200),r)
    masks=jnp.ones((3,2,8),bool).at[0,1,1].set(False).at[1,1,4:].set(False).at[2,1,6:].set(False)
    targets=jnp.zeros_like(masks,jnp.int32)
    prompt=jnp.array([[1,2,3],[4,5,0]])
    pm=jnp.array([[1,1,1],[1,1,0]],bool)
    variables=g.init(jax.random.key(200),books,masks,targets,prompt_ids=prompt,prompt_mask=pm)
    for seed in (200,201,202):
        for temperature in (0.,1.):
            def sample(cached):
                return jax.jit(lambda: g.apply(variables,books,masks,targets,prompt_ids=prompt,
                    prompt_mask=pm,teacher_forcing=False,temperature=temperature,use_cache=cached,
                    rngs={'sample':jax.random.key(seed)}))()
            a,b=sample(False),sample(True)
            np.testing.assert_array_equal(a.indices,b.indices)
            np.testing.assert_allclose(a.logits,b.logits,atol=1e-5,rtol=1e-5)


def test_padded_cache_logits_every_position():
    m=AutoregressiveBytes(TransformerConfig(8,2,2),8)
    x=jax.random.randint(jax.random.key(200),(2,8),0,256)
    valid=jnp.array([[1,1,1,1,1,1,1,1],[1,1,0,1,1,0,0,0]],bool)
    p=m.init(jax.random.key(201),x,valid)
    full=m.apply(p,x,valid)
    shape=(2,8,2,4)
    cache=tuple((jnp.zeros(shape),jnp.zeros(shape)) for _ in range(2))
    for i in range(8):
        logits,cache=m.apply(p,x[:,i],i,cache,valid=valid,method=m.decode_step)
        np.testing.assert_allclose(logits,full[:,i],atol=1e-5,rtol=1e-5)
