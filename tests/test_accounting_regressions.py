"""Structural compiler checks for analytic estimates, not hardware counters."""
import jax
import jax.numpy as jnp
import numpy as np
from ncm.autoregressive import AutoregressiveBytes
from ncm.config import RVQConfig,TransformerConfig
from ncm.flops import FlopModel,ar_cached_generation,codec_forward
from ncm.generator import SharedLevelGenerator
from ncm.rvq import init_codebooks,quantize


def compiler_estimate(compiled):
    result=compiled.cost_analysis()
    if isinstance(result,list):result=result[0]
    return result['flops']


def test_cache_only_prefill_has_no_capacity_attention_cost():
    model=AutoregressiveBytes(TransformerConfig(8,2,1),64)
    params=model.init(jax.random.key(200),jnp.zeros((1,1),jnp.int32),jnp.ones((1,1),bool))['params']
    def cache_only(params,cache):
        return model.apply({'params':params},jnp.array([1]),jnp.asarray(0),cache,method=model.decode_step)[1]
    estimates=[]
    for capacity in (8,64):
        shape=(1,capacity,2,4)
        cache=((jnp.zeros(shape),jnp.zeros(shape)),)
        estimates.append(compiler_estimate(jax.jit(cache_only).lower(params,cache).compile()))
    # Only normalized K/V projections survive, with no attention over capacity.
    np.testing.assert_allclose(estimates[0],estimates[1],atol=0,rtol=0)
    config=FlopModel(8,1,2)
    increments=[ar_cached_generation(config,1,c,2,1)-ar_cached_generation(config,1,c,0,1) for c in (8,64)]
    np.testing.assert_allclose(increments[0],increments[1],atol=0,rtol=0)


def test_padded_rvq_distance_execution_uses_stored_grid():
    estimates=[];analytic=[]
    latent=jax.random.normal(jax.random.key(200),(1,8,4));mask=jnp.ones((1,8),bool)
    for lengths in ((1,2,8),(2,4,8)):
        r=RVQConfig(lengths,4,4);books=init_codebooks(jax.random.key(200),r)
        fn=lambda latent,mask,books:quantize(latent,mask,books,r)
        estimates.append(compiler_estimate(jax.jit(fn).lower(latent,mask,books).compile()))
        analytic.append(codec_forward(1,8,4,levels=3,codes=4,lengths=lengths,decoder=False))
    np.testing.assert_allclose(estimates[0],estimates[1],atol=0,rtol=0)
    np.testing.assert_allclose(analytic[0],analytic[1],atol=0,rtol=0)
