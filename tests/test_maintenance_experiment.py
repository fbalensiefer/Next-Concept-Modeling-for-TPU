"""Behavioral checks for the maintenance ablation and its exact-law evaluator."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from ncm.config import CoarseConfig, RVQConfig
from ncm.coarse import new_coarse_state
from ncm.maintenance_experiment import minimal_ema, ema_recovery
from ncm.rvq import CodebookState, ema_update, quantize


def test_minimal_and_recovery_separate_rescue_from_ordinary_ema():
    cfg=RVQConfig((1,),2,1,coarse=CoarseConfig(enabled=True,usage_decay=0.,
        dead_code_threshold=.5,dead_code_grace_steps=0,replacement_noise=.001))
    e=jnp.array([[[0.],[100.]]]);counts=jnp.array([[1000.,1.]])
    books=CodebookState(e,counts,e*counts[...,None])
    state=new_coarse_state(books,cfg.coarse,jax.random.key(0))
    latent=jnp.array([[[10.]],[[10.]],[[99999.]]]);mask=jnp.array([[1],[1],[0]],bool)
    stats=quantize(latent,mask,books,cfg)
    minimal,mt,_=minimal_ema(books,state,stats.counts,stats.sums,latent,mask,cfg)
    expected=ema_update(books,stats.counts,stats.sums,cfg)
    for a,b in zip(jax.tree.leaves(minimal),jax.tree.leaves(expected)):np.testing.assert_array_equal(a,b)
    rescued,rt,_=ema_recovery(books,state,stats.counts,stats.sums,latent,mask,cfg)
    assert int(mt.replacements.sum())==0 and int(rt.replacements.sum())==1
    assert abs(float(rescued.embeddings[0,1,0])-10)<.01
    # With recovery disabled, ordinary EMA and rate-one maintained EMA agree.
    cfg=replace(cfg,coarse=replace(cfg.coarse,dead_code_threshold=0.))
    recovered,_,_=ema_recovery(books,state,stats.counts,stats.sums,latent,mask,cfg)
    for a,b in zip(jax.tree.leaves(recovered),jax.tree.leaves(expected)):np.testing.assert_allclose(a,b,atol=1e-6)


def test_fixed_binary_preserves_coarse_endpoints_but_updates_fine_books():
    from ncm.maintenance_experiment import fixed_binary
    from ncm.rvq import init_codebooks
    cfg=RVQConfig((1,2),2,2,coarse=CoarseConfig(enabled=True))
    books=init_codebooks(jax.random.key(2),cfg)
    state=new_coarse_state(books,cfg.coarse,jax.random.key(3))
    counts=jnp.ones((2,2))*5;sums=jnp.ones((2,2,2))*10
    updated,_,_=fixed_binary(books,state,counts,sums,jnp.ones((2,2,2)),jnp.ones((2,2),bool),cfg)
    for field in ['embeddings','counts','sums']:
        np.testing.assert_array_equal(getattr(updated,field)[0],getattr(books,field)[0])
    assert not np.array_equal(updated.embeddings[1],books.embeddings[1])
