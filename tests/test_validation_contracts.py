import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ncm.autoregressive import AutoregressiveBytes, shifted_inputs
from ncm.config import NCMConfig, RVQConfig, TransformerConfig
from ncm.generator import SharedLevelGenerator
from ncm.rvq import init_codebooks
from ncm.speculative import DistributionAdapter, speculative_generate


def test_ar_cache_padded_and_unpadded():
    model=AutoregressiveBytes(TransformerConfig(8,2,2),8)
    x=jax.random.randint(jax.random.key(200),(2,8),0,256)
    valid=jnp.arange(8)[None] < jnp.array([8,5])[:,None]
    params=model.init(jax.random.key(200),x,valid)
    full=model.apply(params,x,valid)
    shape=(2,8,2,4)
    cache=tuple((jnp.zeros(shape),jnp.zeros(shape)) for _ in range(2))
    for i in range(8):
        logits,cache=model.apply(params,x[:,i],i,cache,method=model.decode_step)
        np.testing.assert_allclose(np.asarray(logits)[np.asarray(valid[:,i])],
                                   np.asarray(full[:,i])[np.asarray(valid[:,i])],atol=1e-5,rtol=1e-5)
    # Invalid suffix logits are deliberately outside the cache API's contract.


def test_two_coarse_masks_and_prompt_isolation():
    r=RVQConfig((3,4,6),4,4)
    g=SharedLevelGenerator(r,TransformerConfig(8,2,1),NCMConfig(2,2))
    books=init_codebooks(jax.random.key(200),r)
    masks=jnp.ones((3,1,6),bool)
    targets=jnp.zeros((3,1,6),jnp.int32)
    prompt=jnp.array([[1,2]])
    params=g.init(jax.random.key(201),books,masks,targets,prompt_ids=prompt)
    reference=g.apply(params,books,masks,targets,prompt_ids=prompt).logits
    for k,n in enumerate((3,4)):
        for j in range(n-1):
            changed=g.apply(params,books,masks,targets.at[k,:,j+1:].set(3),prompt_ids=prompt).logits
            np.testing.assert_allclose(reference[k,:,:j+1],changed[k,:,:j+1],atol=1e-6)
    a=g.apply(params,books,masks,targets,prompt_ids=prompt,teacher_forcing=False).logits
    b=g.apply(params,books,masks,targets+3,prompt_ids=prompt,teacher_forcing=False).logits
    np.testing.assert_array_equal(a,b)
    c=g.apply(params,books,masks,targets,prompt_ids=prompt+2,teacher_forcing=False).logits
    assert not np.allclose(a,c)
    from ncm.training import NCM,Batch,initialize
    from ncm.generation import encode
    import optax
    model=NCM(r,TransformerConfig(8,2,1),256,config=NCMConfig(2,2))
    batch=Batch(jnp.zeros((1,6),jnp.int32),jnp.ones((1,6),bool),jnp.zeros((1,6),jnp.int32),jnp.zeros((1,6),bool),prompt)
    state=initialize(model,optax.adam(.001),jax.random.key(202),batch)
    cp=state.params['MultiscaleTokenizer_0']
    for field in ('indices','quantized'):
        np.testing.assert_array_equal(getattr(encode(model,cp,state.codebooks,batch),field),
            getattr(encode(model,cp,state.codebooks,batch.replace(prompt_ids=prompt+20)),field))


def test_speculative_sharding():
    if jax.device_count()<4:
        pytest.skip('requires four virtual devices')
    from jax.sharding import NamedSharding,PartitionSpec as P
    from ncm.sharding import create_mesh
    mesh=create_mesh(2,2,jax.devices()[:4])
    prompts=jnp.array([[0],[1],[2],[0]])
    keys=jax.random.split(jax.random.key(200),4)
    for greedy in (False,True):
        def draft(_,buffer,pos,d,key):
            logits=jnp.array([.7,-.2,.1])
            if greedy: logits=jnp.array([0.,-jnp.inf,-jnp.inf])
            return jnp.broadcast_to(logits,(d,3))
        def teacher(_,buffer,pos,d):
            logits=jnp.array([.2,.8,-.3])
            if greedy: logits=jnp.array([-jnp.inf,0.,-jnp.inf])
            return jnp.broadcast_to(logits,(d+1,3))
        fn=jax.vmap(lambda row,key: speculative_generate(DistributionAdapter(None,draft),
                   DistributionAdapter(None,teacher),row,3,7,key))
        expected=jax.jit(fn)(prompts,keys)
        actual=jax.jit(fn,in_shardings=(NamedSharding(mesh,P('dp',None)),NamedSharding(mesh,P('dp'))))(prompts,keys)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a,b)


def test_likelihood_bound_float64():
    # Independent float64 enumeration; no global JAX precision changes.
    import itertools
    words=np.array(list(itertools.product(range(2),repeat=3)))
    rng=np.random.default_rng(200)
    prior=rng.dirichlet(np.ones(4))
    decoder=rng.dirichlet(np.ones(8),size=4)
    codes=np.array([0,1,1,2,2,2,3,3])
    joint=prior[codes]*decoder[codes,np.arange(8)]
    assert joint.sum()<=1+1e-9
    identity=np.eye(8,dtype=np.float64)
    prior=rng.dirichlet(np.ones(8))
    np.testing.assert_allclose(np.sum(prior*identity[np.arange(8),np.arange(8)]),1.,atol=1e-9)


def test_supported_exact_bucket_and_partial_window(tmp_path):
    from ncm.data import load_text
    path=tmp_path/'tail.txt'
    path.write_text('data')
    large=load_text(path,8)
    small=load_text(path,4)
    np.testing.assert_array_equal(large.tokens[:,:4],small.tokens)
    transform=TransformerConfig(8,2,1)
    r8,r4=RVQConfig((8,),4,4),RVQConfig((4,),4,4)
    g8=SharedLevelGenerator(r8,transform,NCMConfig(1,2))
    g4=SharedLevelGenerator(r4,transform,NCMConfig(1,2))
    books=init_codebooks(jax.random.key(200),r8)
    t8,t4=jnp.asarray(large.tokens)[None],jnp.asarray(small.tokens)[None]
    m8,m4=jnp.asarray(large.mask)[None],jnp.asarray(small.mask)[None]
    u=jnp.array([[3,4]])
    p8=g8.init(jax.random.key(201),books,m8,jnp.zeros_like(t8),prompt_ids=u)
    import copy
    p4=copy.deepcopy(p8)
    p4['params']['HybridGenerator_0']['position']=p8['params']['HybridGenerator_0']['position'][:4]
    a=g8.apply(p8,books,m8,jnp.zeros_like(t8),prompt_ids=u,teacher_forcing=False)
    b=g4.apply(p4,books,m4,jnp.zeros_like(t4),prompt_ids=u,teacher_forcing=False)
    np.testing.assert_array_equal(a.indices[:,:,:4],b.indices)
    np.testing.assert_allclose(jax.nn.log_softmax(a.logits[:,:,:4]),jax.nn.log_softmax(b.logits),atol=1e-5,rtol=1e-5)
    def draft(_,buffer,pos,d,key):
        return jnp.broadcast_to(jnp.array([0.,-jnp.inf]),(d,2))
    def teacher(_,buffer,pos,d):
        tokens=(jnp.arange(d+1)+pos)%2
        return jnp.where(jax.nn.one_hot(tokens,2),0.,-jnp.inf)
    def sample(n):
        return speculative_generate(DistributionAdapter(None,draft),DistributionAdapter(None,teacher),
                                    jnp.array([0]),3,n,jax.random.key(200)).tokens
    np.testing.assert_array_equal(sample(4),sample(8)[:4])
    np.testing.assert_array_equal(jax.nn.log_softmax(teacher(None,jnp.zeros(16),1,3)),
                                  jax.nn.log_softmax(teacher(None,jnp.zeros(8),1,3)))


def test_sharded_history_conditioned_sampled_logits():
    if jax.device_count()<4:
        pytest.skip('requires four virtual devices')
    from jax.sharding import NamedSharding,PartitionSpec as P
    from ncm.sharding import create_mesh
    mesh=create_mesh(2,2,jax.devices()[:4])
    rows=jnp.array([[0,1,2],[1,2,0],[2,0,1],[0,2,1]],jnp.int32)
    keys=jax.random.split(jax.random.key(202),4)
    def logits(row,key):
        # Both history and a sampled latent affect each proposal position.
        latent=jax.random.normal(key,(3,3))*.1
        proposal=jnp.sin(row[:,None]+jnp.arange(3)[None]*.7)+latent
        teacher=jnp.cos(row[:,None]*.3+jnp.arange(3)[None])
        return proposal,teacher
    function=jax.vmap(logits)
    expected=jax.jit(function)(rows,keys)
    actual=jax.jit(function,in_shardings=(NamedSharding(mesh,P('dp',None)),NamedSharding(mesh,P('dp'))))(rows,keys)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5)
