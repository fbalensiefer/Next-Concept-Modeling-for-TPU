"""Learned conditional within-level dependence probe and float64 source laws."""
import itertools
from functools import lru_cache
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import optax
from .autoregressive import CausalBlock
from .config import TransformerConfig


class WithinLevelProbe(nn.Module):
    lengths: tuple
    level: int
    config: TransformerConfig=TransformerConfig(16,2,1)
    causal: bool=True
    vocabulary: int=2

    @nn.compact
    def __call__(self,codes):
        start=sum(self.lengths[:self.level]); n=self.lengths[self.level]
        prefix=codes[:,:start]
        if self.causal:
            current=jnp.concatenate([jnp.full((len(codes),1),self.vocabulary,jnp.int32),codes[:,start:start+n-1]],1)
        else:
            current=jnp.full((len(codes),n),self.vocabulary,jnp.int32)
        inputs=jnp.concatenate([prefix,current],1)
        position=self.param('position',nn.initializers.normal(.02),(start+n,self.config.width))
        h=nn.Embed(self.vocabulary+1,self.config.width)(inputs)+position[None]
        for i in range(self.config.layers):
            h=CausalBlock(self.config,name=f'block_{i}')(h,jnp.ones(inputs.shape,bool),causal=self.causal)
        return nn.Dense(self.vocabulary)(nn.LayerNorm()(h[:,start:]))


@lru_cache(None)
def _probe_functions(lengths,level,causal,batch_size):
    model=WithinLevelProbe(lengths,level,causal=causal)
    optimizer=optax.adam(.003)
    start=sum(lengths[:level]); n=lengths[level]
    @jax.jit
    def step(params,state,data,key):
        ids=jax.random.randint(key,(batch_size,),0,len(data)); x=data[ids]
        def loss(p):
            logits=model.apply({'params':p},x)
            return optax.softmax_cross_entropy_with_integer_labels(logits,x[:,start:start+n]).sum(1).mean()
        value,grads=jax.value_and_grad(loss)(params)
        updates,state=optimizer.update(grads,state,params)
        return optax.apply_updates(params,updates),state,value
    @jax.jit
    def score(params,x):
        logits=model.apply({'params':params},x)
        return optax.softmax_cross_entropy_with_integer_labels(logits,x[:,start:start+n]).sum(1)
    return model,optimizer,step,score


def train_probe(train,evaluation,lengths,level,seed,*,steps=200,batch_size=64,causal=True):
    model,optimizer,step,score=_probe_functions(tuple(lengths),level,causal,batch_size)
    key=jax.random.key(seed)
    params=model.init(key,jnp.asarray(train[:1]))['params']
    state=optimizer.init(params); data=jnp.asarray(train)
    for index in range(steps):
        params,state,_=step(params,state,data,jax.random.fold_in(key,index))
    values=[]
    for offset in range(0,len(evaluation),256):
        values.extend(np.asarray(score(params,jnp.asarray(evaluation[offset:offset+256]))).tolist())
    return float(np.mean(values)),params


def source_law(name,length):
    words=np.asarray(list(itertools.product((0,1),repeat=length)),np.int32)
    if name=='markov':
        p=.5*np.prod(np.where(words[:,1:]==words[:,:-1],.85,.15),axis=1)
    elif name=='parity':
        matches=words[:,-1]==(words[:,:-1].sum(1)%2)
        p=np.where(matches,.9,.1)/2**(length-1)
    elif name=='copied_bit':
        p=np.where(np.all(words==words[:,:1],1),.5,0.)
    elif name=='hmm':
        state=np.full((len(words),2),.5,np.float64)
        for i in range(length):
            state*=np.where(words[:,i,None]==np.arange(2)[None],.85,.15)
            if i<length-1: state=state@np.array([[.9,.1],[.1,.9]])
        p=state.sum(1)
    elif name=='constant_mixture':
        ones=words.sum(1)
        p=.5*(.1**ones*.9**(length-ones)+.9**ones*.1**(length-ones))
    elif name=='uniform': p=np.full(len(words),1/len(words),np.float64)
    else: raise ValueError(name)
    p=np.asarray(p,np.float64); p/=p.sum()
    return words,p


def exact_conditional_tc(words,probabilities,lengths):
    p=np.asarray(probabilities,np.float64)
    def entropy(kept):
        if not kept:return 0.
        index=words[:,kept]@(2**np.arange(len(kept)-1,-1,-1))
        mass=np.bincount(index,weights=p,minlength=2**len(kept))
        positive=mass>0
        return float(-np.sum(mass[positive]*np.log(mass[positive])))
    start=0; values=[]
    for n in lengths:
        prefix=list(range(start))
        tc=sum(entropy(prefix+[start+j]) for j in range(n))-entropy(list(range(start+n)))-(n-1)*entropy(prefix)
        values.append(max(tc,0.)); start+=n
    return values
