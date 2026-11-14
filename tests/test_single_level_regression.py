import jax
import jax.numpy as jnp
import numpy as np
from ncm.config import RVQConfig,TransformerConfig
from ncm.generator import SharedLevelGenerator
from ncm.rvq import init_codebooks


def test_legacy_single_level_generator():
    r=RVQConfig((4,),4,4)
    model=SharedLevelGenerator(r,TransformerConfig(8,2,1))
    books=init_codebooks(jax.random.key(200),r)
    mask=jnp.ones((1,1,4),bool);target=jnp.zeros((1,1,4),jnp.int32)
    variables=model.init(jax.random.key(200),books,mask,target)
    output=model.apply(variables,books,mask,target,teacher_forcing=False)
    assert output.logits.shape==(1,1,4,4)
    assert np.isfinite(output.logits).all()

