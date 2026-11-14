import numpy as np
import pytest
from ncm.domain_validation import DOMAINS, source_law


@pytest.mark.parametrize('name,length,seed',DOMAINS)
def test_domain_laws_have_full_normalized_support_and_expected_structure(name,length,seed):
    tokens,p=source_law(name,length)
    assert tokens.shape==(2**length,length)
    assert np.all(p>0)
    np.testing.assert_allclose(p.sum(),1.)
    bits=tokens-65
    if name=='biased_independent':
        np.testing.assert_allclose(p@bits,np.full(length,.2))
    elif name=='persistent_markov':
        np.testing.assert_allclose(p@(bits[:,1:]==bits[:,:-1]),np.full(length-1,.85))
    elif name=='noisy_parity':
        np.testing.assert_allclose(p[bits.sum(-1)%2==0].sum(),.95)
        # Pairwise independence does not remove the higher-order parity constraint.
        np.testing.assert_allclose(p@(bits[:,0]*bits[:,1]),.25)
    elif name=='skewed_modes':
        np.testing.assert_allclose(p[0],.8+.1/(2**length))


@pytest.mark.parametrize('name,levels',[
    ('noisy_parity',[list(range(7)),[7]]),
    ('persistent_markov',[[3],[1,5],[0,2,4,6],[7]]),
])
def test_dependency_aware_source_schedules_recover_the_exact_law(name,levels):
    from ncm.workloads.research.domain_source_baselines import scheduled_law
    tokens,p=source_law(name,8)
    q=scheduled_law(tokens,p,levels)
    np.testing.assert_allclose(q,p,atol=1e-14,rtol=1e-14)


@pytest.mark.parametrize('name', ['persistent_markov','noisy_parity'])
def test_invertible_innovations_preserve_every_input_and_factorize(name):
    from ncm.workloads.research.domain_source_baselines import innovations,invert_innovations
    tokens,p=source_law(name,8)
    values=innovations(name,tokens-65)
    np.testing.assert_array_equal(invert_innovations(name,values),tokens-65)
    assert len(np.unique(values,axis=0))==256
    marginals=p@values
    product=np.prod(np.where(values,marginals,1-marginals),axis=1)
    np.testing.assert_allclose(product,p,atol=1e-14,rtol=1e-14)
