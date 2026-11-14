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
