import itertools
import numpy as np
import pytest
from ncm.dependency_validation import MODELS,fit_law


@pytest.mark.parametrize('model',MODELS)
def test_empirical_priors_normalize_and_do_not_lose_unobserved_code_support(model):
    words=np.asarray(list(itertools.product([0,1],repeat=5)))
    samples=np.repeat(words[[0,31]],10,axis=0)
    q,metadata=fit_law(samples,words,model)
    assert np.all(q>0)
    np.testing.assert_allclose(q.sum(),1.,atol=1e-12)
    assert metadata['serial_rounds']>=2


def test_joint_tables_match_closed_form_posterior_predictive():
    words=np.asarray(list(itertools.product([0,1],repeat=3)))
    samples=np.repeat(words[[0]],4,axis=0)
    q,metadata=fit_law(samples,words,'joint_fine')
    expected=np.full(8,(.5/5)/4)
    expected[:4]=(4.5/5)*np.array([4.25,.25,.25,.25])/5
    np.testing.assert_allclose(q,expected,atol=1e-12)
    assert metadata['free_parameters']==7
