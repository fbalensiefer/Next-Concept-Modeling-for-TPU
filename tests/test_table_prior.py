import numpy as np
import pytest
from ncm.table_prior import TablePrior


def test_smoothed_rows_normalize_and_unseen_context_backs_off():
    prior=TablePrior(np.array([[0,1,0],[0,1,0]]),3,True)
    for context in (None,0,1,2):
        assert sum(prior.probability(i,context) for i in range(3)) == pytest.approx(1)
    assert prior.probability(1,2)==prior.marginal[1]
    assert all(prior.probability(i,0)>0 for i in range(3))


def test_markov_dependency_improves_alternating_heldout_law():
    data=np.tile([0,1,0,1,0,1],(100,1))
    independent=TablePrior(data,2)
    structured=TablePrior(data,2,True)
    assert structured.nll(data)<independent.nll(data)/2
    samples=structured.sample(np.random.default_rng(0),10000,2)
    observed=(samples[:,1]!=samples[:,0]).mean()
    assert observed>.98
    np.testing.assert_array_equal(samples,structured.sample(np.random.default_rng(0),10000,2))
