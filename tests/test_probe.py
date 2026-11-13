import numpy as np
from ncm.probe import source_law,exact_conditional_tc


def test_enumerable_source_tc():
    for n in (8,10):
        for source in ('markov','parity','copied_bit','hmm','constant_mixture','uniform'):
            words,p=source_law(source,n)
            assert p.dtype==np.float64
            np.testing.assert_allclose(p.sum(),1,atol=1e-14)
            values=exact_conditional_tc(words,p,(n//2,n//2))
            assert min(values)>=0
            if source=='uniform':np.testing.assert_allclose(values,0,atol=1e-13)
            if source=='copied_bit':np.testing.assert_allclose(values,[(n//2-1)*np.log(2),0],atol=1e-13)
