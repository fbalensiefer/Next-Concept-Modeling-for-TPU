"""Small stationary categorical priors with sparse smoothed transitions.

Each sequence is modeled independently: start ~ marginal, then either iid or
first-order transitions. Unseen contexts back off to the positive marginal.
"""
from collections import Counter, defaultdict
import numpy as np


class TablePrior:
    def __init__(self, values, vocabulary, structured=False, alpha=1.):
        values = np.asarray(values)
        if values.ndim != 2 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Require nonempty integer [batch,length] values')
        if vocabulary < 2 or np.any(values < 0) or np.any(values >= vocabulary) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Invalid vocabulary, values or smoothing')
        self.vocabulary, self.structured, self.alpha = vocabulary, structured, alpha
        self.marginal = (np.bincount(values.ravel(), minlength=vocabulary) + alpha/vocabulary)/(values.size+alpha)
        self.transitions = defaultdict(Counter)
        if structured:
            for a, b in zip(values[:, :-1].ravel(), values[:, 1:].ravel()):
                self.transitions[int(a)][int(b)] += 1
        self.totals = {a: sum(row.values()) for a, row in self.transitions.items()}
        self.marginal_cdf = np.cumsum(self.marginal)
        self.rows = {a: (np.array(sorted(row)), np.cumsum([row[b]/self.totals[a] for b in sorted(row)]))
                     for a, row in self.transitions.items()}

    def probability(self, value, context=None):
        if context is None or context not in self.totals:
            return self.marginal[value]
        return (self.transitions[context].get(value, 0)+self.alpha*self.marginal[value])/(self.totals[context]+self.alpha)

    def nll(self, values):
        values = np.asarray(values)
        total = 0.
        for row in values:
            total -= np.log(self.marginal[row[0]])
            if self.structured:
                total -= sum(np.log(self.probability(int(b), int(a))) for a,b in zip(row[:-1],row[1:]))
            else:
                total -= np.log(self.marginal[row[1:]]).sum()
        return float(total/values.size)

    def sample(self, rng, count, length):
        if not self.structured:
            return np.minimum(np.searchsorted(self.marginal_cdf, rng.random((count,length))), self.vocabulary-1).astype(np.int32)
        result = np.empty((count,length),np.int32)
        result[:,0] = np.minimum(np.searchsorted(self.marginal_cdf,rng.random(count)),self.vocabulary-1)
        for position in range(1,length):
            result[:,position] = np.minimum(np.searchsorted(self.marginal_cdf,rng.random(count)),self.vocabulary-1)
            if self.structured:
                for index, context in enumerate(result[:,position-1]):
                    total = self.totals.get(int(context),0)
                    if total and rng.random() < total/(total+self.alpha):
                        categories,cdf = self.rows[int(context)]
                        result[index,position] = categories[min(np.searchsorted(cdf,rng.random()),len(categories)-1)]
        return result

    def storage_bytes(self):
        # Numeric payload only; Python dictionary/object overhead is excluded.
        return int(self.marginal.nbytes+self.marginal_cdf.nbytes+
                   sum(keys.nbytes+cdf.nbytes for keys,cdf in self.rows.values()))
