# Information-preserving baselines

These are established constructions applied to the project's synthetic laws, not newly invented algorithms or learned language representations.

For the persistent symmetric binary Markov source, use `y[0]=x[0]` and `y[t]=x[t] XOR x[t-1]`. The initial bit is fair; subsequent flip indicators are independent Bernoulli(0.15). Prefix XOR reconstructs every sequence. An associative parallel scan takes linear work and logarithmic depth.

For noisy parity (90% uniform even-parity sequences, 10% uniform sequences), retain the first `L-1` bits and replace the final bit with total parity. All retained bits are independent and fair, and the parity bit is an independent Bernoulli(0.05) error indicator. XOR reconstructs the original final bit. Both transforms are bijections and allow a single parallel draw of latent bits followed by deterministic decoding. This is stronger than the earlier two-round source-space parity schedule.

For a bijection T, KL(P_X || (T^-1)#q) = KL(P_T(X) || q). Information-preserving coordinates remove codec collisions and make dependence reduction interpretable without a reconstruction tradeoff. For unrestricted independent marginals, the best code KL is total correlation. An arbitrary finite law need not admit independent fixed-length coordinates under a deterministic bijection, and an expensive inverse can eliminate any sampling benefit.

The mixture sources also have simple known-structure generators: a three-state regime selects either deterministic mode or the uniform component. These controls show low description complexity of the benchmarks; they do not establish that structure can be learned from 512 observations or transferred to natural language.

Reproduce exhaustive reconstruction, bijectivity and factorization checks:

```bash
cargo run --bin ncm-run -- domain-source-baselines
.venv312/bin/python -m pytest -q tests/test_domain_validation.py
```

Results are in `reports/invertible-baselines/results.json`. The inputs are analytic laws in `src/ncm/domain_validation.py`; no ignored checkpoint is required.

Reference: Guy E. Blelloch, *Prefix Sums and Their Applications*, 1993, https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf.
