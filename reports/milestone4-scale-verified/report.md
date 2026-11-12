# Minimal SCALE mechanism pilot

Pilot gates passed; larger independent experiments still required.

| Seed | Arm | Byte reconstruction | Hard conditional TC (nats) | Exact output TV | Missing teacher support |
|---:|---|---:|---:|---:|---:|
| 0 | reconstruction | 0.9953 | 0.1694 | 0.1604 | 0.0000 |
| 0 | alignment | 0.9900 | 0.1593 | 0.1635 | 0.0000 |
| 0 | alignment_dependency | 0.9818 | 0.1115 | 0.1518 | 0.0512 |
| 1 | reconstruction | 1.0000 | 1.0019 | 0.6135 | 0.0000 |
| 1 | alignment | 1.0000 | 1.0019 | 0.6135 | 0.0000 |
| 1 | alignment_dependency | 1.0000 | 0.2112 | 0.1784 | 0.0000 |
| 2 | reconstruction | 0.9977 | 0.1385 | 0.2208 | 0.0000 |
| 2 | alignment | 0.9977 | 0.2037 | 0.2247 | 0.0000 |
| 2 | alignment_dependency | 0.9953 | 0.0876 | 0.1931 | 0.0000 |

Dependence-aware seeds with missing teacher support (infinite forward KL): [0].
Mean paired TV improvement: 0.1595.
Mean paired hard-TC reduction: 0.3182 nats.
Fidelity gate passed: True.
Matched student FLOP estimates: True.

Gates and weights were fixed in protocol.json before training. This is a small synthetic
mechanism test, not evidence of pretrained-LLM conversion or general language quality.
See report.json for seed-level results, exact distributions, budgets and limitations.
