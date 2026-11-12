# Minimal SCALE mechanism pilot

Pilot did not meet the fixed gates; no SCALE benefit established.

| Seed | Arm | Byte reconstruction | Hard conditional TC (nats) | Exact output TV | Missing teacher support |
|---:|---|---:|---:|---:|---:|
| 0 | reconstruction | 0.9977 | 0.1694 | 0.1755 | 0.0000 |
| 0 | alignment | 0.9953 | 0.1593 | 0.1692 | 0.0000 |
| 0 | dependency | 0.9953 | 0.2412 | 0.1598 | 0.0000 |
| 0 | alignment_dependency | 1.0000 | 0.1115 | 0.1453 | 0.0000 |
| 1 | reconstruction | 1.0000 | 1.0019 | 0.6135 | 0.0000 |
| 1 | alignment | 1.0000 | 1.0019 | 0.6135 | 0.0000 |
| 1 | dependency | 1.0000 | 1.0019 | 0.6135 | 0.0000 |
| 1 | alignment_dependency | 1.0000 | 0.2112 | 0.1784 | 0.0000 |
| 2 | reconstruction | 0.9977 | 0.1385 | 0.2193 | 0.0000 |
| 2 | alignment | 0.9977 | 0.2037 | 0.2247 | 0.0000 |
| 2 | dependency | 0.9977 | 0.1585 | 0.2436 | 0.0000 |
| 2 | alignment_dependency | 0.9953 | 0.0876 | 0.1920 | 0.0000 |
| 3 | reconstruction | 1.0000 | 1.0019 | 0.6100 | 0.0000 |
| 3 | alignment | 0.9977 | 0.1471 | 0.1305 | 0.0000 |
| 3 | dependency | 1.0000 | 1.0019 | 0.6097 | 0.0000 |
| 3 | alignment_dependency | 0.9977 | 0.0879 | 0.1289 | 0.0000 |
| 4 | reconstruction | 1.0000 | 0.2630 | 0.1256 | 0.0000 |
| 4 | alignment | 1.0000 | 0.1078 | 0.1169 | 0.0000 |
| 4 | dependency | 0.9637 | 0.1210 | 0.1842 | 0.1453 |
| 4 | alignment_dependency | 0.9614 | 0.1353 | 0.1787 | 0.0663 |
| 5 | reconstruction | 0.9930 | 0.3063 | 0.1824 | 0.0000 |
| 5 | alignment | 1.0000 | 0.1858 | 0.2000 | 0.0000 |
| 5 | dependency | 0.9977 | 0.0879 | 0.1688 | 0.0000 |
| 5 | alignment_dependency | 1.0000 | 0.1858 | 0.1981 | 0.0000 |
| 6 | reconstruction | 1.0000 | 0.2604 | 0.1309 | 0.0000 |
| 6 | alignment | 1.0000 | 0.3835 | 0.1315 | 0.0000 |
| 6 | dependency | 1.0000 | 0.2489 | 0.1339 | 0.0000 |
| 6 | alignment_dependency | 1.0000 | 0.0959 | 0.1249 | 0.0000 |
| 7 | reconstruction | 0.9953 | 0.4051 | 0.1596 | 0.0000 |
| 7 | alignment | 0.9977 | 0.1759 | 0.1478 | 0.0093 |
| 7 | dependency | 1.0000 | 0.1757 | 0.1392 | 0.0000 |
| 7 | alignment_dependency | 1.0000 | 0.0959 | 0.1580 | 0.0000 |

Dependence-aware seeds with missing teacher support (infinite forward KL): [4].
Mean paired TV improvement: 0.0537.
Mean paired hard-TC reduction: 0.1692 nats.
Fidelity gate passed: False.
Matched student FLOP estimates: True.

Gates and weights were fixed in protocol.json before training. This is a small synthetic
mechanism test, not evidence of pretrained-LLM conversion or general language quality.
See report.json for seed-level results, exact distributions, budgets and limitations.

## Four-arm and coverage follow-up

Confirmation student seeds: [3, 4, 5, 6, 7].
Coverage repair gate passed: False.
Teacher alignment gate passed: False.
Mean alignment TV improvement over dependence only: 0.0895.

| New seed | Alignment gain over dependence only | Repair TV change | Missing teacher mass before | After | Sequence reconstruction after |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.4808 | -0.0047 | 0.0204 | 0.0000 | 0.9907 |
| 4 | 0.0055 | -0.0297 | 0.1505 | 0.0663 | 0.8787 |
| 5 | -0.0292 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 6 | 0.0091 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 7 | -0.0187 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

Repair replay uses observed training sequences only and updates decoder parameters only.
All four arms receive identical repair updates. Seeds 0–2 replicate known runs;
only the listed new student seeds determine follow-up gates. The teacher/domain is unchanged.
