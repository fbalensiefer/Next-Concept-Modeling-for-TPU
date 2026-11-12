# Minimal SCALE mechanism pilot

Pilot did not meet the fixed gates; no SCALE benefit established.

| Seed | Arm | Byte reconstruction | Hard conditional TC (nats) | Exact output TV | Missing teacher support |
|---:|---|---:|---:|---:|---:|
| 0 | reconstruction | 0.9977 | 0.0879 | 0.1705 | 0.0000 |
| 0 | alignment | 1.0000 | 0.0879 | 0.1734 | 0.0000 |
| 0 | dependency | 0.9977 | 0.1158 | 0.1733 | 0.0000 |
| 0 | alignment_dependency | 0.9977 | 0.1158 | 0.1765 | 0.0000 |
| 4 | reconstruction | 1.0000 | 1.0019 | 0.6135 | 0.0000 |
| 4 | alignment | 1.0000 | 0.0959 | 0.1110 | 0.0000 |
| 4 | dependency | 1.0000 | 0.0879 | 0.1214 | 0.0000 |
| 4 | alignment_dependency | 1.0000 | 0.0879 | 0.1159 | 0.0000 |
| 7 | reconstruction | 1.0000 | 0.0879 | 0.1333 | 0.0000 |
| 7 | alignment | 1.0000 | 0.0879 | 0.1382 | 0.0000 |
| 7 | dependency | 1.0000 | 0.0959 | 0.1276 | 0.0000 |
| 7 | alignment_dependency | 1.0000 | 0.0879 | 0.1374 | 0.0000 |
| 8 | reconstruction | 1.0000 | 1.0019 | 0.6087 | 0.0000 |
| 8 | alignment | 1.0000 | 1.0019 | 0.6124 | 0.0000 |
| 8 | dependency | 1.0000 | 1.0019 | 0.6105 | 0.0000 |
| 8 | alignment_dependency | 1.0000 | 1.0019 | 0.6125 | 0.0000 |
| 9 | reconstruction | 0.9977 | 0.1750 | 0.1638 | 0.0000 |
| 9 | alignment | 0.9953 | 0.1158 | 0.1884 | 0.0000 |
| 9 | dependency | 1.0000 | 0.0959 | 0.1650 | 0.0000 |
| 9 | alignment_dependency | 1.0000 | 0.0879 | 0.1910 | 0.0000 |
| 10 | reconstruction | 1.0000 | 1.0019 | 0.6139 | 0.0000 |
| 10 | alignment | 1.0000 | 0.0879 | 0.1322 | 0.0000 |
| 10 | dependency | 1.0000 | 1.0019 | 0.6139 | 0.0000 |
| 10 | alignment_dependency | 1.0000 | 0.0879 | 0.1313 | 0.0000 |
| 11 | reconstruction | 1.0000 | 1.0019 | 0.6132 | 0.0000 |
| 11 | alignment | 1.0000 | 0.0959 | 0.1390 | 0.0000 |
| 11 | dependency | 1.0000 | 1.0019 | 0.6135 | 0.0000 |
| 11 | alignment_dependency | 1.0000 | 0.0879 | 0.1545 | 0.0000 |
| 12 | reconstruction | 1.0000 | 0.0959 | 0.1291 | 0.0000 |
| 12 | alignment | 1.0000 | 0.0879 | 0.1131 | 0.0000 |
| 12 | dependency | 1.0000 | 1.0019 | 0.6114 | 0.0000 |
| 12 | alignment_dependency | 1.0000 | 0.0959 | 0.1445 | 0.0000 |

Dependence-aware seeds with missing teacher support (infinite forward KL): [].
Mean paired TV improvement: -0.0070.
Mean paired hard-TC reduction: 0.0010 nats.
Fidelity gate passed: True.
Matched student FLOP estimates: True.

Gates and weights were fixed in protocol.json before training. This is a small synthetic
mechanism test, not evidence of pretrained-LLM conversion or general language quality.
See report.json for seed-level results, exact distributions, budgets and limitations.

## Four-arm and coverage follow-up

Confirmation student seeds: [8, 9, 10, 11, 12].
Coverage repair gate passed: True.
Teacher alignment gate passed: False.
Mean alignment TV improvement over dependence only: 0.2761.

| New seed | Alignment gain over dependence only | Repair TV change | Missing teacher mass before | After | Sequence reconstruction after |
|---:|---:|---:|---:|---:|---:|
| 8 | -0.0020 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 9 | -0.0260 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 10 | 0.4826 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 11 | 0.4590 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 12 | 0.4669 | -0.0005 | 0.0000 | 0.0000 | 1.0000 |

Repair replay uses observed training sequences only and updates decoder parameters only.
All four arms receive identical repair updates. Known-seed replications: [0, 4, 7];
only the listed new student seeds determine follow-up gates. The teacher/domain is unchanged.
