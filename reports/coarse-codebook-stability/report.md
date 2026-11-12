# Minimal SCALE mechanism pilot

Pilot did not meet the fixed gates; no SCALE benefit established.

| Seed | Arm | Byte reconstruction | Hard conditional TC (nats) | Exact output TV | Missing teacher support |
|---:|---|---:|---:|---:|---:|
| 4 | reconstruction | 1.0000 | 0.1395 | 0.1061 | 0.0000 |
| 4 | alignment | 1.0000 | 0.0959 | 0.0997 | 0.0000 |
| 4 | dependency | 1.0000 | 0.0879 | 0.1137 | 0.0000 |
| 4 | alignment_dependency | 1.0000 | 0.0959 | 0.1042 | 0.0000 |
| 8 | reconstruction | 1.0000 | 0.0959 | 0.1318 | 0.0000 |
| 8 | alignment | 0.9953 | 0.1158 | 0.1126 | 0.0000 |
| 8 | dependency | 1.0000 | 0.0959 | 0.1271 | 0.0000 |
| 8 | alignment_dependency | 0.9953 | 0.1158 | 0.1037 | 0.0000 |
| 10 | reconstruction | 1.0000 | 0.0959 | 0.1066 | 0.0000 |
| 10 | alignment | 1.0000 | 0.1278 | 0.1036 | 0.0000 |
| 10 | dependency | 1.0000 | 0.0959 | 0.1047 | 0.0000 |
| 10 | alignment_dependency | 1.0000 | 0.0959 | 0.1075 | 0.0000 |
| 13 | reconstruction | 1.0000 | 0.1395 | 0.1406 | 0.0000 |
| 13 | alignment | 1.0000 | 0.0879 | 0.1404 | 0.0000 |
| 13 | dependency | 1.0000 | 0.0879 | 0.1367 | 0.0000 |
| 13 | alignment_dependency | 0.9977 | 0.1158 | 0.1417 | 0.0000 |
| 14 | reconstruction | 1.0000 | 0.1291 | 0.1421 | 0.0000 |
| 14 | alignment | 1.0000 | 0.0959 | 0.1357 | 0.0000 |
| 14 | dependency | 1.0000 | 0.0959 | 0.1361 | 0.0000 |
| 14 | alignment_dependency | 1.0000 | 0.1278 | 0.1279 | 0.0000 |
| 15 | reconstruction | 1.0000 | 0.1291 | 0.1511 | 0.0000 |
| 15 | alignment | 1.0000 | 0.1396 | 0.1473 | 0.0000 |
| 15 | dependency | 1.0000 | 0.0959 | 0.1407 | 0.0000 |
| 15 | alignment_dependency | 1.0000 | 0.0879 | 0.1405 | 0.0000 |
| 16 | reconstruction | 1.0000 | 0.0879 | 0.1259 | 0.0000 |
| 16 | alignment | 1.0000 | 0.0879 | 0.1151 | 0.0000 |
| 16 | dependency | 1.0000 | 0.0879 | 0.1296 | 0.0000 |
| 16 | alignment_dependency | 1.0000 | 0.0879 | 0.1216 | 0.0000 |
| 17 | reconstruction | 1.0000 | 0.1201 | 0.1308 | 0.0000 |
| 17 | alignment | 1.0000 | 0.1210 | 0.1206 | 0.0000 |
| 17 | dependency | 1.0000 | 0.0879 | 0.1527 | 0.0000 |
| 17 | alignment_dependency | 1.0000 | 0.0879 | 0.1393 | 0.0000 |

Dependence-aware seeds with missing teacher support (infinite forward KL): [].
Mean paired TV improvement: -0.0014.
Mean paired hard-TC reduction: 0.0071 nats.
Fidelity gate passed: True.
Matched student FLOP estimates: True.

Gates and weights were fixed in protocol.json before training. This is a small synthetic
mechanism test, not evidence of pretrained-LLM conversion or general language quality.
See report.json for seed-level results, exact distributions, budgets and limitations.

## Four-arm and coverage follow-up

Confirmation student seeds: [13, 14, 15, 16, 17].
Coverage repair gate passed: True.
Teacher alignment gate passed: False.
Mean alignment TV improvement over dependence only: 0.0050.

| New seed | Alignment gain over dependence only | Repair TV change | Missing teacher mass before | After | Sequence reconstruction after |
|---:|---:|---:|---:|---:|---:|
| 13 | -0.0050 | 0.0000 | 0.0000 | 0.0000 | 0.9907 |
| 14 | 0.0082 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 15 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 16 | 0.0081 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 17 | 0.0134 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

Repair replay uses observed training sequences only and updates decoder parameters only.
All four arms receive identical repair updates. Known-seed replications: [4, 8, 10];
only the listed new student seeds determine follow-up gates. The teacher/domain is unchanged.

## Coarse-codebook stability

Stability gate passed: True.
Minimum population coarse perplexity: 1.9734.
Maximum post-annealing collapsed-step fraction: 0.0000.

| Seed | Arm | Coarse codes used | Population perplexity | Collapsed steps / monitored | Replacements |
|---:|---|---:|---:|---:|---:|
| 4 | reconstruction | 2 | 1.9955 | 0 / 200 | 1 |
| 4 | alignment | 2 | 1.9975 | 0 / 200 | 0 |
| 4 | dependency | 2 | 1.9734 | 0 / 200 | 1 |
| 4 | alignment_dependency | 2 | 1.9975 | 0 / 200 | 0 |
| 8 | reconstruction | 2 | 1.9975 | 0 / 200 | 1 |
| 8 | alignment | 2 | 1.9734 | 0 / 200 | 1 |
| 8 | dependency | 2 | 1.9975 | 0 / 200 | 1 |
| 8 | alignment_dependency | 2 | 1.9734 | 0 / 200 | 1 |
| 10 | reconstruction | 2 | 1.9975 | 0 / 200 | 2 |
| 10 | alignment | 2 | 1.9993 | 0 / 200 | 1 |
| 10 | dependency | 2 | 1.9975 | 0 / 200 | 2 |
| 10 | alignment_dependency | 2 | 1.9975 | 0 / 200 | 2 |
| 13 | reconstruction | 2 | 1.9955 | 0 / 200 | 1 |
| 13 | alignment | 2 | 1.9734 | 0 / 200 | 1 |
| 13 | dependency | 2 | 1.9734 | 0 / 200 | 2 |
| 13 | alignment_dependency | 2 | 1.9734 | 0 / 200 | 1 |
| 14 | reconstruction | 2 | 1.9982 | 0 / 200 | 0 |
| 14 | alignment | 2 | 1.9975 | 0 / 200 | 0 |
| 14 | dependency | 2 | 1.9975 | 0 / 200 | 1 |
| 14 | alignment_dependency | 2 | 1.9993 | 0 / 200 | 1 |
| 15 | reconstruction | 2 | 1.9982 | 0 / 200 | 1 |
| 15 | alignment | 2 | 1.9959 | 0 / 200 | 1 |
| 15 | dependency | 2 | 1.9975 | 0 / 200 | 1 |
| 15 | alignment_dependency | 2 | 1.9734 | 0 / 200 | 1 |
| 16 | reconstruction | 2 | 1.9734 | 0 / 200 | 0 |
| 16 | alignment | 2 | 1.9734 | 0 / 200 | 0 |
| 16 | dependency | 2 | 1.9734 | 0 / 200 | 0 |
| 16 | alignment_dependency | 2 | 1.9734 | 0 / 200 | 0 |
| 17 | reconstruction | 2 | 1.9867 | 0 / 200 | 2 |
| 17 | alignment | 2 | 2.0000 | 0 / 200 | 1 |
| 17 | dependency | 2 | 1.9734 | 0 / 200 | 1 |
| 17 | alignment_dependency | 2 | 1.9734 | 0 / 200 | 1 |

Full failure reasons and persisted schedules/usage counters are recorded in report.json.
