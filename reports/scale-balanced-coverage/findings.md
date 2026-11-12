# Coverage repair and four-arm findings

Balanced codec training passed the fixed coverage gate on new student seeds 8–12. The reliable teacher-alignment gate did not pass. These are finite-domain results from one tiny frozen teacher, not evidence of general LLM transfer.

## What changed

The first follow-up added dependence-only controls and decoder-only balanced replay. It fixed the original missing-output case but failed the stronger gates on seeds 3–7. Seed 4 still mapped 16 inputs to 10 encodings; the best deterministic sequence reconstruction from those encodings was only 89.80%. Its repaired decoder missed 6.63% of teacher probability.

The second protocol balanced codec reconstruction over observed training sequences, used empirical-frequency importance weights for soft dependence, and kept prior training empirical. Architecture, loss coefficients, update counts and gates stayed fixed. Known seeds 0, 4 and 7 were checks; only previously unused student seeds 8–12 determined the second protocol’s gates. All four arms received identical budgets and decoder replay.

## New-seed results after equal decoder replay

| Method | Mean exact output TV (lower is better) | Missing-support runs | Minimum sequence reconstruction |
|---|---:|---:|---:|
| reconstruction | 0.4257 | 0/5 | 99.07% |
| alignment | 0.2370 | 0/5 | 99.07% |
| dependency | 0.5229 | 0/5 | 100.00% |
| alignment_dependency | 0.2468 | 0/5 | 100.00% |

All five new combined-arm seeds retained 16 distinct encodings and had 100% sequence reconstruction and zero missing teacher support. This was already true before extra decoder replay, so the new coverage result comes from balanced codec training; replay retained coverage. Across all 32 balanced-protocol models, no teacher output support was missing.

The known failing combined-arm seed 4 recovered from 87.87% to 100% sequence reconstruction and from 6.63% missing teacher mass to zero. The teacher checkpoint and training/calibration/validation draws were identical between protocols.

Teacher alignment beat dependence alone on only three of five new seeds, below the fixed four-of-five requirement. Although its mean advantage was positive, the reliable-alignment gate failed. Alignment alone also had slightly lower mean output TV than alignment plus dependence. These findings do not justify a robust advantage for the full SCALE combination.

## Remaining failure mode

A post-run diagnostic found collapsed coarse codebooks in 4/5 dependence-only runs, 3/5 reconstruction runs, and 1/5 runs in each aligned arm. Collapse coincided with hard conditional TC near 1.002 nats and poor generation despite recoverable input codes. This diagnostic was not a training signal or a revised gate. The next controlled experiment should target coarse-codebook initialization and hard usage, with further unseen seeds, before increasing teacher size.

## Verification and artifacts

The full suite passed 34 tests, including exact mid-replay resume, canonical checkpoint hashes, importance weights and collision ceilings. Both studies archived their source and protocol before training. All compared student step FLOP estimates matched; this does not establish wall-time savings.

Detailed balanced results: [report](report.md) · [raw results](report.json) · [coarse-code diagnostic](coarse_code_diagnosis.json).
Earlier failed study: [report](../scale-coverage-followup/report.md).
These two studies use different new-student-seed sets; their overall mean errors are not a causal before/after quality comparison. The same-seed failure and exact coverage checks provide the targeted repair evidence.
