# Cross-paper consistency audit and revision changelog

## Current research extension

The current revision preserves the historical record below and adds shared maintenance, 360-run broader-source and 480-fit dependency evidence, exact invertible innovations, and a replayable causal-target diagnosis. Current source manifests cover experimental additions; the original production defaults remain unchanged. Current validation reports 62 passing tests, separately from the historical 44. Both papers use generated report-derived tables and the same bibliography registry. Compressed codec/activation inputs allow replay without ignored checkpoints. Numeric interruption, gate limitations, source-law versus teacher-law targets, finite-domain scope and non-independent fit counts remain explicit. No new language, transfer or TPU success is asserted.

## Historical manuscript revision

Revision date: November 12, 2026. Author: Fabian Balensiefer.
Implementation/evidence baseline: `1d600d7f4598ddb7e2808b29ee8f6eefe01cfb78`.
This is a manuscript and evidence-archival revision; model code and experimental results were not changed or rerun.

## Changes by paper

**Systems paper.** Updated the abstract, introduction, methods, results, discussion, conclusion and future work. Added the completed byte-data pipeline, atomic resumable checkpoints, frozen-codec prior, seeded categorical/top-k generation, actual-length/sparse CPU inference, four-arm coverage controls and coarse-codebook maintenance. Documented K-means++ warmup, hard-usage EMA replacement, temperature/commitment schedules, extra entropy and the separate centroid optimizer. Replaced obsolete 11-test/future-work claims with the recorded 44-case suite and current implementation boundaries. Updated the prior diagram, tables, reproduction map and checkpoint transition.

**SCALE paper.** Added editable LaTeX for the formerly PDF-only artifact. Retained the conditional invariant-subspace transfer result, local logit bound, known conditional-TC factorization identity, finite-capacity limitation, calibration counterexample, six-stage architecture and falsification plan. Added the fixed-codec collision ceiling used in the completed failure diagnosis. Integrated the one-teacher finite-domain pilot, D-only controls, decoder-replay failure, balanced-coverage repair and maintained-codebook results. Recast the abstract, introduction, results, discussion, conclusion and next-stage plan around completed mechanisms while preserving the full pretrained compiler as proposed work.

## Discrepancies resolved

| Topic | Resolution in both papers |
| --- | --- |
| Evidence age and tests | Replaced the original November 11 / 11-test systems snapshot with implementation `1d600d7` and its recorded 44 passing cases. Preserved the old record as `validation_2026-11-11.json`; recollected 44 cases during revision. |
| Implemented versus proposed | Data loading, atomic local resume, frozen-codec prior, categorical sampling, CPU optimizations, exact finite-domain dependence and maintenance are completed. External pretrained/cross-size transfer, operator transplantation, a complete adaptive compiler, TPU/multi-host performance, prompt/length/EOS remain open. Rust launches packaged workloads and builds papers; model computation remains in JAX. |
| Teacher and data | One locally trained width-16 teacher; 16 possible length-4 A/B inputs; 512/512/256 draws with repeated support. Teacher and sampled-split hashes match across all four protocols. These are not unseen-domain or external-LLM evaluations. The separate text fixture contains 26/13 bytes. |
| Terminology and symbols | Shared notation uses W for token vocabulary, V for codebook size, D for code dimension, H/J for prior width/depth, I for random codes, C/S for EMA statistics, a for optimizer step and varphi for runtime phase. R_s denotes a canonical injection; J_s denotes the compiler audit certificate. |
| Metrics | Shared exact TV includes an outside-domain bin. Missing teacher support makes forward KL infinite (stored as null plus missing mass). Byte/sequence reconstruction, soft/hard TC, population/minibatch perplexity and teacher-forced/free-prefix CE are distinguished. |
| Early positive result versus later controls | The initial three-arm pilot lacked D-only controls. Later reliable alignment gates failed. The final package won 4/5 AD-versus-D comparisons but improved mean TV by only 0.00498, below 0.02. Positive stability is not reported as a positive alignment result. |
| Stability and reconstruction | All 20 new-seed models passed hard-stability/coverage gates. Minimum population coarse perplexity was 1.9734/2, with zero collapsed observations in each 200-step post-warmup window. AD minimum sequence reconstruction was 99.07%, not an all-arm claim of 100%. |
| Seed and replay accounting | Tables separate known seed checks from newly reserved evaluation seeds. Initial pilot: nine models and no replay. Each follow-up: 32 models, 20 newly reserved-seed models, and 200 replay updates. Different protocols are not treated as a causal before/after comparison. |
| Learning-rate units | Coarse scale 0.25 means a quarter of the EMA centroid displacement through a separate SGD group. It is not 0.25 times neural Adam LR. Fine EMA and neural rates remain unchanged. |
| FLOP parity | Shared maintenance and common objective graphs operate in all arms. Final per-step compiler estimates match: codec 10,564,359; prior 6,996,102; replay 5,837,030. These are not total measured work or parity across different protocols. |
| CPU performance | Shared table reports synchronized batch-16 medians. Length 128/512 models are untrained shape checks, trained toy parameter budgets are unequal, and compiler buffer estimates are not measured peak memory. No quality-matched or TPU speedup is claimed. |
| Text reconstruction | Archived historical logs show perfect training-byte reconstruction but validation CE increasing from 1.949 to 3.550 nats. This is explicitly reported as overfitting. Checkpoint identities match the existing CPU benchmark. |
| Theory | Transfer requires compatible invariant operators. The logit bound is local, not a certified hierarchical free-running bound. The conditional-TC identity is credited as known; finite calibration limits black-box identification, not all full-weight algorithms. |
| Citations and cross-references | Both companion citations describe the revised scope and shared experiments, not independent replication. One registry supplies shared citation metadata. The Wen et al. reference now uses the revised v2 title and four-author list, including Zhouchen Lin; K-means++ is credited to Arthur and Vassilvitskii. |

## Verification record

- `generate_results.py --check`: shared tables and evidence match the committed reports, including implementation hashes, teacher/split identities, recorded test outcome, final support/FLOP checks and archived historical text identities.
- `generate_bibliographies.py --check`: 14 systems citations and 21 SCALE citations resolve from one registry.
- `audit_papers.py --build-dir /tmp/ncm-paper-build`: shared sources, unique labels, internal references, reciprocal citations and implementation manifest pass. Systems: 24 labels; SCALE: nine labels.
- Both manuscripts compiled successfully with Tectonic; final logs contain no undefined citations/references, missing glyphs or overfull boxes. The normal XeTeX notice that UTF-8 `inputenc` is ignored is harmless.
- Both final A4 PDFs were rendered with Poppler and all pages visually inspected: systems 17 pages; SCALE 12 pages. Equations, the prior figure, result tables, bibliography links and page boundaries were reviewed; no clipping or overlapping content was found.
- The current test inventory was recollected: 44 cases. The recorded 44-pass model suite was not rerun for documentation-only changes. No TPU run, new training or additional empirical success is implied.
- `git diff --check` and the implementation/document hash checks pass. Generated PDFs and their editable inputs are committed together.

The papers repeat one shared empirical section deliberately to prevent incompatible summaries. The studies are locally specified and gated, not externally preregistered. Real-corpus generalization, maintenance-component attribution and reliable teacher alignment remain research questions.

## Current publication verification

The extension suite passed 62 tests. All 480 dependency fits replayed with 1e-12 tolerances from the compact tables. Five public replay/summary commands passed in a separate directory containing no runs folder or trained checkpoints. Both updated PDFs compiled successfully: systems 20 pages, SCALE 15 pages. All 35 pages were rendered and visually reviewed, with changed historical/current wording rechecked at full size. Final logs have no undefined references/citations, missing glyphs or overfull boxes. Historical records were preserved rather than relabeled as current training.

## SCALE editorial revision

Removed the entire availability/reproduction/authorship appendix from SCALE and replaced its abstract with a shorter research-focused account of the question, theoretical limits, empirical findings and open transfer objective. Author attribution remains in the title block. The substantive methods, results and conclusions are unchanged. Reproduction information remains in this directory and the repository documentation.

The revised SCALE PDF has 14 pages; every page was rendered and reviewed. Citation/reference and final-log checks passed. The audit now checks the shared alignment result instead of requiring a particular sentence in the abstract.
