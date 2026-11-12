# Causal feature-target mismatch

The historical alignment target comes from shifted causal teacher inputs. Its first-position target is constant, and its entire target array is identical for strings differing only in their final bit. That final bit retains 0.429325 conditional bits under the recorded teacher law. The codec instead receives the full sequence and must retain this distinction. A finite feature penalty therefore competes with reconstruction; it does not make reconstruction mathematically impossible.

This is a replayable structural diagnosis plus frozen-model measurements, not a causal intervention establishing why every alignment model fails. PCA truncation is a weak explanation in this particular domain: projecting and restoring teacher features changes A/B output-head predictions by only 2.3292e-6 mean KL nats.

| Objective | Encoder MSE | Quantized MSE | Quantized teacher-head KL |
|---|---:|---:|---:|
| reconstruction | 5.4162 | 5.6957 | 1.3978 |
| alignment | 1.2187 | 1.5170 | 0.5129 |
| dependency | 5.6759 | 6.0077 | 1.4155 |
| alignment_dependency | 1.3144 | 1.5447 | 0.4986 |

Values average five historical seeds per objective arm, weighted by population probability. The teacher-head diagnostic reverses PCA scaling/projection and applies the frozen teacher A/B head. It measures interface compatibility, not decoded generation quality. Improved feature matching does not establish reduced hard-code dependence.

Run `python examples/alignment_diagnosis.py` to regenerate results from committed `diagnostic_inputs.npz`; `inputs.json` records its hash and source checkpoint identities. The compact file contains frozen activations and teacher projection/head arrays, not trainable model checkpoints. `examples/export_research_inputs.py` records how these inputs were extracted when the local historical runs were available.

The current implementation and production alignment defaults are unchanged. Future controls should compare matched conditioning or post-token teacher features against no alignment, with fidelity and decoded distributions as primary outcomes.
