# Research papers

Both papers are authored by Fabian Balensiefer. The November 12, 2026 research extension preserves the historical baseline and adds the same shared evidence to both papers:

- [Systems paper](../output/pdf/ncm_systems_paper.pdf), [editable source](ncm_systems_paper.tex): functional JAX/Flax contracts, historical engineering/CPU results, and current maintenance and dependency studies.
- [SCALE paper](../output/pdf/SCALE_research_paper.pdf), [editable source](SCALE_research_paper.tex): conditional compiler analysis, broader failure boundaries, empirical dependency controls, exact innovation baselines and the alignment-target diagnosis.

They describe shared experiments, not independent replications. The historical 20-model stability success remains valid within its domain. The later 360-run study finds failures under every quantizer condition; 480 training-only fits on frozen reference codecs demonstrate dependency-model improvements. No pretrained transfer, natural-language quality, total adaptation savings or TPU performance is established. Production defaults remain unchanged.

## Build and check

Use Python 3.10+ and Tectonic or pdfLaTeX with packages in `shared_preamble.tex`:

```sh
python paper/build_papers.py --build-dir /tmp/ncm-paper-build
python paper/generate_results.py --check
python paper/generate_recent_results.py --check
python paper/generate_bibliographies.py --check
python paper/audit_papers.py --build-dir /tmp/ncm-paper-build
```

The build regenerates evidence and bibliography inputs, checks references/logs, then copies final PDFs to `output/pdf`. Tectonic can use `--offline` after its package cache is populated. Render both complete PDFs with Poppler and inspect every page before publishing. Keep logs and page images outside the repository.

## Evidence and provenance

- `shared_experiments.tex` retains historical observations; `shared_recent_findings.tex` adds the new studies and mathematical controls. Both papers include them.
- `generate_results.py` validates historical evidence and the original 44-test record. Existing historical modules must match their recorded hashes after explicitly reversing only the optional maintenance callback hook. It does not assert that today's entire package is the old snapshot.
- `generate_recent_results.py` derives current tables from raw archived results, checks study counts and innovation/alignment invariants, and hashes its inputs in `generated/recent_results.json`.
- `references.json` supplies both generated citation subsets. Classical dependence trees and associative scan are credited; innovation controls are not claimed as novel algorithms.
- `validation.json` preserves the historical 44-test verification. `validation_current.json` records 62 passing tests and the public 480-fit replay for this extension.
- `source_manifest.sha256` identifies current implementation, tests and examples. `document_manifest.sha256` includes manuscripts, all reports, compressed replay inputs and both PDFs. Refresh with `python paper/update_manifests.py` after final build/review, then verify with `shasum -a 256 -c`.
- Historical protocol hashes and new study continuation records identify executed versions; later diagnostic fixes are not retroactively described as training changes. The float32 verification interruption and one missing timing remain documented.

Public replay does not require ignored checkpoints:

```sh
python examples/replay_dependency_results.py
python examples/alignment_diagnosis.py
python examples/domain_source_baselines.py
python examples/summarize_maintenance.py
python examples/summarize_domain_validation.py --from-reports
```

`examples/export_research_inputs.py` documents extraction of compact tables/activations from the original local runs. `examples/write_domain_report.py` and `examples/write_dependency_report.py` preserve report generation. Model checkpoints stay under ignored `runs/`; source laws and compressed public inputs support independent re-evaluation. The old text fixture logs remain in `reports/text-experiments` and show overfitting, not a successful held-out language result.

See [consistency audit](consistency_audit.md). No archival DOI is assigned.
