# Research papers

Both papers are authored by Fabian Balensiefer and revised November 12, 2026:

- [Systems paper](../output/pdf/ncm_systems_paper.pdf): [editable source](ncm_systems_paper.tex). Adds the completed data/checkpoint/prior/generation milestones, CPU inference measurements, coverage controls and coarse-codebook maintenance to the functional JAX/Flax description.
- [SCALE paper](../output/pdf/SCALE_research_paper.pdf): [editable source](SCALE_research_paper.tex), newly supplied for the formerly PDF-only manuscript. Retains the conditional compiler analysis and adds the completed local mechanism studies. The full pretrained, cross-size compiler remains proposed.

The papers describe the same experiments, not independent replications. All 20 newly reserved-seed models passed the final hard-stability/coverage gate; the separate reliable teacher-alignment gate failed. Neither paper claims pretrained transfer, competitive language quality, total adaptation savings or TPU performance.

## Build and check

Use Python 3.10+ and either Tectonic or a standard pdfLaTeX installation with the packages listed in `shared_preamble.tex`. No BibTeX, proprietary fonts or external raster figures are needed. Run from the repository root:

```sh
python paper/build_papers.py
```

This regenerates report-derived values and bibliography subsets, compiles both papers until references settle (three passes for pdfLaTeX), checks final logs, and copies the PDFs to `output/pdf/`. To retain logs/intermediates, add `--build-dir /tmp/ncm-paper-build`. To select an engine, add `--engine tectonic` or `--engine pdflatex`. Tectonic's `--offline` option requires previously cached packages and fonts.

Check committed generated inputs without changing them:

```sh
python paper/generate_results.py --check
python paper/generate_bibliographies.py --check
python paper/audit_papers.py
# After a build with retained logs:
python paper/audit_papers.py --build-dir /tmp/ncm-paper-build
```

After building, render **both complete PDFs** with Poppler (`pdftoppm -png -r 100 ...`) and inspect every page, especially equations, tables, the systems diagram and references. Intermediate files belong outside the repository; only final PDFs are published.

## Shared sources and evidence

- `shared_notation.tex`, `shared_metrics.tex` and `shared_experiments.tex` keep terminology, definitions, experimental settings and interpretation identical.
- `generate_results.py` calculates cohort-specific means, latency tables and stability summaries directly from the five committed reports. It verifies implementation hashes, the recorded 44-test result, final FLOP parity/support and historical text-run identities. `generated/observed_results.json` lists input hashes and the underlying values.
- `references.json` is the common bibliography registry. `generate_bibliographies.py` emits only each paper's cited subset. Shared citations therefore cannot silently acquire different metadata. The Wen et al. entry uses the revised four-author title/version; K-means++ is explicitly credited.
- `validation.json` identifies the **recorded** November 12 implementation verification: 44 passed, zero failed. The 44-case inventory was recollected during this documentation revision; training was not rerun. `validation_2026-11-11.json` preserves the original 11-test record as history.
- `source_manifest.sha256` identifies implementation, tests, fixtures and package configuration at baseline `1d600d7f4598ddb7e2808b29ee8f6eefe01cfb78`. Model code was not changed by this manuscript revision.
- `../reports/text-experiments/` archives the existing 500-step codec/prior logs and configurations, with checkpoint identities independently matched to the committed CPU benchmark. These are historical records, not new runs. Fixture validation CE worsened despite perfect training reconstruction.
- `document_manifest.sha256` hashes manuscript sources, shared inputs, generated evidence, audit notes, historical logs and final PDFs. Run `python paper/update_manifests.py` only after building and visual review; verify from the repository root with `shasum -a 256 -c paper/document_manifest.sha256`.

See [consistency audit and changelog](consistency_audit.md) for the resolved discrepancies, evidence boundaries and verification record. The public repository is a mutable development location; no archival DOI has been assigned.
