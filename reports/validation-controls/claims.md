# Claim ledger

git_commit: bbf82c63d738633b950e5daef5ca862e55f39016

| Claim | Evidence file | Status |
|---|---|---|
| Speculative exactness passes the frozen enumerable power test and rejects all four controls | power.json | supported |
| Hybrid coarse cache agrees with recomputation on the registered finite tests | tests.json | supported |
| AR-drafter control covers every registered draft/prompt/batch/seed case | drafter_comparison.csv | supported |
| General NCM drafter advantage over the distilled AR drafter | drafter_comparison.csv; drafter_paired_statistics.json | untested |
| Exact parameter-count matching was achieved | bpb_vs_train_flops.csv | refuted |
| NCM student beats distilled AR at matched exact BPB | latency_at_matched_bpb.csv | untested |
| Learned probe meets the frozen TC accuracy/disjointness gate | tc-probe/probe_results.json | refuted |
| Hybrid versus parallel removes >=30% of real-text TC | tc_probe_decision.md | untested |
| No TPU evidence has been produced | summary.json | supported |
| General compute savings at matched quality | bpb_vs_train_flops.csv; latency_at_matched_bpb.csv | untested |

Supported finite checks are not universal mathematical proofs. Adapter proposals
must match their reported q, and the teacher must be causal. No real-text TC
interpretation or strong-teacher speedup claim follows from this pilot.
