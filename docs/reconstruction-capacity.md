# Training-duration and code-capacity follow-up

The first pilot left training duration and codec capacity confounded. This follow-up holds 128-byte document-safe windows, model dimensions, optimizer, batch size and seed-addressed training orders fixed. Its sole architecture intervention is 16 versus 64 codebook entries at every level, using three paired model seeds. Each fresh fit runs for 2,000 updates with diagnostics every 100 updates. The final checkpoint is the comparison endpoint; validation does not select a checkpoint or tune the protocol. Test reconstruction remains unevaluated.

Raw bytes and fixed byte pairs are exact controls. `ncm.representations.encode_pairs` packs two adjacent bytes into one integer; `decode_pairs` reverses it. Exhaustive tests verify all 65,536 pairs. These controls answer reconstruction questions only. Predicting pair categories requires accounting for the large output head and cannot be assumed fast.

Run after preparing the first pilot data:

```sh
cargo run --bin ncm-run -- reconstruction-capacity --output runs/reconstruction-capacity-v1
cargo run --bin ncm-run -- summarize-reconstruction-capacity --run runs/reconstruction-capacity-v1 --output reports/reconstruction-capacity
```

Use a fresh output directory. The protocol is written before fitting, and exact source copies are saved under the run directory. A failed incomplete directory is retained rather than silently overwritten. Every completed fit writes all its checkpoint metrics into the summary; the publication script requires six complete fits. The model runs remain sequential and costs include compilation, full-dataset diagnostic passes and checkpoint serialization.

Changing codebook size changes information capacity, lookup work and optimization behavior together. Nominal RVQ storage rises from 7 to 10.5 bits per byte, versus 8 for both exact controls. Report this explicitly: a reconstruction gain at higher capacity is not a compression or generation win. Compare full generation cost only after training suitable priors.

These are exploratory observations after the earlier pilot, with one book per split. Seed ranges do not quantify document-population uncertainty. The experiment separates two practical interventions but does not identify a unique cause of all remaining reconstruction error. A later confirmatory run needs more independent documents and a frozen application-specific tolerance.

[Recorded results](../reports/reconstruction-capacity/report.md)
