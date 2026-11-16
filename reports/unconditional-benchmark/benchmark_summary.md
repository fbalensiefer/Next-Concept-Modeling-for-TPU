# Bounded unconditional generation benchmark

All candidates use non-overlapping 512-byte windows from distinct Gutenberg books in train/validation/test. There is one book per split; window-level uncertainty is descriptive and clustered within the single test book.

| Candidate | Held-out BPB/bound | Bigram JSD | Exact codec rate | Total ms/sequence | Sequential rounds |
|---|---:|---:|---:|---:|---:|
| raw_byte_cached_ar | 4.9298 | 0.5547 | 1.0000 | 3.993 | 512 |
| xor_reversible_ar | 6.3771 | 1.0000 | 1.0000 | 3.910 | 512 |
| rvq_level_parallel | 7.2336 | 0.2799 | 0.0000 | 5.628 | 3 |
| rvq_chow_liu | 9.9206 | 0.3078 | 0.0000 | 0.325 | 3 |
| rvq_full_ar | 7.5959 | 0.2534 | 0.0000 | 6.872 | 896 |

Frozen RVQ reconstruction exact-window rate: 0.000000; byte accuracy: 0.838956.

Mathematically, the RVQ bound is (NLL of level-major codes under the fitted prior + frozen decoder cross-entropy on the held-out bytes) / (byte count × ln 2). It upper-bounds the codec model's held-out byte NLL; it includes both prior and reconstruction error. Raw and reversible candidates report their fitted autoregressive byte-chain NLL in bits/byte, with exact reconstruction by the bijective transform. The codec term breakdown is in JSON.

Uncertainty is summarized over 66 non-overlapping test windows from one held-out book. Window standard errors describe this book only and do not estimate between-document variation.

Stop rule: `halt_quantizer_expansion_and_pivot`. No RVQ candidate strictly dominates both lossless controls on held-out BPB, sample bigram JSD, and complete generation latency within the specified measurement margins. Lowest-BPB RVQ candidate=rvq_level_parallel (BPB/bound=7.2336, bigram JSD=0.2799, total=5.628 ms/sequence); raw control=(4.9298, 0.5547, 3.993); reversible control=(6.3771, 1.0000, 3.910). Frozen codec exact sequence reconstruction=0, byte accuracy=0.838956; within-level Chow-Liu gain=-2.3187 bits/byte; level count=3, full-AR code draws=896, test windows=66.

The Chow–Liu tree is a within-level diagnostic and omits earlier-level conditioning. Equal update counts do not imply equal FLOPs; training time and parameter counts are recorded per candidate.
