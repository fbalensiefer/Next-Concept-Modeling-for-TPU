# Local inference benchmark

Backend: cpu; device: cpu; float32.

These timings do not establish TPU performance or a quality-matched speedup.

| Scope | Model | Batch | Bytes/sequence | Median ms | p90 ms | Bytes/s | Compiler live-buffer bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| trained tiny fixture | ncm_padded | 1 | 8 | 0.127 | 0.132 | 63210 | 62968 |
| trained tiny fixture | ncm_actual_lengths | 1 | 8 | 0.102 | 0.136 | 78383 | 63208 |
| trained tiny fixture | ar_kv_cache | 1 | 8 | 0.101 | 0.105 | 79159 | 56616 |
| trained tiny fixture | ncm_padded | 16 | 8 | 0.489 | 0.509 | 261770 | 194008 |
| trained tiny fixture | ncm_actual_lengths | 16 | 8 | 0.403 | 0.429 | 317454 | 194248 |
| trained tiny fixture | ar_kv_cache | 16 | 8 | 0.937 | 1.030 | 136545 | 166896 |
| untrained shape check | ncm_padded | 1 | 128 | 2.933 | 3.031 | 43643 | 1424088 |
| untrained shape check | ncm_actual_lengths | 1 | 128 | 2.535 | 2.608 | 50489 | 1294152 |
| untrained shape check | ar_kv_cache | 1 | 128 | 3.320 | 3.404 | 38559 | 770632 |
| untrained shape check | ncm_padded | 16 | 128 | 26.877 | 27.645 | 76200 | 10870232 |
| untrained shape check | ncm_actual_lengths | 16 | 128 | 14.390 | 14.611 | 142319 | 8789320 |
| untrained shape check | ar_kv_cache | 16 | 128 | 39.588 | 39.802 | 51732 | 3831392 |
| untrained shape check | ncm_padded | 1 | 512 | 14.961 | 15.504 | 34223 | 11117016 |
| untrained shape check | ncm_actual_lengths | 1 | 512 | 10.316 | 10.488 | 49633 | 9811272 |
| untrained shape check | ar_kv_cache | 1 | 512 | 26.967 | 27.338 | 18986 | 1467976 |
| untrained shape check | ncm_padded | 16 | 512 | 173.887 | 178.206 | 47111 | 87142616 |
| untrained shape check | ncm_actual_lengths | 16 | 512 | 85.831 | 87.490 | 95444 | 81866056 |
| untrained shape check | ar_kv_cache | 16 | 512 | 299.951 | 302.559 | 27311 | 13503584 |

## Trained fixture sample diagnostics

These descriptive statistics do not establish semantic quality or matched quality.

| Model | Distinct fraction | Valid UTF-8 fraction | Byte-bigram TV to held-out |
|---|---:|---:|---:|
| ncm_padded | 0.578 | 1.000 | 0.5637175324675325 |
| ncm_actual_lengths | 0.609 | 1.000 | 0.5452516233766234 |
| ar_kv_cache | 0.328 | 0.984 | 0.4387175324675325 |

AR held-out next-byte CE: 3.3393 nats.
This is not comparable to NCM code cross-entropy.

## Model storage

| Component | Elements | Bytes |
|---|---:|---:|
| ncm_codec | 21328 | 85312 |
| ncm_prior | 4032 | 16128 |
| ncm_codebook_embeddings | 768 | 3072 |
| ncm_full_ema_state | 1584 | 6336 |
| ar | 11904 | 47616 |

Measured per-case peak device memory: unavailable. See report.json for compilation times,
raw latency observations, source hashes, parameter sizes, quality diagnostics and limitations.
