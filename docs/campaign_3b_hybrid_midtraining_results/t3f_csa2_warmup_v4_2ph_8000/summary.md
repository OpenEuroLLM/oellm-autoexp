# Long-context attention probe (T1 / T2)

## NLL by position (held-out long documents)

| length | policy | mean | bins |
|---|---|---:|---|
| 64k | csa2:compress=4,topk_entries=512,window=512 | 2.0148 | 0k-4k 1.955, 4k-16k 1.981, 16k-32k 1.974, 32k-63k 2.055 |

## Retrieval accuracy (10-way)

| length | policy | passkey | kvret64 | vt |
|---|---|---:|---:|---:|
| 64k | csa2:compress=4,topk_entries=512,window=512 | 0.07 | 0.12 | 0.19 |
