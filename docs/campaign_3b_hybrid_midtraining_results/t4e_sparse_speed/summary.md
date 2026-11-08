# Sparse-attention speed (one layer, one document, bf16, 16x256 q / 2 KV heads; NVIDIA H100)

| L | arm | fwd ms | fwd+bwd ms | vs dense | extra peak GB | page union / dense |
|---|---|---:|---:|---:|---:|---:|
| 16k | dense | 3.7 | 22.7 | 1.00x | 1.4 | nan |
| 16k | window | 2.3 | 23.2 | 1.02x | 1.4 | 0.076 |
| 16k | page_mean_flex | 68.1 | 163.1 | 7.18x | 1.4 | 1.000 |
| 16k | quest_flex | 37.6 | 96.7 | 4.25x | 1.4 | 0.636 |
| 16k | page_mean_b64_flex | 4.6 | 36.0 | 1.58x | 1.4 | 0.301 |
| 16k | quest_b64_flex | 4.8 | 33.5 | 1.47x | 1.4 | 0.301 |
| 16k | page_mean_hier_b64_flex | 5.2 | 35.8 | 1.58x | 1.4 | 0.301 |
| 16k | nsa_flex | 88.9 | 212.4 | 9.35x | 2.5 | nan |
| 16k | nsa_b64_flex | 26.1 | 85.8 | 3.78x | 2.5 | nan |
| 16k | csa2_b64_flex | 13.1 | 108.5 | 4.77x | 2.2 | nan |
| 16k | csa2_b64_kv16_flex | 13.0 | 106.7 | 4.70x | 2.2 | nan |
| 16k | lsa_b64_flex | 29.5 | 141.6 | 6.23x | 2.1 | nan |
| 16k | lsa_b64_kv16_flex | 11.5 | 103.2 | 4.54x | 2.1 | nan |
| 64k | dense | 55.8 | 321.1 | 1.00x | 5.5 | nan |
| 64k | window | 7.9 | 90.5 | 0.28x | 5.6 | 0.019 |
| 64k | page_mean_flex | 1239.7 | 2783.7 | 8.67x | 5.6 | 0.942 |
| 64k | quest_flex | 341.2 | 848.6 | 2.64x | 5.6 | 0.300 |
| 64k | page_mean_b64_flex | 18.7 | 150.8 | 0.47x | 5.6 | 0.080 |
| 64k | quest_b64_flex | 17.4 | 140.6 | 0.44x | 5.6 | 0.080 |
| 64k | page_mean_hier_b64_flex | 20.1 | 155.7 | 0.48x | 5.6 | 0.080 |
| 64k | nsa_flex | 1492.9 | 3183.4 | 9.91x | 10.0 | nan |
| 64k | nsa_b64_flex | 287.3 | 536.9 | 1.67x | 10.0 | nan |
| 64k | csa2_b64_flex | 104.8 | 556.4 | 1.73x | 8.9 | nan |
| 64k | csa2_b64_kv16_flex | 90.4 | 425.1 | 1.32x | 9.0 | nan |
| 64k | lsa_b64_flex | 152.1 | 507.8 | 1.58x | 8.3 | nan |
| 64k | lsa_b64_kv16_flex | 39.8 | 274.9 | 0.86x | 8.3 | nan |
