# Sparse-attention speed (one layer, one document, bf16, 16x256 q / 2 KV heads; NVIDIA H100)

| L | arm | fwd ms | fwd+bwd ms | vs dense | extra peak GB | page union / dense |
|---|---|---:|---:|---:|---:|---:|
| 16k | dense | 3.6 | 22.7 | 1.00x | 1.4 | nan |
| 16k | window | 6.1 | 32.0 | 1.41x | 1.4 | 0.076 |
| 16k | page_mean_flex | 65.3 | 165.5 | 7.29x | 1.4 | 1.000 |
| 16k | quest_flex | 40.5 | 108.2 | 4.76x | 1.4 | 0.636 |
| 16k | page_mean_b64_flex | 20.2 | 62.5 | 2.75x | 1.4 | 0.301 |
| 16k | quest_b64_flex | 20.3 | 61.3 | 2.70x | 1.4 | 0.301 |
| 16k | page_mean_hier_b64_flex | 21.6 | 65.0 | 2.86x | 1.4 | 0.301 |
| 64k | dense | 55.8 | 324.6 | 1.00x | 5.5 | nan |
| 64k | window | 25.0 | 131.1 | 0.40x | 5.6 | 0.019 |
| 64k | page_mean_flex | 1147.5 | 2750.8 | 8.47x | 5.6 | 0.942 |
| 64k | quest_flex | 328.1 | 849.1 | 2.62x | 5.6 | 0.300 |
| 64k | page_mean_b64_flex | 102.2 | 294.9 | 0.91x | 5.6 | 0.080 |
| 64k | quest_b64_flex | 94.4 | 272.3 | 0.84x | 5.6 | 0.080 |
| 64k | page_mean_hier_b64_flex | 109.3 | 302.0 | 0.93x | 5.6 | 0.080 |
