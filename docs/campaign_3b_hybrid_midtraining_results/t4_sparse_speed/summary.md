# Sparse-attention speed (one layer, one document, bf16, 16x256 q / 2 KV heads; NVIDIA H100)

| L | arm | fwd ms | fwd+bwd ms | vs dense | extra peak GB |
|---|---|---:|---:|---:|---:|
| 16k | dense | 1921.4 | 22.9 | 1.00x | 1.4 |
| 16k | window | 922.6 | 32.5 | 1.42x | 1.4 |
| 16k | page_mean_flex | 65.2 | 164.8 | 7.21x | 1.4 |
| 16k | quest_flex | 40.2 | 107.7 | 4.71x | 1.4 |
| 16k | page_mean_hier_flex | 66.2 | 165.7 | 7.25x | 1.4 |
| 16k | page_mean_gathered | 352.3 | 1383.8 | 60.54x | 7.3 |
| 16k | nsa | 447.2 | 1779.3 | 77.84x | 7.6 |
| 16k | csa2 | 349.5 | 986.0 | 43.14x | 3.5 |
| 16k | lsa | 535.3 | 2092.5 | 91.54x | 11.0 |
| 64k | dense | 66.9 | 328.2 | 1.00x | 5.5 |
| 64k | window | 936.4 | 128.1 | 0.39x | 5.6 |
| 64k | page_mean_flex | 1142.6 | 2734.7 | 8.33x | 5.6 |
| 64k | quest_flex | 326.8 | 845.6 | 2.58x | 5.6 |
| 64k | page_mean_hier_flex | 1151.2 | 2747.8 | 8.37x | 5.6 |
| 64k | page_mean_gathered | 1421.5 | 5743.9 | 17.50x | 9.3 |
| 64k | nsa | 1952.2 | 8331.1 | 25.38x | 10.5 |
| 64k | csa2 | 3635.7 | 6243.8 | 19.02x | 11.0 |
| 64k | lsa | 2253.2 | 8982.3 | 27.37x | 16.1 |
