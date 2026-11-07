# Long-context attention probe (T1 / T2)

## NLL by position (held-out long documents)

| length | policy | mean | bins |
|---|---|---:|---|
| 64k | dense | 1.9190 | 0k-4k 1.920, 4k-16k 1.876, 16k-32k 1.846, 32k-63k 1.972 |
| 64k | page_mean:topk=32 | 1.9070 | 0k-4k 1.920, 4k-16k 1.886, 16k-32k 1.856, 32k-63k 1.939 |
| 64k | quest:topk=32 | 1.9182 | 0k-4k 1.920, 4k-16k 1.889, 16k-32k 1.864, 32k-63k 1.956 |
| 64k | page_mean:topk=32,hier_group=16,hier_topk=8 | 1.9203 | 0k-4k 1.920, 4k-16k 1.887, 16k-32k 1.870, 32k-63k 1.958 |
| 64k | quest:topk=32,hier_group=16,hier_topk=8 | 1.9352 | 0k-4k 1.920, 4k-16k 1.896, 16k-32k 1.884, 32k-63k 1.977 |
| 64k | lsa:topk=32 | 1.9157 | 0k-4k 1.935, 4k-16k 1.903, 16k-32k 1.856, 32k-63k 1.948 |
| 64k | nsa:topk=32,select_window=512 | 1.9022 | 0k-4k 1.920, 4k-16k 1.884, 16k-32k 1.850, 32k-63k 1.933 |
| 128k | dense | 1.8987 | 0k-4k 1.931, 4k-16k 1.882, 16k-32k 1.799, 32k-64k 1.808, 64k-127k 1.970 |
| 128k | page_mean:topk=32 | 1.8428 | 0k-4k 1.931, 4k-16k 1.891, 16k-32k 1.794, 32k-64k 1.763, 64k-127k 1.881 |
| 128k | quest:topk=32 | 1.8492 | 0k-4k 1.932, 4k-16k 1.897, 16k-32k 1.804, 32k-64k 1.774, 64k-127k 1.884 |
| 128k | page_mean:topk=32,hier_group=16,hier_topk=8 | 1.8488 | 0k-4k 1.931, 4k-16k 1.893, 16k-32k 1.796, 32k-64k 1.768, 64k-127k 1.889 |
| 128k | quest:topk=32,hier_group=16,hier_topk=8 | 1.8641 | 0k-4k 1.932, 4k-16k 1.910, 16k-32k 1.824, 32k-64k 1.792, 64k-127k 1.898 |
| 128k | lsa:topk=32 | 1.8690 | 0k-4k 1.953, 4k-16k 1.913, 16k-32k 1.815, 32k-64k 1.785, 64k-127k 1.911 |
| 128k | nsa:topk=32,select_window=512 | 1.8412 | 0k-4k 1.932, 4k-16k 1.888, 16k-32k 1.791, 32k-64k 1.760, 64k-127k 1.880 |

## Retrieval accuracy (10-way)

| length | policy | passkey | kvret64 | vt |
|---|---|---:|---:|---:|
| 64k | dense | 1.00 | 0.33 | 0.00 |
| 64k | page_mean:topk=32 | 1.00 | 0.00 | 0.17 |
| 64k | quest:topk=32 | 1.00 | 0.33 | 0.17 |
| 64k | page_mean:topk=32,hier_group=16,hier_topk=8 | 0.67 | 0.00 | 0.17 |
| 64k | quest:topk=32,hier_group=16,hier_topk=8 | 1.00 | 0.17 | 0.17 |
| 64k | lsa:topk=32 | 1.00 | 0.33 | 0.33 |
| 64k | nsa:topk=32,select_window=512 | 1.00 | 0.00 | 0.33 |
| 128k | dense | 1.00 | 0.33 | 0.00 |
| 128k | page_mean:topk=32 | 1.00 | 0.00 | 0.17 |
| 128k | quest:topk=32 | 1.00 | 0.17 | 0.17 |
| 128k | page_mean:topk=32,hier_group=16,hier_topk=8 | 1.00 | 0.00 | 0.17 |
| 128k | quest:topk=32,hier_group=16,hier_topk=8 | 1.00 | 0.17 | 0.00 |
| 128k | lsa:topk=32 | 1.00 | 0.17 | 0.17 |
| 128k | nsa:topk=32,select_window=512 | 1.00 | 0.17 | 0.17 |
