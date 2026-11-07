# Long-context attention probe (T1 / T2)

## NLL by position (held-out long documents)

| length | policy | mean | bins |
|---|---|---:|---|
| 64k | page_mean:topk=32,select_block=64,select_score=prev_max | 1.9438 | 0k-4k 1.921, 4k-16k 1.909, 16k-32k 1.903, 32k-63k 1.980 |
| 64k | page_mean:topk=32,select_block=32,select_score=prev_max | 1.9340 | 0k-4k 1.921, 4k-16k 1.907, 16k-32k 1.887, 32k-63k 1.970 |
| 64k | page_mean:topk=32,select_block=16,select_score=prev_max | 1.9243 | 0k-4k 1.920, 4k-16k 1.903, 16k-32k 1.879, 32k-63k 1.955 |
| 64k | page_mean:topk=32,select_block=16 | 1.9234 | 0k-4k 1.921, 4k-16k 1.900, 16k-32k 1.875, 32k-63k 1.957 |
| 64k | quest:topk=32,select_block=64,select_score=prev_max | 1.9467 | 0k-4k 1.922, 4k-16k 1.912, 16k-32k 1.901, 32k-63k 1.985 |
| 64k | oracle:topk=32,select_block=64,select_score=prev_max | 1.9262 | 0k-4k 1.922, 4k-16k 1.906, 16k-32k 1.878, 32k-63k 1.958 |
| 128k | page_mean:topk=32,select_block=64,select_score=prev_max | 1.8640 | 0k-4k 1.932, 4k-16k 1.910, 16k-32k 1.815, 32k-64k 1.786, 64k-127k 1.902 |
| 128k | page_mean:topk=32,select_block=32,select_score=prev_max | 1.8571 | 0k-4k 1.932, 4k-16k 1.909, 16k-32k 1.808, 32k-64k 1.777, 64k-127k 1.895 |
| 128k | page_mean:topk=32,select_block=16,select_score=prev_max | 1.8504 | 0k-4k 1.932, 4k-16k 1.906, 16k-32k 1.804, 32k-64k 1.769, 64k-127k 1.887 |
| 128k | page_mean:topk=32,select_block=16 | 1.8480 | 0k-4k 1.932, 4k-16k 1.901, 16k-32k 1.801, 32k-64k 1.767, 64k-127k 1.885 |
| 128k | quest:topk=32,select_block=64,select_score=prev_max | 1.8627 | 0k-4k 1.933, 4k-16k 1.918, 16k-32k 1.825, 32k-64k 1.788, 64k-127k 1.895 |
| 128k | oracle:topk=32,select_block=64,select_score=prev_max | 1.8458 | 0k-4k 1.933, 4k-16k 1.905, 16k-32k 1.803, 32k-64k 1.768, 64k-127k 1.879 |

## Retrieval accuracy (10-way)

| length | policy | passkey | kvret64 | vt |
|---|---|---:|---:|---:|
| 64k | page_mean:topk=32,select_block=64,select_score=prev_max | 1.00 | 0.17 | 0.17 |
| 64k | page_mean:topk=32,select_block=32,select_score=prev_max | 1.00 | 0.17 | 0.17 |
| 64k | page_mean:topk=32,select_block=16,select_score=prev_max | 1.00 | 0.17 | 0.33 |
| 64k | page_mean:topk=32,select_block=16 | 1.00 | 0.17 | 0.17 |
| 64k | quest:topk=32,select_block=64,select_score=prev_max | 1.00 | 0.17 | 0.00 |
| 64k | oracle:topk=32,select_block=64,select_score=prev_max | 1.00 | 0.50 | 0.17 |
| 128k | page_mean:topk=32,select_block=64,select_score=prev_max | 1.00 | 0.00 | 0.17 |
| 128k | page_mean:topk=32,select_block=32,select_score=prev_max | 1.00 | 0.00 | 0.17 |
| 128k | page_mean:topk=32,select_block=16,select_score=prev_max | 1.00 | 0.00 | 0.17 |
| 128k | page_mean:topk=32,select_block=16 | 1.00 | 0.00 | 0.17 |
| 128k | quest:topk=32,select_block=64,select_score=prev_max | 1.00 | 0.17 | 0.00 |
| 128k | oracle:topk=32,select_block=64,select_score=prev_max | 1.00 | 0.17 | 0.17 |
