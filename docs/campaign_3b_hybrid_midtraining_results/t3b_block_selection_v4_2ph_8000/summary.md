# Long-context attention probe (T1 / T2)

## NLL by position (held-out long documents)

| length | policy | mean | bins |
|---|---|---:|---|
| 64k | page_mean:topk=32,select_block=64 | 1.9391 | 0k-4k 1.922, 4k-16k 1.908, 16k-32k 1.894, 32k-63k 1.975 |
| 64k | quest:topk=32,select_block=64 | 1.9444 | 0k-4k 1.922, 4k-16k 1.911, 16k-32k 1.902, 32k-63k 1.981 |
| 64k | page_mean:topk=32,hier_group=16,hier_topk=8,select_block=64 | 1.9428 | 0k-4k 1.922, 4k-16k 1.908, 16k-32k 1.898, 32k-63k 1.981 |
| 64k | oracle:topk=32,select_block=64 | 1.9324 | 0k-4k 1.922, 4k-16k 1.907, 16k-32k 1.887, 32k-63k 1.966 |
| 64k | oracle:topk=32 | 1.9097 | 0k-4k 1.920, 4k-16k 1.881, 16k-32k 1.844, 32k-63k 1.952 |
| 128k | page_mean:topk=32,select_block=64 | 1.8607 | 0k-4k 1.932, 4k-16k 1.907, 16k-32k 1.810, 32k-64k 1.782, 64k-127k 1.900 |
| 128k | quest:topk=32,select_block=64 | 1.8605 | 0k-4k 1.933, 4k-16k 1.918, 16k-32k 1.822, 32k-64k 1.785, 64k-127k 1.893 |
| 128k | page_mean:topk=32,hier_group=16,hier_topk=8,select_block=64 | 1.8607 | 0k-4k 1.932, 4k-16k 1.907, 16k-32k 1.811, 32k-64k 1.781, 64k-127k 1.900 |
| 128k | oracle:topk=32,select_block=64 | 1.8541 | 0k-4k 1.931, 4k-16k 1.905, 16k-32k 1.808, 32k-64k 1.777, 64k-127k 1.890 |
| 128k | oracle:topk=32 | 1.8714 | 0k-4k 1.931, 4k-16k 1.885, 16k-32k 1.792, 32k-64k 1.781, 64k-127k 1.930 |

## Retrieval accuracy (10-way)

| length | policy | passkey | kvret64 | vt |
|---|---|---:|---:|---:|
| 64k | page_mean:topk=32,select_block=64 | 1.00 | 0.00 | 0.33 |
| 64k | quest:topk=32,select_block=64 | 1.00 | 0.17 | 0.00 |
| 64k | page_mean:topk=32,hier_group=16,hier_topk=8,select_block=64 | 0.67 | 0.00 | 0.33 |
| 64k | oracle:topk=32,select_block=64 | 1.00 | 0.33 | 0.50 |
| 64k | oracle:topk=32 | 1.00 | 0.67 | 0.17 |
| 128k | page_mean:topk=32,select_block=64 | 1.00 | 0.00 | 0.17 |
| 128k | quest:topk=32,select_block=64 | 1.00 | 0.17 | 0.00 |
| 128k | page_mean:topk=32,hier_group=16,hier_topk=8,select_block=64 | 1.00 | 0.00 | 0.33 |
| 128k | oracle:topk=32,select_block=64 | 1.00 | 0.33 | 0.00 |
| 128k | oracle:topk=32 | 1.00 | 0.83 | 0.33 |
