# Expert-aware session scheduling at batch 16 (16 of 32 sessions per step, v4 R0 traces)

LRU cache, step-level eviction; a session waiting >= 4 steps is forced in. fixed = the plain batch (16 sessions to their end, then the next 16); proxy = fewest new experts by the previous token's routing; oracle = by the next token's true routing (bound). Misses per token, mean union per step, total tok/s (cost model as cache_policy; int4 at the larger cache).

| domain | cache | fixed miss/tok | proxy | oracle | fixed union | proxy | oracle | fixed tok/s | proxy | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 3000 fp8 | 69.9 | 75.2 | 76.0 | 2607 | 2611 | 2181 | 37.4 | 34.8 | 34.4 |
| code | 6000 fp8 | 32.4 | 35.0 | 39.8 | 2607 | 2611 | 2169 | 80.6 | 74.8 | 65.7 |
| code | 6000 int4 | 32.4 | 35.0 | 39.8 | 2607 | 2611 | 2169 | 161.2 | 149.6 | 131.3 |
| math | 3000 fp8 | 105.2 | 114.0 | 103.7 | 3064 | 3066 | 2536 | 24.9 | 22.9 | 25.2 |
| math | 6000 fp8 | 61.5 | 67.3 | 62.8 | 3064 | 3064 | 2457 | 42.5 | 38.9 | 41.6 |
| math | 6000 int4 | 61.5 | 67.3 | 62.8 | 3064 | 3064 | 2457 | 85.0 | 77.7 | 83.2 |
| multilingual | 3000 fp8 | 91.2 | 97.5 | 92.6 | 2970 | 2949 | 2444 | 28.7 | 26.8 | 28.2 |
| multilingual | 6000 fp8 | 46.2 | 50.3 | 51.4 | 2970 | 2964 | 2433 | 56.6 | 52.0 | 50.9 |
| multilingual | 6000 int4 | 46.2 | 50.3 | 51.4 | 2970 | 2964 | 2433 | 113.1 | 104.0 | 101.7 |
| reasoning | 3000 fp8 | 96.8 | 107.7 | 100.2 | 3006 | 3011 | 2481 | 27.0 | 24.3 | 26.1 |
| reasoning | 6000 fp8 | 52.1 | 59.3 | 59.6 | 3006 | 3010 | 2499 | 50.2 | 44.1 | 43.9 |
| reasoning | 6000 int4 | 52.1 | 59.3 | 59.6 | 3006 | 3010 | 2499 | 100.5 | 88.1 | 87.7 |
| web_en | 3000 fp8 | 96.7 | 99.3 | 95.1 | 3000 | 2917 | 2480 | 27.0 | 26.3 | 27.5 |
| web_en | 6000 fp8 | 49.2 | 51.9 | 50.6 | 3000 | 2967 | 2479 | 53.2 | 50.4 | 51.7 |
| web_en | 6000 int4 | 49.2 | 51.9 | 50.6 | 3000 | 2967 | 2479 | 106.4 | 100.7 | 103.4 |
