# Compatibility

| Compatibility key | Status | Notes |
| --- | --- | --- |
| `oellm-megatron-7be91731d-plus-moe-viability` | current | Source locations and metric names target the repository fork. |
| `upstream-unknown` | unsupported | Discover source paths and metric availability before interpreting values. |

Add a new explicit row when backporting the diagnostics to a later Megatron revision. Include changed
metric keys, source locations, process-group behavior, and tests; do not silently alias incompatible metrics.
