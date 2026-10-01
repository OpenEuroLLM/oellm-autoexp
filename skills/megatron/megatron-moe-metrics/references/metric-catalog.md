# Metric catalog

Use the same names and availability rules as `megatron-moe-health`:

- Router/load health measures assignment distribution, not functional usefulness.
- Weight and gradient RMS measure parameter movement and training signal.
- Routed-output RMS measures the routed path before shared-expert addition.
- Only paired masked validation can provide causal loss-impact evidence.

For full current-fork names, inspect `../../megatron-moe-health/references/metric-catalog.md` in the
repository source tree. After installation by copy, use the repository source as the canonical catalog.
