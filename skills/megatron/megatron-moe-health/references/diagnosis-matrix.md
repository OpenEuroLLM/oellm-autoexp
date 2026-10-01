# Diagnosis matrix

| Evidence pattern | Bounded interpretation | Next check |
| --- | --- | --- |
| Low dispatched entropy, rising zero-load count | Imbalanced or starved routing | Router score entropy and logit spread |
| High selected load, high dropped fraction | Capacity pressure, not necessarily poor preference | Capacity configuration and dispatched counts |
| Healthy load, low gradient RMS | Work assigned but weak learning signal | Update-to-weight ratio when available |
| Low relative weight RMS, weak gradients | Possible expert collapse | Persistent load history and optimizer settings |
| Healthy load, low routed/output ratio | Routed path is numerically small | Paired masked validation |
| Near-zero masked NLL delta on paired batches | Functionally silent routed layer | Shared-expert dominance and reproducibility check |

Never infer functional silence from routing or RMS ratios alone. Compare trends with nearby layers and
state architecture-specific alternatives such as shared-expert dominance or deliberate sparsity.
