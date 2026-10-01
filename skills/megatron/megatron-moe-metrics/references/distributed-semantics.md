# Distributed semantics

| Parallel mode | Metric implication |
| --- | --- |
| TP / CP | Sum numerator and denominator shards before calculating RMS or ratios. |
| PP | Different physical layers live on different stages; sum layer slots without duplicating ownership. |
| DP | Average replicated statistics after sums; ratios cancel constant replication only when both terms match. |
| EP | Experts are partitioned by rank; place local values at global expert indices before EP reduction. |

Reduce sufficient statistics, not finalized non-linear values. For RMS, reduce squared sums and element
counts, then compute `sqrt(sum_sq / count)`. For ratios, reduce both components before division.
