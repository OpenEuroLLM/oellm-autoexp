# Finding schema

Emit JSON or YAML with this shape:

```yaml
schema_version: 1
run_context: /absolute/path/run-context.json
findings:
  - category: routing | parameters | learning | contribution | causal_effect
    state: healthy | imbalanced | starved | static | collapsed | weak_learning | small_contribution | functionally_silent | insufficient_evidence
    layer: 0 # optional
    expert: 0 # optional
    confidence: low | medium | high
    evidence:
      - metric: moe/example
        value: 0.0
        step_range: [100, 200]
        interpretation: short, bounded claim
    alternatives: [possible competing explanations]
    recommended_next_action: read-only next check or explicitly requested action
```

Use `insufficient_evidence` whenever required metrics are absent. Use `functionally_silent` only
when paired masked validation shows a near-zero loss effect after numerical tolerance is considered.
