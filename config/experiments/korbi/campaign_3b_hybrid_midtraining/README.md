# campaign_3b_hybrid_midtraining

Configs for the inference-efficiency campaign that follows `campaign_3b`. The target is 1M+ token context
on consumer hardware at batch 16. The campaign covers:

- expert prediction and offload;
- sparse attention with KV offload;
- speculative decoding;
- the midtraining stage of ambitious v4.

The plan, the ablation IDs (R0-R4, M1-M2, N1, S0-S4, B4) and the log live in
`docs/campaign_3b_hybrid_midtraining.md`. Name each config after its ablation ID, e.g. `r1_missdrop_s3_mn5.yaml`.
