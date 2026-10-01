# Metric catalog for the current fork

## Routing/load health

- `moe/dispatched_expert_load_entropy_layer_N`: normalized distributional balance after dispatch.
- `moe/expert_dead_count`: number of dispatched experts with no work.
- `moe/selected_zero_load_expert_layer_slots`: layer/expert slots never selected before capacity.
- `moe/selected_near_dead_expert_layer_slots`: selected load below 10% of layer mean.
- `moe/persistent_selected_near_dead_expert_layer_slots_100`: near-dead streaks at 100 windows.
- `moe/dropped_assignment_frac_max`: capacity/drop pressure.
- `moe/router_mean_max_prob_layer_N`, `moe/router_mean_score_entropy_layer_N`, and
  `moe/expert_logit_spread_layer_N`: router score concentration and preference spread.

## Emitted expert viability metrics

- `moe/expert_weight_rms_{median,p10,min}_layer_N`
- `moe/expert_weight_rms_relative_to_init_median_layer_N`
- `moe/expert_weight_collapsed_frac_layer_N`
- `moe/expert_grad_rms_median_layer_N`
- `moe/routed_expert_output_rms_layer_N`
- `moe/routed_expert_output_to_input_rms_layer_N`
- `moe/routed_expert_output_to_layer_output_rms_layer_N`

## Planned, not observed

`expert_adaptive_update_rms`, `expert_decay_update_rms`, `expert_update_to_weight_ratio`, and
`masked_layer_{nll,ppl}_delta` require optimizer-update and paired-validation instrumentation.
Do not represent them as available merely because a dashboard reference contains them.
