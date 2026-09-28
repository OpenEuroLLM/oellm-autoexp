# Complex-KDA Megatron scaling sweeps

The Megatron half of the [ComplexKDA](https://github.com/OpenEuroLLM/ComplexKDA)
scaling ladder (`lm_scaling/`), as autoexp sweeps instead of the per-cell
`submit_ladder.py` submitter.

| file | what | edited by |
|---|---|---|
| `sweep_spec.yaml` | sizes, arms, ladders (size \| gbs \| lr \| budgets), measured rates | hand |
| `ckda_base.yaml` | everything shared: data, optimizer, checkpointing, env | hand |
| `ckda_scaling_all.yaml` | every ladder in one sweep | generated |
| `ckda_ladder.yaml`, `ckda_ladder_medium.yaml`, `ckda_ladder_finegrained.yaml` | one ladder each | generated |
| `MODELS.md` | parameter match: params, delta vs dense, ffn per arm and size | generated |
| `BUDGET.md` | per chain: nodes, mbs, trunk/cooldown hours, GPU-h | generated |

Regenerate after editing the spec, and commit spec and outputs together:

```bash
.venv/bin/python scripts/korbi/gen_ckda_sweep.py
# optional: verify the analytic counts against ComplexKDA's fla-instantiated table
.venv/bin/python scripts/korbi/gen_ckda_sweep.py --check-ladder-matched <ComplexKDA>/lm_scaling/ladder_matched.json
```

## Shape of a sweep

`ARM x RECIPE`, a plain product:

- **ARM**: one entry per arm, holding its Megatron flags and per-size tables
  (`aux.ffn_by_size`, `aux.params_by_size`, `aux.tok_s`). Each non-dense arm's
  MLP width is matched to the dense `attn` arm's total parameter count, on a
  64 grid.
- **RECIPE**: one entry per chain (size | data | gbs | lr). The chain sets
  `aux.size` and geometry and runs:
  1. `stable`: constant LR to 0.8 × the largest budget. It saves persistently
     at every branch (`save_extra_steps`) and about every 3h.
  2. Per budget, `firstcd`: loads the stable with `--ckpt-step <branch>`,
     trains one step, saves into its own dir. It starts once the stable's
     `latest_checkpointed_iteration.txt` reaches the branch, which is written
     only after a save completes.
  3. Per budget, `contcd`: resumes firstcd's dir to the budget with a 20%
     linear decay and saves about 3 times, so it survives restarts. It then
     runs the 204,800-sequence held-out eval. This is the reported number.

Subsets: `--array-subset`, or submit one ladder's file.

## Adding an architecture

Add an arm to `sweep_spec.yaml` (`layout` for the counter, `megatron` flags,
`rate_as` for a throughput row) and add it to `run_arms`, then regenerate.
A mixer that is not yet counted also needs a function in `MIXER_PARAMS` of
the generator. Check it once against Megatron's startup line
`number of parameters on (tensor, pipeline) model parallel rank (0, 0): N`,
which should equal `aux.params_by_size[size]`.

## Requirements

- Megatron with `complex_kda` (submodule `feat/complex-kda`) **and** the
  `load_checkpoint` fix that makes `--ckpt-step` also select the
  distributed-optimizer state of a `torch`-format checkpoint. Without the fix,
  every firstcd silently resumes the trunk's latest state.
- `$CKDA_REPO`: a ComplexKDA checkout, whose vendored `fla` provides
  `fla.layers.complex_kda_layer` (it goes on `PYTHONPATH`).
- `$CKDA_HF_HOME`: the HF cache holding the GPT-NeoX-20B tokenizer.
