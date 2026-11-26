# MGSM-English probe (v4b@63000, 2026-11-26)

Question: the English MGSM CoT row scores 54 % where the *same 250 problems* score 79 % in
Spanish, 63 % in German and 61 % in French. Is that demonstration-driven (template mimicry) or a
genuine language-dependent weakness?

## Probe (one node, `config/evals/mgsm_probe_mn5.yaml`)

| cell | exact_match | n |
|---|---|---|
| `mgsm_native_cot_en` 8-shot | 56.40 % | 500 |
| `mgsm_native_cot_en` 4-shot | **59.00 %** | 500 |
| `mgsm_native_cot_en` 0-shot | **22.60 %** | 500 |
| `mgsm_direct_en` 8-shot | 32.40 % | 500 |
| `mgsm_direct_en` 0-shot | 29.60 % | 500 |

## Reading

- **Demonstrations help, they do not hurt**: 0-shot collapses to 22.6 %. Template mimicry as the
  cause of the English deficit is refuted.
- **No dose dependence**: 4-shot 59.0 % vs 8-shot 56.4 % — inside the ~3 pp standard error of 250
  problems.
- **CoT in the shots is what carries the score**: direct prompting is 32.4 % (8-shot) / 29.6 %
  (0-shot) against 56.4 % with CoT demonstrations.
- Therefore the English-vs-translation gap (~56 % vs ~79 %) is a **language-dependent capability
  difference of the model**, not a prompt artefact. A scan for "the dataset that taught the shot
  template" would be aimed at the wrong mechanism.

## Caveat and next discriminator

The remaining asymmetry is that the model is *better* at the translated problems than at the
English ones. One explanation is that the translations are closer to the model's multilingual
pretraining mixture, another is that the English items were removed by decontamination while their
translations were not. The second is testable with the existing `flag_decontam` machinery: scan the
training tree for MGSM English items and their translations and compare per-language hit rates.
