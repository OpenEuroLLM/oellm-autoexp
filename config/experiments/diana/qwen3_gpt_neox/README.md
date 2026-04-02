### Qwen-like architecture

Scripts that reproduce exactly the dense monolingual experiments of Niccolo, where the model architecture is adjusted to Qwen-like style ([Qwen3 + GPT-OSS tokenizer](https://docs.google.com/spreadsheets/d/129mvSQ2K3m_hLaH2zk8R2Zr1-RsFyhTphXyN_ErSWFs/edit?gid=1329178961#gid=1329178961)) Tab. The configuration reuses as much as possible the long stable runs to preseve compute and decays at target token budgets. Thus most hyperparameters are optimal and a few almost optimal based on the validation loss results [here](https://github.com/OpenEuroLLM/dense_english_scaling_results/tree/main/fig/loss_valid/grid).

`dense_qwen_150M_gpt_neox`

- Group 1: GBSZ 32, LR 0.001, stable at 12BT (91,553 iters), decay at 6BT and 12BT
- Group 2: GBSZ 128, LR 0.002, stable at 80BT (152,588 iters), decay at 20BT and 80BT
- Group 3: GBSZ 256, LR 0.002, stable at 300BT (286,103 iters), decay at 120BT and 300BT

`dense_qwen_300M_gpt_neox`

- Group 1: GBSZ 64, LR 0.0005, stable at 30BT (114,441 iters), decay at 6BT, 12BT, 20BT and 30BT
- Group 2: GBSZ 256, LR 0.002, stable at 120BT (114,441 iters), decay at 50BT, 80BT, and 120BT
- Group 3: GBSZ 512, LR 0.002, stable at 300BT (143,052 iters), decay at 200BT and 300BT

`dense_qwen_600M_gpt_neox`

- Group 1: GBSZ 64, LR 0.001, stable at 6BT (22,889 iters), decay at 6BT
- Group 2: GBSZ 128, LR 0.001, stable at 80BT (152,588 iters), decay at 12BT, 20BT, 30BT, 50BT, and 80BT
- Group 3: GBSZ 512, LR 0.002, stable at 300BT (143,052 iters), decay at 120BT, 200BT and 300BT

`dense_qwen_1B_gpt_neox`

- Group 1: GBSZ 128, LR 0.0005, stable at 120BT (228,882 iters), decay at 6BT, 12BT, 20BT, 30BT, 50BT, 80BT, 120BT
- Group 2: GBSZ 512, LR 0.002, stable at 300BT (143,052 iters), decay at 200BT and 300BT


#### Validation loss evaluation script
`eval_qwen3_all`