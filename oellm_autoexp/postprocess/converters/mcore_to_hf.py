# Copyright (c) 2023 Alibaba PAI, Nvidia Megatron-LM Team and Taishi Nakamura.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import re
import sys

import torch
from huggingface_hub import save_torch_state_dict
from transformers import AutoConfig, AutoTokenizer

from oellm_autoexp.postprocess.resources import prepare_resources

megatron_to_transformers = {"self_attention.linear_proj": "self_attn.o_proj"}

tensor_parallel_params_mg = [
    "self_attention.linear_proj.weight",
    "self_attention.linear_qkv.weight",
    "self_attention.linear_proj.bias",
    "self_attention.linear_qkv.bias",
]

column_split_tensor_parallel_params_mg = ["self_attention.linear_proj"]


def get_checkpoint_sub_dir_name(tp_rank, pp_rank, pp_size):
    sub_dir_name = f"mp_rank_{tp_rank:02d}"
    if pp_size > 1:
        sub_dir_name = f"{sub_dir_name}_{pp_rank:03d}"
    return sub_dir_name


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """Load sharded checkpoints across TP ranks for a given pipeline stage."""
    tp_state_dicts = [{"model": {}} for _ in range(tp_size)]
    for tp_index in range(tp_size):
        sub_dir_name = get_checkpoint_sub_dir_name(tp_index, pp_rank, pp_size)
        logging.info("Loading %s...", sub_dir_name)
        checkpoint_path = os.path.join(
            args.load_path, sub_dir_name, "model_optim_rng.pt"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Could not find model_optim_rng.pt in {os.path.join(args.load_path, sub_dir_name)}. "
                f"Available files: {os.listdir(os.path.join(args.load_path, sub_dir_name))}"
            )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tp_state_dicts[tp_index]["model"].update(state_dict["model"])
    return tp_state_dicts


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """Permute QKV layout for compatibility with Megatron-LM checkpoint versions.

    See https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    """
    input_shape = param.size()
    if checkpoint_version == 1.0:
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def get_element_from_dict_by_path(d, path):
    if path not in d:
        d[path] = {}
    return d[path]



def _find_rank0_checkpoint(load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint load_path does not exist: {load_path}")
    possible_sub_dirs = [
        "mp_rank_00",
        "mp_rank_00_000",
        "mp_rank_00_dp_000",
        "mp_rank_00_000_dp_000",
    ]
    state_dirs = os.listdir(load_path)
    for sub_dir in possible_sub_dirs:
        if sub_dir in state_dirs:
            return os.path.join(load_path, sub_dir, "model_optim_rng.pt")
    raise FileNotFoundError(
        f"Could not find any of {possible_sub_dirs} in {load_path}. "
        f"Available: {state_dirs}"
    )


def _load_megatron_args(rank0_checkpoint_path):
    logging.info("Loading Megatron-LM checkpoint arguments from: %s", rank0_checkpoint_path)
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu", weights_only=False)
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint instead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )
    return megatron_args


def _resolve_dtype(dtype_str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype_str, torch.float32)


def _build_hf_config(args, megatron_args, tokenizer):
    logging.info("Loading HF config from: %s", args.source_model)
    config = AutoConfig.from_pretrained(args.source_model, trust_remote_code=True)
    config.architectures = [args.architecture]
    config.attention_bias = megatron_args.add_qkv_bias
    config.attention_dropout = megatron_args.attention_dropout
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.hidden_act = "silu"  # swiglu
    config.hidden_size = megatron_args.hidden_size
    config.initializer_range = megatron_args.init_method_std
    config.intermediate_size = megatron_args.ffn_hidden_size
    config.layer_norm_eps = megatron_args.norm_epsilon
    config.max_position_embeddings = megatron_args.seq_length
    config.mlp_bias = megatron_args.add_bias_linear
    config.model_type = args.model_type
    config.num_attention_heads = megatron_args.num_attention_heads
    config.num_hidden_layers = megatron_args.num_layers
    config.num_key_value_heads = (
        args.num_key_value_heads
        if args.num_key_value_heads is not None
        else megatron_args.num_query_groups
    )
    config.qk_layernorm = megatron_args.qk_layernorm
    config.rms_norm_eps = megatron_args.norm_epsilon
    config.rope_scaling = (
        None
        if megatron_args.use_rope_scaling is False
        else megatron_args.use_rope_scaling
    )
    config.rope_theta = megatron_args.rotary_base
    config.tie_word_embeddings = not megatron_args.untie_embeddings_and_output_weights
    return config


def _convert_embeddings(tp_state_dicts, tp_size, dtype):
    logging.info("Converting embeddings")
    word_embeddings = [
        tp_state_dicts[tp_rank]["model"]["embedding.word_embeddings.weight"]
        for tp_rank in range(tp_size)
    ]
    return torch.cat(word_embeddings, dim=0).to(dtype)


def _convert_mlp_params(key, val, tp_state_dicts, tp_size, path, dtype, layer_id, output_state_dict):
    if "weight" in key:
        dim = 1 if "linear_fc2" in key else 0
        params = torch.cat(
            [val] + [
                get_element_from_dict_by_path(tp_state_dicts[tp_rank], path)[key]
                for tp_rank in range(1, tp_size)
            ],
            dim=dim,
        ).to(dtype)

        if "linear_fc2" in key:
            output_state_dict[f"model.layers.{layer_id}.mlp.down_proj.weight"] = params
        else:
            params_split = [torch.chunk(i, 2, 0) for i in torch.chunk(params, tp_size, 0)]
            output_state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"] = torch.cat([i[0] for i in params_split])
            output_state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"] = torch.cat([i[1] for i in params_split])

    elif "bias" in key:
        params = torch.cat(
            [val] + [
                get_element_from_dict_by_path(tp_state_dicts[tp_rank], path)[key]
                for tp_rank in range(1, tp_size)
            ],
            dim=0,
        ).to(dtype)

        if "linear_fc2" in key:
            output_state_dict[f"model.layers.{layer_id}.mlp.down_proj.bias"] = params
        else:
            params_split = [torch.chunk(i, 2, 0) for i in torch.chunk(params, tp_size, 0)]
            output_state_dict[f"model.layers.{layer_id}.mlp.gate_proj.bias"] = torch.cat([i[0] for i in params_split])
            output_state_dict[f"model.layers.{layer_id}.mlp.up_proj.bias"] = torch.cat([i[1] for i in params_split])


def _convert_attention_params(
    op_name, weight_or_bias, params, layer_name,
    heads, num_groups, hidden_size, hidden_size_per_head, tp_size,
    output_state_dict,
):
    if "q_layernorm" in op_name:
        output_state_dict[f"{layer_name}.self_attn.q_layernorm.{weight_or_bias}"] = params.clone()
    elif "k_layernorm" in op_name:
        output_state_dict[f"{layer_name}.self_attn.k_layernorm.{weight_or_bias}"] = params.clone()
    elif op_name.endswith("layer_norm_weight") or op_name.endswith("layernorm"):
        if "qkv" in op_name:
            output_state_dict[f"{layer_name}.input_layernorm.{weight_or_bias}"] = params.clone()
        elif "mlp.linear_fc1" in op_name:
            output_state_dict[f"{layer_name}.post_attention_layernorm.{weight_or_bias}"] = params.clone()
    elif op_name in ("attention.linear_qkv", "self_attention.linear_qkv") and weight_or_bias == "weight":
        _convert_qkv_weight(params, layer_name, heads, num_groups, hidden_size, hidden_size_per_head, tp_size, output_state_dict)
    elif op_name in ("attention.linear_qkv", "self_attention.linear_qkv") and weight_or_bias == "bias":
        _convert_qkv_bias(params, layer_name, heads, num_groups, hidden_size_per_head, tp_size, output_state_dict)
    elif weight_or_bias == "weight":
        out_name = megatron_to_transformers[op_name]
        output_state_dict[f"{layer_name}.{out_name}.weight"] = params.clone()
    elif weight_or_bias == "bias":
        out_name = megatron_to_transformers[op_name]
        output_state_dict[f"{layer_name}.{out_name}.bias"] = params.clone()


def _convert_qkv_weight(params, layer_name, heads, num_groups, hidden_size, hidden_size_per_head, tp_size, output_state_dict):
    all_qkvs = [
        i.reshape(
            num_groups // tp_size,
            heads // num_groups * hidden_size_per_head + 2 * hidden_size_per_head,
            hidden_size,
        )
        for i in torch.chunk(params, tp_size, 0)
    ]
    split_size = heads // num_groups * hidden_size_per_head
    all_qs = torch.cat([i[:, :split_size, :].reshape(-1, hidden_size) for i in all_qkvs])
    all_kvs = torch.cat([i[:, split_size:, :].reshape(-1, hidden_size) for i in all_qkvs])

    checkpoint_version = 3.0
    out_q = megatron_to_transformers_fix_query_key_value_ordering(
        all_qs, checkpoint_version, 1, heads, hidden_size_per_head
    )
    out_kv = megatron_to_transformers_fix_query_key_value_ordering(
        all_kvs, checkpoint_version, 2, num_groups, hidden_size_per_head
    )
    out_kv = torch.chunk(out_kv, 2)

    output_state_dict[f"{layer_name}.self_attn.q_proj.weight"] = out_q.clone()
    output_state_dict[f"{layer_name}.self_attn.k_proj.weight"] = out_kv[0].clone()
    output_state_dict[f"{layer_name}.self_attn.v_proj.weight"] = out_kv[1].clone()


def _convert_qkv_bias(params, layer_name, heads, num_groups, hidden_size_per_head, tp_size, output_state_dict):
    all_qkv_biases = [
        i.reshape(
            num_groups // tp_size,
            heads // num_groups * hidden_size_per_head + 2 * hidden_size_per_head,
        )
        for i in torch.chunk(params, tp_size, 0)
    ]
    split_size = heads // num_groups * hidden_size_per_head
    all_q_biases = torch.cat([i[:, :split_size].reshape(-1) for i in all_qkv_biases])
    all_kv_biases = torch.cat([i[:, split_size:].reshape(-1) for i in all_qkv_biases])

    checkpoint_version = 3.0
    out_q_bias = megatron_to_transformers_fix_query_key_value_ordering(
        all_q_biases.unsqueeze(-1), checkpoint_version, 1, heads, hidden_size_per_head,
    ).squeeze(-1)
    out_kv_bias = megatron_to_transformers_fix_query_key_value_ordering(
        all_kv_biases.unsqueeze(-1), checkpoint_version, 2, num_groups, hidden_size_per_head,
    ).squeeze(-1)
    out_kv_bias = torch.chunk(out_kv_bias, 2)

    output_state_dict[f"{layer_name}.self_attn.q_proj.bias"] = out_q_bias.clone()
    output_state_dict[f"{layer_name}.self_attn.k_proj.bias"] = out_kv_bias[0].clone()
    output_state_dict[f"{layer_name}.self_attn.v_proj.bias"] = out_kv_bias[1].clone()


def _convert_final_layernorm(tp_state_dicts, path, dtype):
    logging.info("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    try:
        return params["decoder.final_layernorm.weight"].to(dtype).clone()
    except KeyError:
        return params["decoder.final_norm.weight"].to(dtype).clone()


def _convert_lm_head(tp_state_dicts, tp_size, dtype):
    logging.info("Converting LM head")
    params = torch.cat([
        get_element_from_dict_by_path(tp_state_dicts[i]["model"], "output_layer.weight")
        for i in range(tp_size)
    ])
    return params.to(dtype).clone()


def convert_checkpoint_from_megatron_to_transformers(args):
    """Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint."""
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    rank0_checkpoint_path = _find_rank0_checkpoint(args.load_path)
    megatron_args = _load_megatron_args(rank0_checkpoint_path)
    dtype = _resolve_dtype(args.target_params_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    config = _build_hf_config(args, megatron_args, tokenizer)

    tp_size = args.target_tensor_model_parallel_size
    pp_size = args.target_pipeline_model_parallel_size
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    output_state_dict = {}

    # Embeddings
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)
    word_embeddings = _convert_embeddings(tp_state_dicts, tp_size, dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings.clone()
    config.vocab_size = word_embeddings.shape[0]

    # Transformer layers
    logging.info("Converting transformer layers")
    heads = config.num_attention_heads
    hidden_size_per_head = config.hidden_size // heads
    num_layers = config.num_hidden_layers // pp_size
    hidden_size = config.hidden_size
    num_groups = config.num_key_value_heads
    path = "model"

    for pp_rank in range(pp_size):
        if pp_size > 0:
            logging.info("Converting pipeline parallel rank %d", pp_rank)
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            if key.endswith("_extra_state"):
                continue

            # MLP params (linear_fc) handled separately
            if "layer_norm_weight" not in key and "linear_fc" in key:
                layer_id = int(key.split(".")[2]) + pp_rank * num_layers
                _convert_mlp_params(key, val, tp_state_dicts, tp_size, path, dtype, layer_id, output_state_dict)
                continue

            new_key = key.replace("decoder.", "")
            if "layer_norm_weight" in new_key:
                new_key += ".weight"
            m = layer_re.match(new_key)
            if m is None:
                continue

            layer_idx = int(m.group(1)) + pp_rank * num_layers
            op_name = m.group(2)
            weight_or_bias = m.group(3)
            layer_name = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params_mg:
                params = val.to(dtype)
            else:
                dim = (1 if op_name in column_split_tensor_parallel_params_mg else 0) if weight_or_bias == "weight" else 0
                params = torch.cat(
                    [val] + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], path)[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            _convert_attention_params(
                op_name, weight_or_bias, params, layer_name,
                heads, num_groups, hidden_size, hidden_size_per_head, tp_size,
                output_state_dict,
            )

    if config.num_hidden_layers != (layer_idx + 1):  # noqa: F821
        raise ValueError(
            f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}"  # noqa: F821
        )

    output_state_dict["model.norm.weight"] = _convert_final_layernorm(tp_state_dicts, path, dtype)

    if not config.tie_word_embeddings:
        output_state_dict["lm_head.weight"] = _convert_lm_head(tp_state_dicts, tp_size, dtype)

    logging.info("Conversion done, saving to %s", args.save_path)
    config.save_pretrained(args.save_path)
    save_torch_state_dict(
        state_dict=output_state_dict,
        save_directory=args.save_path,
        safe_serialization=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-path", type=str, default=None)
    parser.add_argument("--convert_checkpoint_from_megatron_to_transformers", action="store_true")
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--target_tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--target_pipeline_model_parallel_size", type=int, default=1)
    parser.add_argument("--source_model", type=str, default=None)
    parser.add_argument("--target_params_dtype", type=str, default="fp32")
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--architecture", type=str, default="OpenSciForCausalLM")
    parser.add_argument("--model_type", type=str, default="opensci")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--prepare_resources", type=str, default=None,
                        help="Copy HF template files for this architecture to --save_path")
    args = parser.parse_args()

    if args.prepare_resources and args.save_path:
        prepare_resources(args.prepare_resources, args.save_path)

    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        raise NotImplementedError("Only megatron-to-transformers conversion is supported.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
