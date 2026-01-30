"""Megatron CLI metadata (auto-generated)."""

from collections.abc import Mapping

from oellm_autoexp.backends.megatron_args import MegatronArgMetadata, MegatronActionSpec

MEGATRON_ARG_METADATA: Mapping[str, MegatronArgMetadata] = {
    "account_for_embedding_in_pipeline_split": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, *input* embedding layer will be treated as a standard transformerlayer in the"
            " context of partition and placement for pipeline parallelism."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "account_for_loss_in_pipeline_split": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, loss layer will be treated as a standard transformerlayer in the context of"
            " partition and placement for pipeline parallelism."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "accumulate_allreduce_grads_in_fp32": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Gradient accumulation and all-reduce in fp32.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "adam_beta1": MegatronArgMetadata(
        arg_type=float,
        default=0.9,
        help="First coefficient for computing running averages of gradient and its square",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "adam_beta2": MegatronArgMetadata(
        arg_type=float,
        default=0.999,
        help="Second coefficient for computing running averages of gradient and its square",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "adam_eps": MegatronArgMetadata(
        arg_type=float,
        default=1e-08,
        help="Term added to the denominator to improvenumerical stability",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "add_bias_linear": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable bias in the linear layers",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "add_position_embedding": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable position embedding. Deprecated: use --position-embedding-type",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "add_qkv_bias": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable bias only in the QKV linear layers",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "adlr_autoresume": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable autoresume on adlr cluster.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "adlr_autoresume_interval": MegatronArgMetadata(
        arg_type=int,
        default=1000,
        help="Intervals over which check for autoresumetermination signal",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "align_grad_reduce": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "If not set, all PP stages will launch gradient reduces simultaneously. Otherwise, each"
            " PP stage will independently launch as needed."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "align_param_gather": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "If not set, all PP stages will launch param all-gathers simultaneously. Otherwise,"
            " each PP stage will independently launch as needed."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "app_tag_run_name": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Jobs belonging to same training run, suppose to have the same name. It will be used to"
            " track progress of a training done over multiple different jobs"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "app_tag_run_version": MegatronArgMetadata(
        arg_type=str,
        default="0.0.0",
        help=(
            "The version of the training of which current job is part of. It will be used to track"
            " the changes in the application side which might change the performance baseline"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "apply_layernorm_1p": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Adjust LayerNorm weights such that they are centered around zero. This improves"
            " numerical stability."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "apply_query_key_layer_scaling": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Scale Q * K^T by 1 / layer-number. Useful for fp16 training. Also sets"
            " `attention_softmax_in_fp32` to True."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "apply_residual_connection_post_layernorm": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, use original BERT residula connection ordering.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "apply_rope_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable rope fusion, the fusion is available only when using megatron-core.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "async_save": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help=(
            "Apply async checkpointing save. Currently works only with`torch_dist` distributed"
            " checkpoint format."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "async_tensor_model_parallel_allreduce": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="DEPRECATED. This flag is ignored.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "attention_backend": MegatronArgMetadata(
        arg_type=None,
        default="auto",
        help="Attention backend to use (flash,fused,unfused,local,auto). Defaults to auto",
        choices=(1, 2, 3, 4, 5),
        nargs=None,
        element_type=None,
    ),
    "attention_dropout": MegatronArgMetadata(
        arg_type=float,
        default=0.1,
        help="Post attention dropout probability.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "attention_softmax_in_fp32": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Run attention masking and softmax in fp32.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "auto_detect_ckpt_format": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Determine if the checkpoint format is in legacy or distributed format. If False,"
            ' expects distributed checkpoint iff args.ckpt_format != "torch". Might slow down'
            " loading a bit (double rank0 ckpt load)."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "barrier_with_L1_time": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "If not set, use barrier with level 1 time measurements. Note that this is up to the"
            " user to make sure calling barrier with their timers will not result in hangs. This"
            " can happen if for example the user adds a level 1 timer that is not called by all"
            " ranks."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "batch_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Old batch size parameter, do not use. Use --micro-batch-size instead",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "bert_binary_head": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable BERT binary head.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "bert_embedder_type": MegatronArgMetadata(
        arg_type=None,
        default="megatron",
        help="Select either Megatron or Huggingface as the Bert embedder.",
        choices=("megatron", "huggingface"),
        nargs=None,
        element_type=None,
    ),
    "bert_load": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Directory containing an BertModel checkpoint (needed to start ICT and REALM)",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "bf16": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Run model in bfloat16 mode.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "bias_dropout_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable bias and dropout fusion.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "bias_gelu_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable bias and gelu fusion.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "bias_swiglu_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "Disable bias and swiglu fusion, the fusion is available only when using megatron-core."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "biencoder_projection_dim": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help="Size of projection head used in biencoder (paper default: 128)",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "biencoder_shared_query_context_model": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Whether to share the parameters of the query and context models or not",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "block_data_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Where to save/load BlockData to/from",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "cache_mla_latents": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set caches the mla down projected latents with mla flash decode.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "calc_ft_timeouts": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, FT package will try to automatically compute the timeouts. Note: This feature"
            " is for Nvidia internal use only."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "calculate_per_token_loss": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Scale cross entropy loss by the number of non-padded tokens in the global batch,"
            " versus the default behavior of assuming all tokens are non-padded."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "check_for_large_grads": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Check for unexpectedly large grads",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "check_for_nan_in_loss_and_grad": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Check for NaNs in loss and grad",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "check_for_spiky_loss": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Check for spiky loss",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "check_weight_hash_across_dp_replicas_interval": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Interval to check weight hashes are same across DP replicas. If not specified, weight"
            " hashes not checked."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "checkpoint_activations": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Checkpoint activation to allow for training with larger models, sequences, and batch"
            " sizes."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "ckpt_assume_constant_structure": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If the model and optimizer state dict structure isconstant throughout a *single"
            " training job*, it allows fordifferent checkpointing performance optimizations."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "ckpt_convert_format": MegatronArgMetadata(
        arg_type=None,
        default=None,
        help="Checkpoint format for conversion.",
        choices=("torch", "torch_dist", "zarr"),
        nargs=None,
        element_type=None,
    ),
    "ckpt_convert_save": MegatronArgMetadata(
        arg_type=None,
        default=None,
        help="Save directory for converted checkpoint.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "ckpt_convert_update_legacy_dist_opt_format": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "When loading a checkpoint, update the legacy format for the distributed optimizer,"
            " which previously used a merged param/grad buffer and a different bucket mapping. The"
            " legacy format was deprecated on Feb 13, 2024."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "ckpt_format": MegatronArgMetadata(
        arg_type=None,
        default="torch_dist",
        help=(
            "Checkpoint format to use. torch is the format used by torch.save/load. torch_dist is a"
            " megatron built-in distributed checkpointing format. torch_dcp is the"
            " torch.distributed.checkpoint format. fsdp_dtensor is a torch DCP native, Megatron"
            " FSDP training-specific checkpoint format."
        ),
        choices=("torch", "torch_dist", "zarr", "torch_dcp", "fsdp_dtensor"),
        nargs=None,
        element_type=None,
    ),
    "ckpt_fully_parallel_load": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Apply full load parallelization across DP for distributed checkpoints.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "ckpt_fully_parallel_save": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "Disable applying full save parallelization across DP for distributed checkpoints."
            " Depending on ckpt format might decrease the number of files in the checkpoint. Makes"
            " DistributedOptimizer checkpoint non-reshardable."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "ckpt_fully_parallel_save_deprecated": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Deprecated: see --no-ckpt-fully-parallel-save.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "ckpt_step": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Checkpoint step to load model from.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "classes_fraction": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="training with fraction of classes.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "clip_grad": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Gradient clipping based on global L2 norm.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "clone_scatter_output_in_embedding": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "If not set, clone the output of the scatter in embedding layer to GC original tensor."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "config_logger_dir": MegatronArgMetadata(
        arg_type=str,
        default="",
        help="If set, will dump all configs to --config-logger-dir",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "context_parallel_size": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Degree of context parallelism.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "cp_comm_type": MegatronArgMetadata(
        arg_type=list,
        default=["p2p"],
        help=(
            "Inter-gpu communication type for context parallelism: p2p, a2a, allgather or a2a+p2p."
            " If a single string is provided, all layers will share the same communication type."
            " Users can also specify separated types for each layer like --cp-comm-type p2p p2p a2a"
            " a2a a2a+p2p a2a+p2p"
        ),
        choices=None,
        nargs="+",
        element_type=str,
    ),
    "create_attention_mask_in_dataloader": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="If set, do not create attention_masks in dataloader.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "cross_entropy_fusion_impl": MegatronArgMetadata(
        arg_type=str,
        default="native",
        help="Implementation of cross entropy loss calculation.",
        choices=("native", "te"),
        nargs=None,
        element_type=None,
    ),
    "cross_entropy_loss_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enabled fusion of cross entropy loss calculation.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "cuda_graph_scope": MegatronArgMetadata(
        arg_type=str,
        default="full",
        help=(
            'Determines the CUDA graphs capturing scope. Valid values are "full", "attn" and'
            ' "full_iteration". "Full" scope captures a whole Transformer layer. "Attn" scope only'
            ' captures operations in TransformerLayer._forward_attention(). "ful_iteration" scope'
            " captures a whole iteration."
        ),
        choices=("full", "attn", "full_iteration"),
        nargs=None,
        element_type=None,
    ),
    "cuda_graph_warmup_steps": MegatronArgMetadata(
        arg_type=int,
        default=3,
        help="Number of CUDA graph warmup steps",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "data_args_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Path to data-args. Instead of feeding `--data-path` with weighted dataset, we pass in"
            " a file path from which we read that argument. This is useful when the list of data is"
            " too big."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "data_cache_path": MegatronArgMetadata(
        arg_type=None,
        default=None,
        help="Path to a directory to hold cached index files.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "data_parallel_random_init": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable random initialization of params across data parallel ranks",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "data_parallel_sharding_strategy": MegatronArgMetadata(
        arg_type=str,
        default="no_shard",
        help="Sharding strategy of data parallelism.",
        choices=("no_shard", "optim", "optim_grads", "optim_grads_params"),
        nargs=None,
        element_type=None,
    ),
    "data_path": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help=(
            "The weight and prefix list for a set of train, validation, and testdatasets which"
            " split according to --split. The accepted formats are: (1) a single prefix, (2) a list"
            " of weight prefix pairs e.g. weight1 prefix1 weight2 prefix2, (3) a list of prefixes"
            " e.g. prefix1 prefix2. For (3), weights are inferred from the lengths of the"
            " contributing datasets. This argument is exclusive to the other independent"
            " --*-data-path arguments."
        ),
        choices=None,
        nargs="*",
        element_type=str,
    ),
    "data_per_class_fraction": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="training with fraction of data per class.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "data_sharding": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable data sharding.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "dataloader_type": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Single pass vs multiple pass data loader",
        choices=("single", "cyclic", "external"),
        nargs=None,
        element_type=None,
    ),
    "ddp_average_in_collective": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, average directly in data-parallel communication collective.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "ddp_bucket_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Bucket size for data-parallel communication",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "ddp_num_buckets": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of buckets for data-parallel communication",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "ddp_pad_buckets_for_high_nccl_busbw": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, make sure the bucket size is divisible by a large power of 2 (2^16) to ensure"
            " NCCL collectives have high bus bandwidth at large DP counts, since NCCL message size"
            " (which for ring algorithms is bucket_size / dp_size) apparently needs to be divisible"
            " by a power of 2 for high busbw."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "decoder_first_pipeline_num_layers": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "The number of transformer layers on the first pipeline stage of the decoder. Default"
            " None is even split of transformer layers across all pipeline stages"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "decoder_last_pipeline_num_layers": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "The number of transformer layers on the last pipeline stage of the decoder. Default"
            " None is even split of transformer layers across all pipeline stages"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "decoder_num_layers": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of decoder transformer layers.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "decoder_seq_length": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Maximum decoder sequence length to process.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "decoupled_lr": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help="Separate learning rate for the input and output layer",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "decoupled_min_lr": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help=(
            "Minimum value for learning rate for the input and output layer. The schedulerclip"
            " values below this threshold"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "decrease_batch_size_if_needed": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, decrease batch size if microbatch_size * dp_sizedoes not divide batch_size."
            " Useful for KSO (Keep Soldiering On)to continue making progress if number of healthy"
            " GPUs (andcorresponding dp_size) does not support current batch_size.Old batch_size"
            " will be restored if training is re-started withdp_size that divides batch_size //"
            " microbatch_size."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "defer_embedding_wgrad_compute": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, defers the vocabulary projection linear layer weightgradient compute to"
            " pipeline flush."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "delay_wgrad_compute": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Delay the wgrad compute for batch-level overlapping",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "deprecated_use_mcore_models": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "DEPRECATED. Use the implementation from megatron core.Now ignored and mcore models are"
            " the default, use --use-legacy-models to not use core models."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "deterministic_mode": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Choose code that has deterministic execution. This usually means slower execution, but"
            " is good for debugging and testing."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "dino_bottleneck_size": MegatronArgMetadata(
        arg_type=int,
        default=256,
        help="Bottle neck dimension in dino head ",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "dino_freeze_last_layer": MegatronArgMetadata(
        arg_type=float,
        default=1,
        help="Freezing last layer weights",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "dino_head_hidden_size": MegatronArgMetadata(
        arg_type=int,
        default=2048,
        help="Hidden dimension size in dino head",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "dino_local_crops_number": MegatronArgMetadata(
        arg_type=int,
        default=10,
        help="Number of local crops",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "dino_local_img_size": MegatronArgMetadata(
        arg_type=int,
        default=96,
        help="Image size for vision classification task",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "dino_norm_last_layer": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Disable Norm in last layer.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "dino_teacher_temp": MegatronArgMetadata(
        arg_type=float,
        default=0.07,
        help="teacher temperature",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "dino_warmup_teacher_temp": MegatronArgMetadata(
        arg_type=float,
        default=0.04,
        help="warump teacher temperature",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "dino_warmup_teacher_temp_epochs": MegatronArgMetadata(
        arg_type=int,
        default=30,
        help="warmup teacher temperaure epochs",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "disable_bf16_reduced_precision_matmul": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If True, sets torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False"
            " to prevent matmul from using reduced precision accumulation when using BF16."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "disable_mamba_mem_eff_path": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Disable Mamba efficient path.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "disable_straggler_on_startup": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, StragglerDetector is disabled on startup.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "dist_ckpt_format_deprecated": MegatronArgMetadata(
        arg_type=None,
        default=None,
        help="Deprecated: see --ckpt-format.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "dist_ckpt_strictness": MegatronArgMetadata(
        arg_type=str,
        default="assume_ok_unexpected",
        help=(
            "Determine handling of key mismatch during checkpoint load. Check StrictHandling docs"
            " for flags meaning. NOTE: This flag controls only distributed checkpoint load from"
            " storage, not loading state dict into the model."
        ),
        choices=(
            "assume_ok_unexpected",
            "log_unexpected",
            "log_all",
            "raise_unexpected",
            "raise_all",
            "return_unexpected",
            "return_all",
            "ignore_all",
        ),
        nargs=None,
        element_type=None,
    ),
    "distribute_saved_activations": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, distribute recomputed activations across model parallel group.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "distributed_backend": MegatronArgMetadata(
        arg_type=None,
        default="nccl",
        help="Which backend to use for distributed training.",
        choices=("nccl", "gloo"),
        nargs=None,
        element_type=None,
    ),
    "distributed_timeout_minutes": MegatronArgMetadata(
        arg_type=int,
        default=10,
        help="Timeout minutes for torch.distributed.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "embedding_init_method_std": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help=(
            "Standard deviation of the zero mean normal distribution used for embedding weight"
            " initialization. If unset, embeddings will be initialized the same way as other"
            " weights. Setting this to a value around 1.0 may avoid loss spikes in training."
            " Setting this to any value will also skip applying weight decay on embedding weights"
            " to avoid shrinkage towards zero. See https://arxiv.org/abs/2312.16903 for more"
            " details."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "embedding_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Where to save/load Open-Retrieval Embedding data to/from",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "empty_unused_memory_level": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help=(
            "Call torch.cuda.empty_cache() each iteration (training and eval), to reduce"
            " fragmentation.0=off, 1=moderate, 2=aggressive."
        ),
        choices=(0, 1, 2),
        nargs=None,
        element_type=None,
    ),
    "enable_cuda_graph": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            'Use CUDA graph capture and replay. --cuda-graph-scope="full_iteration" enables whole'
            " iteration CUDA graph. "
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "enable_experimental": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable experimental features.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "enable_ft_package": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, Fault Tolerance package is enabled. Note: This feature is for Nvidia internal"
            " use only."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "enable_full_sharding_in_hsdp": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, enable full sharding in megatron-fsdp Hybrid Sharded Data Parallel (HSDP)"
            " mode."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "enable_gloo_process_groups": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disables creation and usage of Gloo process groups.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "enable_msc": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable the usage of Multi-Storage Client (MSC) in Megatron Core.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "enable_one_logger": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "If set, disable using one_logger to track E2E metricsNote that one_logger is an"
            " internal tool and not available externally. For installation, please go to"
            " https://confluence.nvidia.com/display/MLWFO/Package+Repositoriesfor more details"
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "encoder_num_layers": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of encoder transformer layers.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "encoder_seq_length": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Maximum encoder sequence length to process.This should be exclusive of --seq-length",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "end_weight_decay": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help="End of run weight decay coefficient for L2 regularization.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "eod_mask_loss": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Mask loss for the end of document tokens.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "error_injection_rate": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help=(
            "Rate at which to inject unexpected results, e.g. 1000 means once every 1000 result"
            " validations"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "error_injection_type": MegatronArgMetadata(
        arg_type=str,
        default="transient_error",
        help="Type of error to inject. ",
        choices=("correct_result", "transient_error", "persistent_error"),
        nargs=None,
        element_type=None,
    ),
    "eval_interval": MegatronArgMetadata(
        arg_type=int,
        default=1000,
        help="Interval between running evaluation on validation set.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "eval_iters": MegatronArgMetadata(
        arg_type=int,
        default=100,
        help="Number of iterations to run for evaluationvalidation/test for.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "evidence_data_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Path to Wikipedia Evidence frm DPR paper",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "exit_duration_in_mins": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Exit the program after this many minutes.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "exit_interval": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Exit the program after the iteration is divisible by this value.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "exit_on_missing_checkpoint": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If '--load' is set, but checkpoint is not found (e.g., path typo), then exit instead"
            " of random initialization."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "exit_signal_handler": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Dynamically save the checkpoint and shutdown the training if SIGTERM is received",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "exp_avg_dtype": MegatronArgMetadata(
        arg_type=None,
        default="fp32",
        help=(
            "Dtype of exp_avg (1st moment in adam optimizer) when enabling"
            " precision-aware-optimizer. This dtype is used for storing the optimizer state in"
            " memory during training but does not affect the precision in the kernel computation."
        ),
        choices=("fp32", "fp16", "bf16", "fp8"),
        nargs=None,
        element_type=None,
    ),
    "exp_avg_sq_dtype": MegatronArgMetadata(
        arg_type=None,
        default="fp32",
        help=(
            "Dtype of exp_avg_sq (2nd moment in adam optimizer) when enabling"
            " precision-aware-optimizer. This dtype is used for storing the optimizer state in"
            " memory during training but does not affect the precision in the kernel computation."
        ),
        choices=("fp32", "fp16", "bf16", "fp8"),
        nargs=None,
        element_type=None,
    ),
    "expert_model_parallel_size": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Degree of expert model parallelism.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "expert_tensor_parallel_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Degree of expert model parallelism. Default is None, which will be set to the value of"
            " --tensor-model-paralle-size."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "external_cuda_graph": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use CUDA graph capture and replay. The CUDA graphs aremanually captured in the"
            " training script."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "ffn_hidden_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Transformer Feed-Forward Network hidden size. This is set to 4*hidden-size if not"
            " provided"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "finetune": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Load model for finetuning. Do not load optimizer or rng state from checkpoint and set"
            " iteration to 0. Assumed when loading a release checkpoint."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "first_last_layers_bf16": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Construct first and last layers in bf16 when doing FP8 training.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "flash_decode": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Whether to use the flash decoding kernel.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "fp16": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Run model in fp16 mode.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "fp16_lm_cross_entropy": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Move the cross entropy unreduced loss calculationfor lm head to fp16.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "fp32_residual_connection": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Move residual connections to fp32.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "fp8": MegatronArgMetadata(
        arg_type=None,
        default=None,
        help="Which fp8 format scheme to use for FP8 tensors in the forward and backward pass",
        choices=("e4m3", "hybrid"),
        nargs=None,
        element_type=None,
    ),
    "fp8_amax_compute_algo": MegatronArgMetadata(
        arg_type=None,
        default="most_recent",
        help="Algorithm for computing amax from history",
        choices=("most_recent", "max"),
        nargs=None,
        element_type=None,
    ),
    "fp8_amax_history_len": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Number of steps for which amax history is recorded per tensor",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "fp8_interval": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="DEPRECATED. This flag is ignored. Scaling update interval for fp8",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "fp8_margin": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help="Scaling margin for fp8",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "fp8_param_gather": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Keep the compute param in fp8 (do not use any other intermediate dtype) and perform"
            " the param all-gather in fp8."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "fp8_recipe": MegatronArgMetadata(
        arg_type=None,
        default="delayed",
        help="Which fp8 recipe to use for FP8 tensors in the forward and backward pass",
        choices=("tensorwise", "delayed", "mxfp8", "blockwise"),
        nargs=None,
        element_type=None,
    ),
    "fp8_wgrad": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Execute wgrad in higher precision even for FP8 runs",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "fsdp_double_buffer": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Enable double buffering for temporary memory needed for Megatron FSDP communications."
            " Double-buffering the communication memory improves memory management efficiency by"
            " reusing previously allocated buffers, rather than creating new buffers for each FSDP"
            " communication. This is required for user buffer registration and is enabled by"
            " default when using NCCL user buffers."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "full_validation": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, each time validation occurs it uses the full validation dataset(s). This"
            " currently only works for GPT datasets!"
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "global_batch_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Training batch size. If set, it should be a multiple of micro-batch-size times"
            " data-parallel-size. If this value is None, then use micro-batch-size *"
            " data-parallel-size as the global batch size. This choice will result in 1 for number"
            " of micro-batches."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "grad_reduce_in_bf16": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Reduce gradients in bfloat16.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "gradient_accumulation_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable fusing gradient accumulation to weight gradient computation of linear layers",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "gradient_reduce_div_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="If not set, fuse the division in gradient reduce.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "group_query_attention": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use group-query attention.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "head_lr_mult": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="learning rate multiplier for head during finetuning",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "heterogeneous_layers_config_encoded_json": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "This is encoded json string of the heterogeneous model configuration. Used to keep the"
            " content of the heterogeneous model specification in args when the model is loaded"
            " from a checkpoint. Use the format of the HuggingFace config files in llama nemotron"
            " models, e.g."
            " https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/resolve/main/config.json."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "heterogeneous_layers_config_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Path to json file containing heterogeneous model configuration. Use the format of the"
            " HuggingFace config files in llama nemotron models, e.g."
            " https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/resolve/main/config.json."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "hidden_dropout": MegatronArgMetadata(
        arg_type=float,
        default=0.1,
        help="Dropout probability for hidden state transformer.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "hidden_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Tansformer hidden size.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "hierarchical_context_parallel_sizes": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help=(
            "Degrees of the hierarchical context parallelism. Users should provide a list to"
            " specify the sizes for different levels. --hierarchical-context-parallel-sizes 2 4"
            " indicates every two adjacent gpus forms the first level of cp groups and the cp ranks"
            " with the same odevity forms the second level of cp groups."
        ),
        choices=None,
        nargs="+",
        element_type=int,
    ),
    "high_priority_stream_groups": MegatronArgMetadata(
        arg_type=list,
        default=[],
        help="The communicator group names to use high priority streams.",
        choices=None,
        nargs="*",
        element_type=str,
    ),
    "hybrid_attention_ratio": MegatronArgMetadata(
        arg_type=float,
        default=0.0,
        help="Ratio of attention layers to total layers, in the range [0.0, 1.0].",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "hybrid_mlp_ratio": MegatronArgMetadata(
        arg_type=float,
        default=0.0,
        help="Ratio of mlp layers to total layers, in the range [0.0, 1.0].",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "hybrid_override_pattern": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Force a specific hybrid layer pattern. The valueshould be a string of characters"
            " chosen fromcore.ssm.mamba_hybrid_layer_allocation.Symbols.If a value greater than 0.0"
            " is supplied to any of the hybrid ratio arguments, then the number of each typeof"
            " layer in the override pattern must match number inthe overidden pattern"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "hysteresis": MegatronArgMetadata(
        arg_type=int,
        default=2,
        help="hysteresis for dynamic loss scaling",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "ict_head_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Size of block embeddings to be used in ICT and REALM (paper default: 128)",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "ict_load": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Directory containing an ICTBertModel checkpoint",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "img_h": MegatronArgMetadata(
        arg_type=int,
        default=224,
        help="Image height for vision classification task",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "img_w": MegatronArgMetadata(
        arg_type=int,
        default=224,
        help="Image height for vision classification task",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "indexer_batch_size": MegatronArgMetadata(
        arg_type=int,
        default=128,
        help="How large of batches to use when doing indexing jobs",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "indexer_log_interval": MegatronArgMetadata(
        arg_type=int,
        default=1000,
        help="After how many batches should the indexer report progress",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_batch_times_seqlen_threshold": MegatronArgMetadata(
        arg_type=int,
        default=-1,
        help=(
            "If (batch-size * sequence-length) is smaller than this thresholdthen batches will not"
            " be split up for pipelining.Requires setting --pipeline-model-parallel-size >"
            " 1.Setting this to -1 indicates that batch pipelining is not used."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_dynamic_batching": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable dynamic batching mode.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "inference_dynamic_batching_buffer_guaranteed_fraction": MegatronArgMetadata(
        arg_type=float,
        default=0.2,
        help=(
            "Space is reserved within the inference context memory buffer to guarantee that a"
            " minimum number of active requests will always be able to run to completion. This is"
            " to avoid the context being blocked by paused requests."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_dynamic_batching_buffer_overflow_factor": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help=(
            "Scaling factor over the memory buffer size for auto computing `max_requests` and"
            " `max_tokens`. This scaling factor is used for fitting more requests and tokens in the"
            " memory buffer than it can safely hold, which in turn increases throughput."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_dynamic_batching_buffer_size_gb": MegatronArgMetadata(
        arg_type=float,
        default=40.0,
        help="Total buffer size (GB) allocated for the chunked KV memory.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_dynamic_batching_chunk_size": MegatronArgMetadata(
        arg_type=int,
        default=256,
        help="KV cache chunk size. It should be a multiple of 256",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_dynamic_batching_max_requests_override": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "If set, this overrides the max requests as computed from"
            " `--inference-dynamic-batching-buffer-overflow-factor`."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_dynamic_batching_max_tokens_override": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "If set, this overrides the max tokens as computed from"
            " `--inference-dynamic-batching-buffer-overflow-factor`."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_dynamic_batching_num_cuda_graphs": MegatronArgMetadata(
        arg_type=int,
        default=16,
        help=(
            "Maximum number of cuda graphs to capture, where the cuda graph batch sizes range from"
            " 1 to `max_requests`. (See `dynamic_context.py` for details on how `max_requests` is"
            " computed). Due to rounding, the actual number of cuda graphs may not equal this"
            " argument."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_max_batch_size": MegatronArgMetadata(
        arg_type=int,
        default=8,
        help="Maximum number of requests for inference.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_max_seq_length": MegatronArgMetadata(
        arg_type=int,
        default=2560,
        help="Maximum sequence length expected for inference (prefill + decode).",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inference_rng_tracker": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use a random number generator configured for inference.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "init_method_std": MegatronArgMetadata(
        arg_type=float,
        default=0.02,
        help=(
            "Standard deviation of the zero mean normal distribution used for weight"
            " initialization."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "init_method_xavier_uniform": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable Xavier uniform parameter initialization",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "init_model_with_meta_device": MegatronArgMetadata(
        arg_type=bool, default=False, help=None, choices=None, nargs=0, element_type=None
    ),
    "initial_loss_scale": MegatronArgMetadata(
        arg_type=float,
        default=4294967296,
        help="Initial loss-scale for dynamic loss scaling.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_active_world_size": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help=(
            "The number of ranks initially executing the workload. The remaining ranks from the"
            " allocation are set aside as warm reserve."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_barrier_timeout": MegatronArgMetadata(
        arg_type=float,
        default=120,
        help="Timeout (in seconds) for internal distributed barrier",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_completion_timeout": MegatronArgMetadata(
        arg_type=float,
        default=120,
        help="Timeout (in seconds) for barrier on completion on all ranks",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_empty_cuda_cache": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Release all unoccupied cached GPU memory on every in-process restart.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "inprocess_granularity": MegatronArgMetadata(
        arg_type=str,
        default="node",
        help="Granularity for in-process restart.",
        choices=("node", "rank"),
        nargs=None,
        element_type=None,
    ),
    "inprocess_hard_timeout": MegatronArgMetadata(
        arg_type=float,
        default=90,
        help="Hard progress timeout (in seconds).",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_heartbeat_interval": MegatronArgMetadata(
        arg_type=float,
        default=30,
        help="Monitoring interval (in seconds) for detecting unresponsive ranks.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_heartbeat_timeout": MegatronArgMetadata(
        arg_type=float,
        default=60,
        help="Timeout (in seconds) for a missing rank detection heartbeat.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_last_call_wait": MegatronArgMetadata(
        arg_type=float,
        default=1,
        help="Time interval (in seconds) for other ranks to report concurrent terminal failures.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_max_iterations": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Maximum number of in-process restart iterations.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_monitor_process_interval": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Monitoring interval (in seconds) for the monitoring process.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_monitor_thread_interval": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Monitoring interval (in seconds) for the monitoring thread.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_progress_watchdog_interval": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Interval (in seconds) for automatic progress watchdog timestamp updates.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_restart": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enables in-process restart.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "inprocess_soft_timeout": MegatronArgMetadata(
        arg_type=float,
        default=60,
        help="Soft progress timeout (in seconds).",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "inprocess_termination_grace_time": MegatronArgMetadata(
        arg_type=float,
        default=1,
        help="Interval (in seconds) between SIGTERM and SIGKILL issued on hard timeout",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "is_hybrid_model": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Indicates whether the model is a hybrid model.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "iter_per_epoch": MegatronArgMetadata(
        arg_type=int,
        default=1250,
        help="iterations per epoch",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "iterations_to_skip": MegatronArgMetadata(
        arg_type=list,
        default=[],
        help="List of iterations to skip, empty by default.",
        choices=None,
        nargs="+",
        element_type=int,
    ),
    "keep_fp8_transpose_cache": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, keep the fp8 transpose cache when using Megatron FSDP.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "kitchen_config_file": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Use the config .yaml file at the specified location to configure kitchen quantization."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "kitchen_recipe_number": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Use a default kitchen recipe for all layers as defined by QAT_PARAMS index",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "kv_channels": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Projection weights dimension in multi-head attention. This is set to   "
            " args.hidden_size // args.num_attention_heads if not provided."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "kv_lora_rank": MegatronArgMetadata(
        arg_type=int,
        default=32,
        help="Rank of Key and Value tensors' low rank representation.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lazy_mpu_init": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help=(
            "If set to True, initialize_megatron() skips DDP initialization and returns function to"
            " complete it instead. Also turns on --use-cpu-initialization flag. This is for"
            " external DDP manager."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "load": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Directory containing a model checkpoint.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "load_main_params_from_ckpt": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help="Load main parameters from checkpoint directly.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "load_model_opt_format": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Load a checkpoint for TensorRT model optimizer (nvidia-modelopt).This function can"
            " also be used to load NeMo .nemo sharded checkpoints."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "local_rank": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help="local rank passed from distributed launcher.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "log_energy": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, log energy consumption (in Joules)",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_interval": MegatronArgMetadata(
        arg_type=int,
        default=100,
        help="Report loss and timing interval.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "log_loss_scale_to_tensorboard": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable loss-scale logging to tensorboard.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_memory_to_tensorboard": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable memory logging to tensorboard.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_num_zeros_in_grad": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, calculate and log the number of zeros in gradient.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_params_norm": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, calculate and log parameters norm.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_progress": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, log progress (in terms of number of processed tokens and number of"
            " floating-point operations) to progress.txt file in checkpoint directory."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_straggler": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, tracks and logs straggler per GPU.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_throughput": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, calculate and log throughput per GPU.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_timers_to_tensorboard": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, write timers to tensorboard.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_validation_ppl_to_tensorboard": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, write validation perplexity to tensorboard.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "log_world_size_to_tensorboard": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable world size logging to tensorboard.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "logging_level": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Set default logging level",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "loss_scale": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help=(
            "Static loss scaling, positive power of 2 values can improve fp16 convergence. If None,"
            " dynamicloss scaling is used."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "loss_scale_window": MegatronArgMetadata(
        arg_type=float,
        default=1000,
        help="Window over which to raise/lower dynamic scale.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help=(
            "Initial learning rate. Depending on decay style and initial warmup, the learning rate"
            " at each iteration would be different."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_decay_iters": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "number of iterations to decay learning rate over, If None defaults to `--train-iters`"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_decay_samples": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="number of samples to decay learning rate over, If None defaults to `--train-samples`",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_decay_style": MegatronArgMetadata(
        arg_type=str,
        default="linear",
        help="Learning rate decay function.",
        choices=("constant", "linear", "cosine", "inverse-square-root", "WSD"),
        nargs=None,
        element_type=None,
    ),
    "lr_warmup_fraction": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help="fraction of lr-warmup-(iters/samples) to use for warmup (as a float)",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_warmup_init": MegatronArgMetadata(
        arg_type=float,
        default=0.0,
        help="Initial value for learning rate warmup. The scheduler starts warmup from this value.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_warmup_iters": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help="number of iterations to linearly warmup learning rate over.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_warmup_samples": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help="number of samples to linearly warmup learning rate over.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_wsd_decay_iters": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="number of iterations for the annealing phase in the wsd schedule",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_wsd_decay_samples": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="number of samples for the annealing phase in the wsd schedule",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "lr_wsd_decay_style": MegatronArgMetadata(
        arg_type=str,
        default="exponential",
        help="Decay style for the annealing phase of WSD",
        choices=("exponential", "linear", "cosine", "minus_sqrt"),
        nargs=None,
        element_type=None,
    ),
    "main_grads_dtype": MegatronArgMetadata(
        arg_type=None,
        default="fp32",
        help="Dtype of main grads when enabling precision-aware-optimizer",
        choices=("fp32", "bf16"),
        nargs=None,
        element_type=None,
    ),
    "main_params_dtype": MegatronArgMetadata(
        arg_type=None,
        default="fp32",
        help="Dtype of main params when enabling precision-aware-optimizer",
        choices=("fp32", "fp16"),
        nargs=None,
        element_type=None,
    ),
    "make_vocab_size_divisible_by": MegatronArgMetadata(
        arg_type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value.This is added for computational"
            " efficieny reasons."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mamba_head_dim": MegatronArgMetadata(
        arg_type=int,
        default=64,
        help="Head dimension for Mamba layers.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mamba_num_groups": MegatronArgMetadata(
        arg_type=int,
        default=8,
        help="Number of groups for Mamba layers.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mamba_num_heads": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Number of heads for Mamba layers.If not set, then the number of heads will be"
            " --hidden-size * expand // --mamba-head-dim"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mamba_state_dim": MegatronArgMetadata(
        arg_type=int,
        default=128,
        help="State dimension for Mamba layers.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "manual_gc": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Disable the threshold-based default garbage collector and trigger the garbage"
            " collection manually. Manual garbage collection helps to align the timing of the"
            " collection across ranks which mitigates the impact of CPU-associated jitters. When"
            " the manual gc is enabled, garbage collection is performed only at the start and the"
            " end of the validation routine by default."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "manual_gc_eval": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "When using manual garbage collection, disable garbage collection at the start and the"
            " end of each evaluation run."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "manual_gc_interval": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help=(
            "Training step interval to trigger manual garbage collection. When the value is set to"
            " 0, garbage collection is not triggered between training steps."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mask_factor": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="mask size scaling parameter",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mask_prob": MegatronArgMetadata(
        arg_type=float,
        default=0.15,
        help="Probability of replacing a token with mask.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mask_type": MegatronArgMetadata(
        arg_type=str,
        default="random",
        help="mask types",
        choices=("random", "row"),
        nargs=None,
        element_type=None,
    ),
    "masked_softmax_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable fusion of query_key_value scaling, masking, and softmax.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "max_position_embeddings": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Maximum number of position embeddings to use. This is the size of position embedding."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "max_tokens_to_oom": MegatronArgMetadata(
        arg_type=int,
        default=12000,
        help=(
            "Maximum number of tokens during inferencetokens here is # in prompt + # to"
            " generateAllows us to throw an error before OOM crashes server"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "memory_snapshot_path": MegatronArgMetadata(
        arg_type=str,
        default="snapshot.pickle",
        help="Specifies where to dump the memory history pickle.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "merge_file": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Path to the BPE merge file.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "micro_batch_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Batch size per model instance (local batch size). Global batch size is local batch"
            " size times data parallel size times number of micro batches."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "microbatch_group_size_per_vp_stage": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of contiguous microbatches per virtual pipeline stage",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mid_level_dataset_surplus": MegatronArgMetadata(
        arg_type=float,
        default=0.005,
        help="The sample surplus to build for the mid-level datasets(s)",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "min_loss_scale": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Minimum loss scale for dynamic loss scaling.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "min_lr": MegatronArgMetadata(
        arg_type=float,
        default=0.0,
        help="Minimum value for learning rate. The schedulerclip values below this threshold.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mlp_chunks_for_prefill": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Number of chunks along sequence dimension for MLP computation during prefill",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mmap_bin_files": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable mmap-ing of .bin files.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "mock_data": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Skip data loading and validation and opt for artificial generation of mock data when"
            " an implementation is available."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "model_parallel_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Old model parallel argument, do not use. Use --tensor-model-parallel-size instead.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_apply_probs_on_input": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Apply probs before mlp activation for moe routing.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_aux_loss_coeff": MegatronArgMetadata(
        arg_type=list,
        default=0.0,
        help="Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.",
        choices=None,
        nargs="+",
        element_type=float,
    ),
    "moe_deepep_num_sms": MegatronArgMetadata(
        arg_type=int,
        default=20,
        help="Number of SMs to use for DeepEP.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_enable_deepep": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "[Experimental] Enable DeepSeek/DeepEP for efficient token dispatching and combine in"
            " MoE models. Only works with flex token dispatcher by setting"
            " --moe-token-dispatcher-type=flex."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_expert_capacity_factor": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help="The capacity factor for each expert, None means no token will be dropped.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_extended_tp": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Deprecated. Use --expert-tensor-parallel-size instead.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_ffn_hidden_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "The hidden size of each expert's feed-forward network (ffn). If not specified,"
            " defaults to the ffn_hidden_size."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_grouped_gemm": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "When there are multiple experts per rank, launch multiple local GEMM kernels in"
            " multiple streams to improve the utilization and performance with GroupedLinear in"
            " TransformerEngine."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_input_jitter_eps": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help="Add noise to the input tensor by applying jitter with a specified epsilon value.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_layer_freq": MegatronArgMetadata(
        arg_type=None,
        default=1,
        help=(
            "Frequency between MoE layers and Dense layers. Accepts either: - An integer N:"
            " Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers - A"
            " string containing a Python list expression that defines a custom pattern, e.g.:"
            ' "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0] where 1 indicates an expert'
            ' layer and 0 indicates a dense layer. Examples: "([0]+[1]*23)": 1 dense layer followed'
            ' by 23 experts layers, "([1]*3+[0]*2)*2": Three expert layers followed by two dense'
            " layers, repeated twice."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_layer_recompute": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Enable checkpointing for moe_layer, should be used when memory is not sufficient."
            ' Deprecated. Use "--recompute-granularity selective --recompute-modules moe" instead.'
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_pad_expert_input_to_capacity": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Pads the input for each expert to match the expert capacity length, effective only"
            " after the --moe-expert-capacity-factor is set."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_per_layer_logging": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable per-layer logging for MoE, currently supports auxiliary loss and z loss.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_permute_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Fuse token rearrangement ops during token dispatching.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_router_bias_update_rate": MegatronArgMetadata(
        arg_type=float,
        default=0.001,
        help=(
            "Expert bias update rate in the aux-loss-free load balancing strategy. The expert bias"
            " is updated based on the number of assigned tokens to each expert in a global batch,"
            " where the bias is increased for the experts with less assigned tokens and decreased"
            " for the experts with more assigned tokens. The default value 1e-3 is same as that"
            " used in DeepSeekV3."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_router_dtype": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Data type for routing computation and expert output weighted averaging. Fp32/fp64"
            " enhances numerical stability, especially with numerous experts. The perf impact"
            " should be negligible when used with permute fusion. None means no changes for dtype."
        ),
        choices=("fp32", "fp64"),
        nargs=None,
        element_type=None,
    ),
    "moe_router_enable_expert_bias": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "TopK routing with dynamic expert bias in the aux-loss-free load balancing strategy."
            " The routing decision is based on the sum of the routing scores and the expert bias."
            " See https://arxiv.org/abs/2408.15664 for details."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_router_force_load_balancing": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "[Experimental] Force override routing to balance token distribution using random"
            " logits for MoE routers, supporting naive top-k and group-limited top-k. This"
            " experimental feature is for benchmarking purposes only!"
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_router_fusion": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Enable fusion for MoE TopK routing and aux-loss computation. This is only supported in"
            " TransformerEngine 2.7.0 and above."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_router_group_topk": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of selected groups for group-limited routing.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_router_load_balancing_type": MegatronArgMetadata(
        arg_type=list,
        default="aux_loss",
        help=(
            'Determines the load balancing strategy for the router. "aux_loss" corresponds to the'
            ' load balancing loss used in GShard and SwitchTransformer; "seq_aux_loss" corresponds'
            " to the load balancing loss used in DeepSeekV2, which computes the loss for each"
            ' individual sample; "sinkhorn" corresponds to the balancing algorithm used in S-BASE,'
            ' and "none" implies no load balancing. The default is "aux_loss".'
        ),
        choices=("aux_loss", "seq_aux_loss", "sinkhorn", "none"),
        nargs="+",
        element_type=str,
    ),
    "moe_router_num_groups": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Number of groups to divide experts into for group-limited routing. When using"
            " group-limited routing: 1) Experts are divided into equal-sized groups, 2) For each"
            " token, a subset of groups are selected based on routing scores (sum of top-2 expert"
            " scores within each group), 3) From these selected groups, moe_router_topk experts are"
            " chosen.Two common use cases: 1) Device-limited routing: Set equal to expert parallel"
            " size (EP) to limit each token to experts on a subset of devices (See DeepSeek-V2:"
            " https://arxiv.org/pdf/2405.04434) 2) Node-limited routing: Set equal to number of"
            " nodes in EP group to limit each token to experts on a subset of nodes (See"
            " DeepSeek-V3: https://arxiv.org/pdf/2412.19437)"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_router_padding_for_fp8": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Pad the routing_map to make sure the number of tokens each expert received is a"
            " multiple of 16/32 for FP8 precision. It is suggested to enable this for dropless"
            " training with FP8 precision when num_local_experts > 1. This is a more efficient way"
            " to pad for FP8 which eliminates the explicit padding in the GroupedMLP layer."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_router_pre_softmax": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Enable pre-softmax routing for MoE, which means softmax is before the top-k selection."
            " By default, softmax is done after top-k."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_router_score_function": MegatronArgMetadata(
        arg_type=str,
        default="softmax",
        help='Score function for MoE TopK routing. Can be "softmax" or "sigmoid".',
        choices=("softmax", "sigmoid"),
        nargs=None,
        element_type=None,
    ),
    "moe_router_topk": MegatronArgMetadata(
        arg_type=int,
        default=2,
        help="Number of experts to route to for each token. The default is 2.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_router_topk_scaling_factor": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help=(
            "Scaling factor for routing score in top-k selection, only works when"
            " --moe-router-pre-softmax enabled. Defaults to None, which means no scaling."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_shared_expert_intermediate_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            'Shared expert total ffn hidden size. It should be equal to "num_shared_experts *'
            ' ffn_size_of_each_shared_expert" if there are multiple shared experts. None means no'
            " shared expert."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_shared_expert_overlap": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Enable overlapping between shared expert computations and dispatcher communications."
            " Without this, the shared epxerts execute after the routed experts. Only effective"
            " when moe-shared-expert-intermediate-size is set."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_token_dispatcher_type": MegatronArgMetadata(
        arg_type=str,
        default="allgather",
        help=(
            "The type of token dispatcher to use. The default is 'allgather'. Options are"
            " 'allgather', 'alltoall'. We recommend using 'alltoall' when applying expert"
            " parallelism. For more information, please refer to the documentation in"
            " core/moe/README."
        ),
        choices=("allgather", "alltoall", "flex"),
        nargs=None,
        element_type=None,
    ),
    "moe_token_drop_policy": MegatronArgMetadata(
        arg_type=str,
        default="probs",
        help=(
            'The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens'
            ' with the lowest probabilities will be dropped. If "position", tokens at the end of'
            " each batch will be dropped."
        ),
        choices=("probs", "position"),
        nargs=None,
        element_type=None,
    ),
    "moe_upcycling_granularity": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help=(
            "This param sepecifics how many times smaller is the expert hidden size compared with"
            " the original dense FFN hidden size. For using granular upcycling strategy, please set"
            " this param as a positive integer. If this param is set to 1, it means using the"
            " default upcycling strategy."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "moe_use_legacy_grouped_gemm": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use legacy GroupedMLP rather than TEGroupedMLP. Note: The legacy one will be"
            " deprecated soon."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_use_upcycling": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Load a checkpoint of a dense model, convert it into an MoE model, and save the"
            " converted model to the path specified by --save. Upcycling is implemented on the top"
            " of distributed checkpointing, so it supports parallel modes different from the dense"
            " model."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "moe_z_loss_coeff": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help="Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mrope_section": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help="Multimodal rope section is for channel dimension, empty by default.",
        choices=None,
        nargs="+",
        element_type=int,
    ),
    "mscale": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Mscale for YaRN RoPE in multi-latent attention.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mscale_all_dim": MegatronArgMetadata(
        arg_type=float,
        default=0.0,
        help="Mscale all dimensions for YaRN RoPE in multi-latent attention.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mtp_loss_scaling_factor": MegatronArgMetadata(
        arg_type=float,
        default=0.1,
        help=(
            "Scaling factor of Multi-Token Prediction (MTP) loss. We compute the average of the MTP"
            " losses across all depths, and multiply it the scaling factor to obtain the overall"
            " MTP loss, which serves as an additional training objective."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "mtp_num_layers": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Number of Multi-Token Prediction (MTP) Layers.MTP extends the prediction scope to"
            " multiple future tokens at each position.This MTP implementation sequentially predict"
            " additional tokens by using D sequential modules to predict D additional tokens."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "multi_latent_attention": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use multi-latent attention for model.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "multiple_validation_sets": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, multiple datasets listed in the validation split are evaluated independently"
            " with a separate loss for each dataset in the list. This argument requires that no"
            " weights are included in the list"
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "nccl_all_reduce_for_prefill": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "When using symmeric all reduce kernels this will use regular nccl kernels for prefill."
            " This can be more effecient when prefill is large as the nccl kernels can be more"
            " bandwith optimized"
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "nccl_communicator_config_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Path to the yaml file with NCCL communicator configurations. The number of min/max"
            " thread groups and thread group cluster size of each communicator can be configured by"
            " setting `min_ctas`, `max_ctas`, and `cga_cluster_size`."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "nccl_ub": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use the userbuffer registration for DP/FSDP communication buffers.This option will"
            " reduce GPU SM usage for the DP/FSDP communication,which is improving the performance"
            " of the overlapped computation."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "no_load_optim": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help="Do not load optimizer when loading checkpoint.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "no_load_rng": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help="Do not load rng state when loading checkpoint.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "no_persist_layer_norm": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Disable using persistent fused layer norm kernel. This kernel supports only a set of"
            " hidden sizes. Please check persist_ln_hidden_sizes if your hidden size is supported."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "no_rope_freq": MegatronArgMetadata(
        arg_type=None,
        default=None,
        help=(
            "Controls which layers to skip performing Rotary Position Embedding. Accepts either: -"
            " An integer N: Represents a 1:N ratio, meaning RoPE is skipped every N-1 layers. - A"
            " string containing a Python list expression that defines a custom pattern, e.g.:"
            ' "([0]*3+[1]*1)*3" evaluates to [0,0,0,1,0,0,0,1,0,0,0,1] where 1 indicates no-rope'
            " layer. This patten is equivalent to --no-rope-freq=4.By default this is disabled and"
            " set to None, indicating RoPE will be performedon every layer."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "no_save_optim": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help="Do not save current optimizer.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "no_save_rng": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help="Do not save current rng state.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "non_persistent_ckpt_type": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            'Type of non-persistent model checkpoints. "global" - Saved as a standard checkpoint'
            ' (e.g., on Lustre) with old checkpoints being removed. "local" - Each rank saves a'
            " portion of the checkpoint locally (e.g., on SSD/ramdisk). None - No non-persistent"
            " checkpointing (default option)."
        ),
        choices=("global", "local", "in_memory", None),
        nargs=None,
        element_type=None,
    ),
    "non_persistent_global_ckpt_dir": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Directory containing global non-persistent model checkpoints.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "non_persistent_local_ckpt_algo": MegatronArgMetadata(
        arg_type=str,
        default="fully_parallel",
        help="Algorithm for local non-persistent checkpointing.",
        choices=("fully_parallel", "atomic"),
        nargs=None,
        element_type=None,
    ),
    "non_persistent_local_ckpt_dir": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Directory containing local non-persistent model checkpoints.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "non_persistent_save_interval": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of iterations between non-persistent saves.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "norm_epsilon": MegatronArgMetadata(
        arg_type=float,
        default=1e-05,
        help="Epsilon for layer norm and RMS norm.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "normalization": MegatronArgMetadata(
        arg_type=None,
        default="LayerNorm",
        help="Which normalization technique to use.",
        choices=("LayerNorm", "RMSNorm"),
        nargs=None,
        element_type=None,
    ),
    "num_attention_heads": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of transformer attention heads.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_channels": MegatronArgMetadata(
        arg_type=int,
        default=3,
        help="Number of channels in input image data",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_classes": MegatronArgMetadata(
        arg_type=int,
        default=1000,
        help="num of classes in vision classificaiton task",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_dataset_builder_threads": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Number of parallel threads per rank for dataset builder",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_distributed_optimizer_instances": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Number of Distributed Optimizer copies across Data Parallel domain.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_experts": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of Experts in MoE (None means no MoE)",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_layers": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of transformer layers.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_layers_at_end_in_bf16": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help=(
            "Number of layers at end to construct in bf16 when --first-last-layers-bf16 is enabled."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_layers_at_start_in_bf16": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help=(
            "Number of layers at start to construct in bf16 when --first-last-layers-bf16 is"
            " enabled."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_layers_per_virtual_pipeline_stage": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of layers per virtual pipeline stage",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_query_groups": MegatronArgMetadata(
        arg_type=int, default=1, help=None, choices=None, nargs=None, element_type=None
    ),
    "num_virtual_stages_per_pipeline_rank": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of virtual pipeline stages per pipeline parallelism rank",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "num_workers": MegatronArgMetadata(
        arg_type=int,
        default=2,
        help="Dataloader number of workers.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "object_storage_cache_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Path to cache index files when using s3 or msc dataloader",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "one_logger_async": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, forces one_logger to use async mode.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "one_logger_project": MegatronArgMetadata(
        arg_type=str,
        default="megatron-lm",
        help="The one-logger project name. Will ignore if --no-one-logger is set",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "one_logger_run_name": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="The one-logger run name displayed. Will ignore if --no-one-logger is set",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "onnx_safe": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help="Use workarounds for known problems with Torch ONNX exporter",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "openai_gelu": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use OpenAIs GeLU implementation. This optionshould not be used unless for backward"
            " compatibilityreasons."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "optimizer": MegatronArgMetadata(
        arg_type=str,
        default="adam",
        help="Optimizer function",
        choices=("adam", "sgd"),
        nargs=None,
        element_type=None,
    ),
    "optimizer_cpu_offload": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Offload optimizer state to CPU",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "optimizer_offload_fraction": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Ratio of optimizer state to offload to CPU",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "output_bert_embeddings": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Output Bert embeddings (via mean pooling) from model, rather than its binary head"
            " output or entire hidden batch."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "overlap_cpu_optimizer_d2h_h2d": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Overlap CPU optimizer step, gradients D2H and updated parameters H2D.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "overlap_grad_reduce": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, overlap DDP grad reduce.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "overlap_moe_expert_parallel_comm": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Overlap the EP A2A communication by batch-level overlapping in 1f1b stage.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "overlap_p2p_comm": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="overlap pipeline parallel communication with forward and backward chunks in 1F1B",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "overlap_p2p_comm_warmup_flush": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="if set, overlap pipeline parallel communication in warmup and flush",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "overlap_param_gather": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, overlap param all-gather in distributed optimizer.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "overlap_param_gather_with_optimizer_step": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, overlap param all-gather of first bucket with optimizer step.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "override_opt_param_scheduler": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Reset the values of the scheduler (learning rate,warmup iterations, minimum learning"
            " rate, maximum number of iterations, and decay style from input arguments and ignore"
            " values from checkpoints. Notethat all the above values will be reset."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "padded_vocab_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Vocabulary size of the model (padded to be divisible by tensor model parallel size)."
            " If not provided, it will be automatically calculated from vocab-size."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "patch_dim": MegatronArgMetadata(
        arg_type=int,
        default=16,
        help="patch dimension",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "per_split_data_args_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Path to per-split-data-args. Instead of feeding `--(train|valid|test)-data-path` with"
            " weighted dataset, we pass in a file path from which we read those arguments. This is"
            " useful when the list of data is too big. Format is a json file with `train`, `valid,"
            " `test` keys"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "perform_initialization": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "Do not perform initialization when building model, can reduce startup time when"
            " definitely loading from a checkpoint"
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "pin_cpu_grads": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable pinning of CPU memory for gradients.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "pin_cpu_params": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disable pinning of CPU memory for parameters.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "pipeline_model_parallel_comm_backend": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Select a communicator backend for pipeline parallel communication. If None, the"
            " default backend will be used."
        ),
        choices=("nccl", "ucc"),
        nargs=None,
        element_type=None,
    ),
    "pipeline_model_parallel_layout": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "A string that describes a custom pipeline model parallel layout. e.g.,"
            ' "E|(t|)*3,m|m||L". E, L, t, m denotes embedding, loss, transformer decoder layer, and'
            ' mtp layer, respectively. Stages are split by "|". Replicated stages or layers can be'
            " described with multiplication. Commas can be used cosmetically. Default None is not"
            " using this argument to set the layout."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "pipeline_model_parallel_size": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Degree of pipeline model parallelism.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "position_embedding_type": MegatronArgMetadata(
        arg_type=str,
        default="learned_absolute",
        help="Position embedding type.",
        choices=("learned_absolute", "rope", "mrope", "relative", "none"),
        nargs=None,
        element_type=None,
    ),
    "pretrained_checkpoint": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Directory containing a pretrained model checkpoint for finetuning.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "profile": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Enable nsys profiling. When using this option, nsys options should be specified in"
            " commandline. An example nsys commandline is `nsys profile -s none -t nvtx,cuda -o"
            " <path/to/output_file> --force-overwrite true --capture-range=cudaProfilerApi"
            " --capture-range-end=stop`."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "profile_ranks": MegatronArgMetadata(
        arg_type=list,
        default=[0],
        help="Global ranks to profile.",
        choices=None,
        nargs="+",
        element_type=int,
    ),
    "profile_step_end": MegatronArgMetadata(
        arg_type=int,
        default=12,
        help="Global step to stop profiling.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "profile_step_start": MegatronArgMetadata(
        arg_type=int,
        default=10,
        help="Global step to start profiling.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "q_lora_rank": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Rank of Query tensor's low rank representation.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "qk_head_dim": MegatronArgMetadata(
        arg_type=int,
        default=128,
        help=(
            "Dimension of the head in the QK projection. q_head_dim = qk_head_dim +"
            " qk_pos_emb_head_dim"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "qk_l2_norm": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use llama 4 qk l2 norm",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "qk_layernorm": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Whether to layer normalize the q and k attention embeddings.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "qk_pos_emb_head_dim": MegatronArgMetadata(
        arg_type=int,
        default=64,
        help="Dimension of the position embedding in the QK projection.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "query_in_block_prob": MegatronArgMetadata(
        arg_type=float,
        default=0.1,
        help="Probability of keeping query in block for ICT dataset",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "rampup_batch_size": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help=(
            "Batch size ramp up with the following values:  --rampup-batch-size <start batch size> "
            "                      <batch size incerement>                       <ramp-up samples>"
            " For example:   --rampup-batch-size 16 8 300000 \\    --global-batch-size 1024will"
            " start with global batch size 16 and over  (1024 - 16) / 8 = 126 intervals will"
            " increasethe batch size linearly to 1024. In each intervalwe will use approximately"
            " 300000 / 126 = 2380 samples."
        ),
        choices=None,
        nargs="*",
        element_type=str,
    ),
    "recompute_activations": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "recompute activation to allow for training with larger models, sequences, and batch"
            " sizes."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "recompute_granularity": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Checkpoint activations to allow for training with larger models, sequences, and batch"
            " sizes. It is supported at two granularities 1) full: whole transformer layer is"
            " recomputed, 2) selective: submodules set in --recompute-modules are recomputed,"
            " default is core_attn."
        ),
        choices=("full", "selective"),
        nargs=None,
        element_type=None,
    ),
    "recompute_method": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "1) uniform: uniformly divide the total number of Transformer layers and recompute the"
            " input activation of each divided chunk at specified granularity, 2) recompute the"
            " input activations of only a set number of individual Transformer layers per pipeline"
            " stage and do the rest without any recomputing at specified granularitydefault) do not"
            " apply activations recompute to any layers"
        ),
        choices=("uniform", "block"),
        nargs=None,
        element_type=None,
    ),
    "recompute_modules": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help=(
            'The submodules to recompute. choices: "core_attn", "moe_act", "layernorm",'
            ' "mla_up_proj",          "mlp", "moe", "shared_experts". default:'
            ' ["core_attn"]."core_attn": recompute the core attention part of the transformer'
            ' layer. "moe_act": recompute the MoE MLP activation function. "layernorm": recompute'
            ' the input_layernorm and pre_mlp_layernorm. "mla_up_proj": recompute the MLA up'
            ' projection and RoPE applying parts."mlp": recompute the dense MLP layer."moe":'
            ' recompute the MoE layer."shared_experts": recompute the shared experts in the MoE'
            ' layer."moe_act", "layernorm", and "mla_up_proj" use output-discarding checkpointing,'
            ' "core_attn", "mlp", "moe", and "shared_experts" use normal checkpointing.'
        ),
        choices=None,
        nargs="*",
        element_type=str,
    ),
    "recompute_num_layers": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "1) uniform: the number of Transformer layers in each uniformly divided recompute unit,"
            " 2) block: the number of individual Transformer layers to recompute within each"
            " pipeline stage."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "record_memory_history": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Record memory history in last rank.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "relative_attention_max_distance": MegatronArgMetadata(
        arg_type=int,
        default=128,
        help="Maximum distance for relative position embeddings calculation.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "relative_attention_num_buckets": MegatronArgMetadata(
        arg_type=int,
        default=32,
        help="Number of buckets for relative position embeddings.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "replication": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, replication of local checkpoints is enabled. Needs to be enabled on all ranks."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "replication_factor": MegatronArgMetadata(
        arg_type=int,
        default=2,
        help="Number of machines storing the replica of a given rank's data.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "replication_jump": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Specifies `J`, the spacing between ranks storing replicas of a given rank's data."
            " Replicas for rank `n` may be on ranks `n+J`, `n+2J`, ..., or `n-J`, `n-2J`, etc. This"
            " flag has an effect only if --replication is used. and must be consistent across all"
            " ranks."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "rerun_mode": MegatronArgMetadata(
        arg_type=str,
        default="validate_results",
        help=(
            "Use re-run engine to validate results (default) or to emit stats on variability of"
            " computations due to non-deterministic algorithms."
        ),
        choices=("disabled", "validate_results", "report_stats"),
        nargs=None,
        element_type=None,
    ),
    "reset_attention_mask": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Reset self attention maske after end-of-document token.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "reset_position_ids": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Reset posistion ids after end-of-document token.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "result_rejected_tracker_filename": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Optional name of file tracking `result_rejected` events.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retriever_report_topk_accuracies": MegatronArgMetadata(
        arg_type=list,
        default=[],
        help="Which top-k accuracies to report (e.g. '1 5 20')",
        choices=None,
        nargs="+",
        element_type=int,
    ),
    "retriever_score_scaling": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Whether to scale retriever scores by inverse square root of hidden size",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "retriever_seq_length": MegatronArgMetadata(
        arg_type=int,
        default=256,
        help="Maximum sequence length for the biencoder model for retriever",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_add_retriever": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Add a retriever to the transformer, for use in pretraining a Retro model.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "retro_attention_gate": MegatronArgMetadata(
        arg_type=float,
        default=1,
        help="Gated cross attention.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_cyclic_train_iters": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Set number of training iterations for cyclic Retro training.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_encoder_attention_dropout": MegatronArgMetadata(
        arg_type=float,
        default=0.1,
        help="Attention dropout for retrieval encoder.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_encoder_hidden_dropout": MegatronArgMetadata(
        arg_type=float,
        default=0.1,
        help="Hidden dropout for retrieval encoder.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_encoder_layers": MegatronArgMetadata(
        arg_type=int,
        default=2,
        help="Number of layers to use for the retrieval encoder.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_num_neighbors": MegatronArgMetadata(
        arg_type=int,
        default=2,
        help="Number of neighbors to retrieve during pretraining.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_num_retrieved_chunks": MegatronArgMetadata(
        arg_type=int,
        default=2,
        help="Number of chunks to retrieve from the retrieval database.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_project_dir": MegatronArgMetadata(
        arg_type=None,
        default=None,
        help=(
            "Retro project directory, which contains the preprocessed data for pretraining. This"
            " directory is built during preprocessing (see tools/retro/README.md), and contains"
            " subdirectories for the chunk database and pretraining neighbors."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "retro_verify_neighbor_count": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Skip verifying that len(GPT dataset) == len(saved neighbors).",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "reuse_grad_buf_for_mxfp8_param_ag": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If True, reuse the grad buffer for MXFP8 parameter all-gather.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "rope_scaling_factor": MegatronArgMetadata(
        arg_type=float,
        default=8.0,
        help="Rope scaling factor in llama3.x models",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "rope_type": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Type of rope to use. Note that MLA takes yarn by default, and common attention takes"
            " rope by default."
        ),
        choices=("rope", "yarn"),
        nargs=None,
        element_type=None,
    ),
    "rotary_base": MegatronArgMetadata(
        arg_type=int,
        default=10000,
        help="Base to use for rotary positional embeddings, default 10000",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "rotary_interleaved": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use interleaved rotary embedding.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "rotary_percent": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Percent of rotary dimension to use, default 100%%",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "rotary_scaling_factor": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="Rotary scaling factor for the rotary embeddings.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "rotary_seq_len_interpolation_factor": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Sequence length interpolation factor for rotary embeddings.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "run_workload_inspector_server": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="If set, enables workload inspector server for on-demand profiling.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "sample_rate": MegatronArgMetadata(
        arg_type=float,
        default=1.0,
        help="sample rate for training data. Supposed to be 0  < sample_rate < 1",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "save": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Output directory to save checkpoints to.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "save_interval": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Number of iterations between persistent checkpoint saves.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "save_retain_interval": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Number of iterations between retained checkpoints (othercheckpoints _except the last"
            " checkpoint_ are automatically deleted)."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "scatter_gather_tensors_in_pipeline": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="If not set, use scatter/gather to optimize communication of tensors in pipeline.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "seed": MegatronArgMetadata(
        arg_type=int,
        default=1234,
        help="Random seed used for python, numpy, pytorch, and cuda.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "seq_length": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Maximum sequence length to process.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "sequence_parallel": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enable sequence parallel optimization.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "sft": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Megatron SFT training",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "sft_tokenizer_prompt_format": MegatronArgMetadata(
        arg_type=str,
        default="nemotron-h-aligned",
        help="SFT prompt format.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "sgd_momentum": MegatronArgMetadata(
        arg_type=float,
        default=0.9,
        help="Momentum factor for sgd",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "sharp_enabled_group": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "IB SHARP can be enabled from only one communication group. By default, it is enabled"
            " from dp group. Available options: [dp, dp_replica]"
        ),
        choices=("dp", "dp_replica"),
        nargs=None,
        element_type=None,
    ),
    "short_seq_prob": MegatronArgMetadata(
        arg_type=float,
        default=0.1,
        help="Probability of producing a short sequence.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "skip_train": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, bypass the training loop, optionally do evaluation for validation/test, and"
            " exit."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "spec": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help=(
            "Specify the <module_location function_name> pair that returns a spec to customize a"
            " model, transformer block, or transformer layer, depending on the use case.To use"
            " local spec specify local as the argument.For more details, see the model class,"
            " `transformer_block.py`, or `transformer_layer.py`"
        ),
        choices=None,
        nargs="*",
        element_type=str,
    ),
    "split": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "Comma-separated list of proportions for training, validation, and test split. For"
            " example the split `90,5,5` will use 90%% of data for training, 5%% for validation and"
            " 5%% for test."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "squared_relu": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use squared relu activation instead of default gelu",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "start_weight_decay": MegatronArgMetadata(
        arg_type=float,
        default=None,
        help="Initial weight decay coefficient for L2 regularization.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "straggler_ctrlr_port": MegatronArgMetadata(
        arg_type=int,
        default=65535,
        help="Port number to toggle StragglerDetector on/off at runtime",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "straggler_minmax_count": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Number of ranks to report with high/low estimated throughput",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "strict_fsdp_dtensor_load": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Do not strict loading for fsdp_dtensor checkpoint format.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "suggested_communication_unit_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Specifies the number of elements to communicate at once during FSDP (Fully Sharded"
            " Data Parallel) operations. This flag also affects FSDP all-gather prefetch behavior."
            " Setting a larger value increases the communication buffer size, while a smaller value"
            " disables prefetching and may degrade performance. Adjust this value based on your"
            " system's memory and performance requirements."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "swiglu": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use gated linear units and SiLU activation instead of default gelu",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "swin_backbone_type": MegatronArgMetadata(
        arg_type=str,
        default="tiny",
        help="pretraining objectives",
        choices=("tiny", "base", "h3"),
        nargs=None,
        element_type=None,
    ),
    "symmetric_ar_type": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help=(
            "What type of symmetric all reduce to use. The default is none which is no use of"
            " symetric memory"
        ),
        choices=("two_shot", "one_shot", "multimem_all_reduce", None),
        nargs=None,
        element_type=None,
    ),
    "te_rng_tracker": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use the Transformer Engine version of the random number generator. Required for CUDA"
            " graphs support."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tensor_model_parallel_size": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Degree of tensor model parallelism.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "tensorboard_dir": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Write TensorBoard logs to this directory.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "tensorboard_log_interval": MegatronArgMetadata(
        arg_type=int,
        default=1,
        help="Report to tensorboard interval.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "tensorboard_queue_size": MegatronArgMetadata(
        arg_type=int,
        default=1000,
        help=(
            'Size of the tensorboard queue for pending events and summaries before one of the "add"'
            " calls forces a flush to disk."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "test_data_path": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help=(
            "The weight and prefix list for an independent test dataset. Follows the same pattern"
            " rules as --data-path."
        ),
        choices=None,
        nargs="*",
        element_type=str,
    ),
    "test_mode": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Run all real-time test alongside the experiment.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tiktoken_num_special_tokens": MegatronArgMetadata(
        arg_type=int,
        default=1000,
        help="Number of special tokens in tiktoken tokenizer",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "tiktoken_pattern": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Which tiktoken pattern to use. Options: [v1, v2]",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "tiktoken_special_tokens": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help='List of tiktoken special tokens, needs to have ["<unk>", "<s>", "</s>"]',
        choices=None,
        nargs="+",
        element_type=str,
    ),
    "timing_log_level": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help=(
            "Granularity level to measure and report timing.    0: report only iteration time and"
            " make sure timing       does not introduce extra overhead.   1: report timing for"
            " operations that are executed       very limited times (basically once) during      "
            " each iteration (such as gradient all-reduce)    2: report timing for operations that"
            " migh be       executed numerous times during each iteration. Note that setting the"
            " level to 1 or 2 might cause increase in iteration time."
        ),
        choices=(0, 1, 2),
        nargs=None,
        element_type=None,
    ),
    "timing_log_option": MegatronArgMetadata(
        arg_type=str,
        default="minmax",
        help=(
            "Options for logging timing:  max: report the max timing across all ranks  minmax:"
            " report min and max timings across all ranks  all: report timings of all ranks."
        ),
        choices=("max", "minmax", "all"),
        nargs=None,
        element_type=None,
    ),
    "titles_data_path": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Path to titles dataset used for ICT",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "tokenizer_model": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Sentencepiece tokenizer model.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "legacy_tokenizer": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use Megatron-LM legacy tokenizer system (works offline without HuggingFace).",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tokenizer_type": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="What type of tokenizer to use.",
        choices=(
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "SentencePieceTokenizer",
            "GPTSentencePieceTokenizer",
            "HuggingFaceTokenizer",
            "Llama2Tokenizer",
            "TikTokenizer",
            "MultimodalTokenizer",
            "NullTokenizer",
            "NullMultimodalTokenizer",
            "SFTTokenizer",
        ),
        nargs=None,
        element_type=None,
    ),
    "torch_fsdp2_reshard_after_forward": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "Whether to reshard weights after forward pass when using PyTorch FSDP2. Set to enable"
            " FSDP ZeRO-2."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tp_comm_bootstrap_backend": MegatronArgMetadata(
        arg_type=str,
        default="nccl",
        help="Set the bootstrapping backend of Tensor parallel communications.",
        choices=("nccl", "mpi", "gloo"),
        nargs=None,
        element_type=None,
    ),
    "tp_comm_bulk_dgrad": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disables the All-Gather overlap with bprop activation gradient GEMM.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tp_comm_bulk_wgrad": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tp_comm_overlap": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enables the  overlap of Tensor parallel communication and GEMM kernels.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tp_comm_overlap_ag": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disables the All-Gather overlap with GEMM by pipelining the GEMM and All-Gather.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tp_comm_overlap_cfg": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Config file when tp_comm_overlap is enabled.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "tp_comm_overlap_rs": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help=(
            "Disables the Reduce-Scatter overlap with GEMM by pipelining the GEMM and"
            " Reduce-Scatter."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tp_comm_overlap_rs_dgrad": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enables the Reduce-Scatter overlap with dgrad GEMM.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tp_comm_split_ag": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disables the All-Gather overlap with fprop GEMM.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "tp_comm_split_rs": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="Disables the Reduce-Scatter overlap with fprop GEMM.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "train_data_path": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help=(
            "The weight and prefix list for an independent train dataset. Follows the same pattern"
            " rules as --data-path."
        ),
        choices=None,
        nargs="*",
        element_type=str,
    ),
    "train_iters": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Total number of iterations to train over all training runs. Note that either"
            " train-iters or train-samples should be provided."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "train_samples": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Total number of samples to train over all training runs. Note that either train-iters"
            " or train-samples should be provided."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "train_sync_interval": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help=(
            "Training CPU-GPU synchronization interval, to ensure that CPU is not running too far"
            " ahead of GPU."
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "transformer_impl": MegatronArgMetadata(
        arg_type=None,
        default="transformer_engine",
        help="Which Transformer implementation to use.",
        choices=("local", "transformer_engine"),
        nargs=None,
        element_type=None,
    ),
    "untie_embeddings_and_output_weights": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Untie embeddings and output weights.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_checkpoint_args": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Override model-related command-line arguments with arguments from checkpoint",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_checkpoint_opt_param_scheduler": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use checkpoint to set the values of the scheduler (learning rate, warmup iterations,"
            " minimum learning rate, maximum number of iterations, and decay style from checkpoint"
            " and ignore input arguments."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_cpu_initialization": MegatronArgMetadata(
        arg_type=bool,
        default=None,
        help=(
            "If set, initialize weights on the CPU. This eliminates init differences based on"
            " tensor parallelism."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_dist_ckpt_deprecated": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Deprecated: see --ckpt-format.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_distributed_optimizer": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use distributed optimizer.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_flash_attn": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="use FlashAttention implementation of attention. https://arxiv.org/abs/2205.14135",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_fused_weighted_squared_relu": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use fused weighted squared relu when using MoE.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_legacy_models": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use the legacy Megatron models, not Megatron-Core models.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_megatron_fsdp": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use the Megatron FSDP code path in DDP.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_mp_args_from_checkpoint_args": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Copy model parallelism command-line arguments from checkpoint",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_one_sent_docs": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Whether to use one sentence documents in ICT",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_persistent_ckpt_worker": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Enables a persitent checkpoint worker for async save",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_precision_aware_optimizer": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use the precision-aware optimizer in TransformerEngine, which allows setting the main"
            " params and optimizer states to lower precision, such as fp16, bf16 and fp8."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_pytorch_profiler": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use the built-in pytorch profiler. Useful if you wish to view profiles in tensorboard."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_ring_exchange_p2p": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, use custom-built ring exchange for p2p communications. Note that this option"
            " will require a custom built image that support ring-exchange p2p."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_rope_scaling": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Apply rope scaling as used in llama3.x",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_rotary_position_embeddings": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Use rotary positional embeddings or not. Deprecated: use --position-embedding-type",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_sharp": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="Required to enable SHARP communication.",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_tokenizer_model_from_checkpoint_args": MegatronArgMetadata(
        arg_type=bool,
        default=True,
        help="If set, do not use tokenizer model path from checkpoint",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_torch_fsdp2": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use the torch FSDP2 implementation. FSDP2 has not been tested with pipeline"
            " parallelism, and may contain bugs."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_torch_optimizer_for_cpu_offload": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "Use torch.optim.Optimizer instead of Megatron's optimizer in optimizer cpu offload"
            " mode."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "use_tp_pp_dp_mapping": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help=(
            "If set, distributed ranks initialize order is changed from tp-cp-ep-dp-pp to"
            " tp-cp-ep-pp-dp."
        ),
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "v_head_dim": MegatronArgMetadata(
        arg_type=int,
        default=128,
        help="Dimension of the head in the V projection.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "valid_data_path": MegatronArgMetadata(
        arg_type=list,
        default=None,
        help=(
            "The weight and prefix list for an independent validation dataset. Follows the same"
            " pattern rules as --data-path."
        ),
        choices=None,
        nargs="*",
        element_type=str,
    ),
    "vision_backbone_type": MegatronArgMetadata(
        arg_type=str,
        default="vit",
        help="backbone types types",
        choices=("vit", "mit", "swin"),
        nargs=None,
        element_type=None,
    ),
    "vision_pretraining": MegatronArgMetadata(
        arg_type=bool,
        default=False,
        help="flag to indicate vision pretraining",
        choices=None,
        nargs=0,
        element_type=None,
    ),
    "vision_pretraining_type": MegatronArgMetadata(
        arg_type=str,
        default="classify",
        help="pretraining objectives",
        choices=("classify", "inpaint", "dino"),
        nargs=None,
        element_type=None,
    ),
    "vocab_extra_ids": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help=(
            "Number of additional vocabulary tokens. They are used for span masking in the T5 model"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "vocab_file": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Path to the vocab file.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "vocab_size": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Size of vocab before EOD or padding.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "wandb_exp_name": MegatronArgMetadata(
        arg_type=str,
        default="",
        help="The wandb experiment name.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "wandb_project": MegatronArgMetadata(
        arg_type=str,
        default="",
        help="The wandb project name. Ignore wandb by default.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "wandb_save_dir": MegatronArgMetadata(
        arg_type=str,
        default="",
        help="Path to save the wandb results locally.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "warmup": MegatronArgMetadata(
        arg_type=int,
        default=None,
        help="Old lr warmup argument, do not use. Use one of the--lr-warmup-* arguments above",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "weight_decay": MegatronArgMetadata(
        arg_type=float,
        default=0.01,
        help="Weight decay coefficient for L2 regularization.",
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "weight_decay_incr_style": MegatronArgMetadata(
        arg_type=str,
        default="constant",
        help="Weight decay increment function.",
        choices=("constant", "linear", "cosine"),
        nargs=None,
        element_type=None,
    ),
    "wgrad_deferral_limit": MegatronArgMetadata(
        arg_type=int,
        default=0,
        help=(
            "Number of micro-batches for whichweight gradient computation of vocabulary projection"
            " is deferred, defaults to 0 whichmeans all the micro-batches are deferred. Invalid if"
            " `defer-embedding-wgrad-compute`is not set"
        ),
        choices=None,
        nargs=None,
        element_type=None,
    ),
    "yaml_cfg": MegatronArgMetadata(
        arg_type=str,
        default=None,
        help="Config file to add additional arguments",
        choices=None,
        nargs=None,
        element_type=None,
    ),
}

MEGATRON_ACTION_SPECS: Mapping[str, MegatronActionSpec] = {
    "account_for_embedding_in_pipeline_split": MegatronActionSpec(
        option_strings=("--account-for-embedding-in-pipeline-split",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "account_for_loss_in_pipeline_split": MegatronActionSpec(
        option_strings=("--account-for-loss-in-pipeline-split",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "accumulate_allreduce_grads_in_fp32": MegatronActionSpec(
        option_strings=("--accumulate-allreduce-grads-in-fp32",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "adam_beta1": MegatronActionSpec(
        option_strings=("--adam-beta1",), action_type="store", nargs=None, const=None, default=0.9
    ),
    "adam_beta2": MegatronActionSpec(
        option_strings=("--adam-beta2",), action_type="store", nargs=None, const=None, default=0.999
    ),
    "adam_eps": MegatronActionSpec(
        option_strings=("--adam-eps",), action_type="store", nargs=None, const=None, default=1e-08
    ),
    "add_bias_linear": MegatronActionSpec(
        option_strings=("--disable-bias-linear",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "add_position_embedding": MegatronActionSpec(
        option_strings=("--no-position-embedding",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "add_qkv_bias": MegatronActionSpec(
        option_strings=("--add-qkv-bias",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "adlr_autoresume": MegatronActionSpec(
        option_strings=("--adlr-autoresume",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "adlr_autoresume_interval": MegatronActionSpec(
        option_strings=("--adlr-autoresume-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=1000,
    ),
    "align_grad_reduce": MegatronActionSpec(
        option_strings=("--no-align-grad-reduce",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "align_param_gather": MegatronActionSpec(
        option_strings=("--no-align-param-gather",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "app_tag_run_name": MegatronActionSpec(
        option_strings=("--app-tag-run-name",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "app_tag_run_version": MegatronActionSpec(
        option_strings=("--app-tag-run-version",),
        action_type="store",
        nargs=None,
        const=None,
        default="0.0.0",
    ),
    "apply_layernorm_1p": MegatronActionSpec(
        option_strings=("--apply-layernorm-1p",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "apply_query_key_layer_scaling": MegatronActionSpec(
        option_strings=("--apply-query-key-layer-scaling",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "apply_residual_connection_post_layernorm": MegatronActionSpec(
        option_strings=("--apply-residual-connection-post-layernorm",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "apply_rope_fusion": MegatronActionSpec(
        option_strings=("--no-rope-fusion",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "async_save": MegatronActionSpec(
        option_strings=("--async-save",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=None,
    ),
    "async_tensor_model_parallel_allreduce": MegatronActionSpec(
        option_strings=("--no-async-tensor-model-parallel-allreduce",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "attention_backend": MegatronActionSpec(
        option_strings=("--attention-backend",),
        action_type="store",
        nargs=None,
        const=None,
        default="auto",
    ),
    "attention_dropout": MegatronActionSpec(
        option_strings=("--attention-dropout",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.1,
    ),
    "attention_softmax_in_fp32": MegatronActionSpec(
        option_strings=("--attention-softmax-in-fp32",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "auto_detect_ckpt_format": MegatronActionSpec(
        option_strings=("--auto-detect-ckpt-format",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "barrier_with_L1_time": MegatronActionSpec(
        option_strings=("--no-barrier-with-level-1-timing",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "batch_size": MegatronActionSpec(
        option_strings=("--batch-size",), action_type="store", nargs=None, const=None, default=None
    ),
    "bert_binary_head": MegatronActionSpec(
        option_strings=("--bert-no-binary-head",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "bert_embedder_type": MegatronActionSpec(
        option_strings=("--bert-embedder-type",),
        action_type="store",
        nargs=None,
        const=None,
        default="megatron",
    ),
    "bert_load": MegatronActionSpec(
        option_strings=("--bert-load",), action_type="store", nargs=None, const=None, default=None
    ),
    "bf16": MegatronActionSpec(
        option_strings=("--bf16",), action_type="store_true", nargs=0, const=True, default=False
    ),
    "bias_dropout_fusion": MegatronActionSpec(
        option_strings=("--no-bias-dropout-fusion",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "bias_gelu_fusion": MegatronActionSpec(
        option_strings=("--no-bias-gelu-fusion",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "bias_swiglu_fusion": MegatronActionSpec(
        option_strings=("--no-bias-swiglu-fusion",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "biencoder_projection_dim": MegatronActionSpec(
        option_strings=("--biencoder-projection-dim",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "biencoder_shared_query_context_model": MegatronActionSpec(
        option_strings=("--biencoder-shared-query-context-model",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "block_data_path": MegatronActionSpec(
        option_strings=("--block-data-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "cache_mla_latents": MegatronActionSpec(
        option_strings=("--cache-mla-latents",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "calc_ft_timeouts": MegatronActionSpec(
        option_strings=("--calc-ft-timeouts",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "calculate_per_token_loss": MegatronActionSpec(
        option_strings=("--calculate-per-token-loss",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "check_for_large_grads": MegatronActionSpec(
        option_strings=("--check-for-large-grads",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "check_for_nan_in_loss_and_grad": MegatronActionSpec(
        option_strings=("--no-check-for-nan-in-loss-and-grad",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "check_for_spiky_loss": MegatronActionSpec(
        option_strings=("--check-for-spiky-loss",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "check_weight_hash_across_dp_replicas_interval": MegatronActionSpec(
        option_strings=("--check-weight-hash-across-dp-replicas-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "checkpoint_activations": MegatronActionSpec(
        option_strings=("--checkpoint-activations",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "ckpt_assume_constant_structure": MegatronActionSpec(
        option_strings=("--ckpt-assume-constant-structure",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "ckpt_convert_format": MegatronActionSpec(
        option_strings=("--ckpt-convert-format",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "ckpt_convert_save": MegatronActionSpec(
        option_strings=("--ckpt-convert-save",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "ckpt_convert_update_legacy_dist_opt_format": MegatronActionSpec(
        option_strings=("--ckpt-convert-update-legacy-dist-opt-format",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "ckpt_format": MegatronActionSpec(
        option_strings=("--ckpt-format",),
        action_type="store",
        nargs=None,
        const=None,
        default="torch_dist",
    ),
    "ckpt_fully_parallel_load": MegatronActionSpec(
        option_strings=("--ckpt-fully-parallel-load",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "ckpt_fully_parallel_save": MegatronActionSpec(
        option_strings=("--no-ckpt-fully-parallel-save",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "ckpt_fully_parallel_save_deprecated": MegatronActionSpec(
        option_strings=("--ckpt-fully-parallel-save",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "ckpt_step": MegatronActionSpec(
        option_strings=("--ckpt-step",), action_type="store", nargs=None, const=None, default=None
    ),
    "classes_fraction": MegatronActionSpec(
        option_strings=("--classes-fraction",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "clip_grad": MegatronActionSpec(
        option_strings=("--clip-grad",), action_type="store", nargs=None, const=None, default=1.0
    ),
    "clone_scatter_output_in_embedding": MegatronActionSpec(
        option_strings=("--no-clone-scatter-output-in-embedding",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "config_logger_dir": MegatronActionSpec(
        option_strings=("--config-logger-dir",),
        action_type="store",
        nargs=None,
        const=None,
        default="",
    ),
    "context_parallel_size": MegatronActionSpec(
        option_strings=("--context-parallel-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "cp_comm_type": MegatronActionSpec(
        option_strings=("--cp-comm-type",),
        action_type="store",
        nargs="+",
        const=None,
        default=["p2p"],
    ),
    "create_attention_mask_in_dataloader": MegatronActionSpec(
        option_strings=("--no-create-attention-mask-in-dataloader",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "cross_entropy_fusion_impl": MegatronActionSpec(
        option_strings=("--cross-entropy-fusion-impl",),
        action_type="store",
        nargs=None,
        const=None,
        default="native",
    ),
    "cross_entropy_loss_fusion": MegatronActionSpec(
        option_strings=("--cross-entropy-loss-fusion",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "cuda_graph_scope": MegatronActionSpec(
        option_strings=("--cuda-graph-scope",),
        action_type="store",
        nargs=None,
        const=None,
        default="full",
    ),
    "cuda_graph_warmup_steps": MegatronActionSpec(
        option_strings=("--cuda-graph-warmup-steps",),
        action_type="store",
        nargs=None,
        const=None,
        default=3,
    ),
    "data_args_path": MegatronActionSpec(
        option_strings=("--data-args-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "data_cache_path": MegatronActionSpec(
        option_strings=("--data-cache-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "data_parallel_random_init": MegatronActionSpec(
        option_strings=("--data-parallel-random-init",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "data_parallel_sharding_strategy": MegatronActionSpec(
        option_strings=("--data-parallel-sharding-strategy",),
        action_type="store",
        nargs=None,
        const=None,
        default="no_shard",
    ),
    "data_path": MegatronActionSpec(
        option_strings=("--data-path",), action_type="store", nargs="*", const=None, default=None
    ),
    "data_per_class_fraction": MegatronActionSpec(
        option_strings=("--data-per-class-fraction",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "data_sharding": MegatronActionSpec(
        option_strings=("--no-data-sharding",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "dataloader_type": MegatronActionSpec(
        option_strings=("--dataloader-type",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "ddp_average_in_collective": MegatronActionSpec(
        option_strings=("--ddp-average-in-collective",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "ddp_bucket_size": MegatronActionSpec(
        option_strings=("--ddp-bucket-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "ddp_num_buckets": MegatronActionSpec(
        option_strings=("--ddp-num-buckets",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "ddp_pad_buckets_for_high_nccl_busbw": MegatronActionSpec(
        option_strings=("--ddp-pad-buckets-for-high-nccl-busbw",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "decoder_first_pipeline_num_layers": MegatronActionSpec(
        option_strings=("--decoder-first-pipeline-num-layers",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "decoder_last_pipeline_num_layers": MegatronActionSpec(
        option_strings=("--decoder-last-pipeline-num-layers",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "decoder_num_layers": MegatronActionSpec(
        option_strings=("--decoder-num-layers",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "decoder_seq_length": MegatronActionSpec(
        option_strings=("--decoder-seq-length",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "decoupled_lr": MegatronActionSpec(
        option_strings=("--decoupled-lr",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "decoupled_min_lr": MegatronActionSpec(
        option_strings=("--decoupled-min-lr",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "decrease_batch_size_if_needed": MegatronActionSpec(
        option_strings=("--decrease-batch-size-if-needed",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "defer_embedding_wgrad_compute": MegatronActionSpec(
        option_strings=("--defer-embedding-wgrad-compute",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "delay_wgrad_compute": MegatronActionSpec(
        option_strings=("--delay-wgrad-compute",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "deprecated_use_mcore_models": MegatronActionSpec(
        option_strings=("--use-mcore-models",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "deterministic_mode": MegatronActionSpec(
        option_strings=("--deterministic-mode",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "dino_bottleneck_size": MegatronActionSpec(
        option_strings=("--dino-bottleneck-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=256,
    ),
    "dino_freeze_last_layer": MegatronActionSpec(
        option_strings=("--dino-freeze-last-layer",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "dino_head_hidden_size": MegatronActionSpec(
        option_strings=("--dino-head-hidden-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=2048,
    ),
    "dino_local_crops_number": MegatronActionSpec(
        option_strings=("--dino-local-crops-number",),
        action_type="store",
        nargs=None,
        const=None,
        default=10,
    ),
    "dino_local_img_size": MegatronActionSpec(
        option_strings=("--dino-local-img-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=96,
    ),
    "dino_norm_last_layer": MegatronActionSpec(
        option_strings=("--dino-norm-last-layer",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "dino_teacher_temp": MegatronActionSpec(
        option_strings=("--dino-teacher-temp",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.07,
    ),
    "dino_warmup_teacher_temp": MegatronActionSpec(
        option_strings=("--dino-warmup-teacher-temp",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.04,
    ),
    "dino_warmup_teacher_temp_epochs": MegatronActionSpec(
        option_strings=("--dino-warmup-teacher-temp-epochs",),
        action_type="store",
        nargs=None,
        const=None,
        default=30,
    ),
    "disable_bf16_reduced_precision_matmul": MegatronActionSpec(
        option_strings=("--disable-bf16-reduced-precision-matmul",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "disable_mamba_mem_eff_path": MegatronActionSpec(
        option_strings=("--disable-mamba-mem-eff-path",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "disable_straggler_on_startup": MegatronActionSpec(
        option_strings=("--disable-straggler-on-startup",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "dist_ckpt_format_deprecated": MegatronActionSpec(
        option_strings=("--dist-ckpt-format",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "dist_ckpt_strictness": MegatronActionSpec(
        option_strings=("--dist-ckpt-strictness",),
        action_type="store",
        nargs=None,
        const=None,
        default="assume_ok_unexpected",
    ),
    "distribute_saved_activations": MegatronActionSpec(
        option_strings=("--distribute-saved-activations",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "distributed_backend": MegatronActionSpec(
        option_strings=("--distributed-backend",),
        action_type="store",
        nargs=None,
        const=None,
        default="nccl",
    ),
    "distributed_timeout_minutes": MegatronActionSpec(
        option_strings=("--distributed-timeout-minutes",),
        action_type="store",
        nargs=None,
        const=None,
        default=10,
    ),
    "embedding_init_method_std": MegatronActionSpec(
        option_strings=("--embedding-init-method-std",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "embedding_path": MegatronActionSpec(
        option_strings=("--embedding-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "empty_unused_memory_level": MegatronActionSpec(
        option_strings=("--empty-unused-memory-level",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "enable_cuda_graph": MegatronActionSpec(
        option_strings=("--enable-cuda-graph",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "enable_experimental": MegatronActionSpec(
        option_strings=("--enable-experimental",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "enable_ft_package": MegatronActionSpec(
        option_strings=("--enable-ft-package",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "enable_full_sharding_in_hsdp": MegatronActionSpec(
        option_strings=("--enable-full-sharding-in-hsdp",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "enable_gloo_process_groups": MegatronActionSpec(
        option_strings=("--disable-gloo-process-groups",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "enable_msc": MegatronActionSpec(
        option_strings=("--disable-msc",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "enable_one_logger": MegatronActionSpec(
        option_strings=("--no-one-logger",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "encoder_num_layers": MegatronActionSpec(
        option_strings=("--encoder-num-layers",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "encoder_seq_length": MegatronActionSpec(
        option_strings=("--encoder-seq-length",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "end_weight_decay": MegatronActionSpec(
        option_strings=("--end-weight-decay",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "eod_mask_loss": MegatronActionSpec(
        option_strings=("--eod-mask-loss",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "error_injection_rate": MegatronActionSpec(
        option_strings=("--error-injection-rate",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "error_injection_type": MegatronActionSpec(
        option_strings=("--error-injection-type",),
        action_type="store",
        nargs=None,
        const=None,
        default="transient_error",
    ),
    "eval_interval": MegatronActionSpec(
        option_strings=("--eval-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=1000,
    ),
    "eval_iters": MegatronActionSpec(
        option_strings=("--eval-iters",), action_type="store", nargs=None, const=None, default=100
    ),
    "evidence_data_path": MegatronActionSpec(
        option_strings=("--evidence-data-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "exit_duration_in_mins": MegatronActionSpec(
        option_strings=("--exit-duration-in-mins",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "exit_interval": MegatronActionSpec(
        option_strings=("--exit-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "exit_on_missing_checkpoint": MegatronActionSpec(
        option_strings=("--exit-on-missing-checkpoint",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "exit_signal_handler": MegatronActionSpec(
        option_strings=("--exit-signal-handler",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "exp_avg_dtype": MegatronActionSpec(
        option_strings=("--exp-avg-dtype",),
        action_type="store",
        nargs=None,
        const=None,
        default="fp32",
    ),
    "exp_avg_sq_dtype": MegatronActionSpec(
        option_strings=("--exp-avg-sq-dtype",),
        action_type="store",
        nargs=None,
        const=None,
        default="fp32",
    ),
    "expert_model_parallel_size": MegatronActionSpec(
        option_strings=("--expert-model-parallel-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "expert_tensor_parallel_size": MegatronActionSpec(
        option_strings=("--expert-tensor-parallel-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "external_cuda_graph": MegatronActionSpec(
        option_strings=("--external-cuda-graph",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "ffn_hidden_size": MegatronActionSpec(
        option_strings=("--ffn-hidden-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "finetune": MegatronActionSpec(
        option_strings=("--finetune",), action_type="store_true", nargs=0, const=True, default=False
    ),
    "first_last_layers_bf16": MegatronActionSpec(
        option_strings=("--first-last-layers-bf16",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "flash_decode": MegatronActionSpec(
        option_strings=("--flash-decode",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "fp16": MegatronActionSpec(
        option_strings=("--fp16",), action_type="store_true", nargs=0, const=True, default=False
    ),
    "fp16_lm_cross_entropy": MegatronActionSpec(
        option_strings=("--fp16-lm-cross-entropy",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "fp32_residual_connection": MegatronActionSpec(
        option_strings=("--fp32-residual-connection",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "fp8": MegatronActionSpec(
        option_strings=("--fp8-format",), action_type="store", nargs=None, const=None, default=None
    ),
    "fp8_amax_compute_algo": MegatronActionSpec(
        option_strings=("--fp8-amax-compute-algo",),
        action_type="store",
        nargs=None,
        const=None,
        default="most_recent",
    ),
    "fp8_amax_history_len": MegatronActionSpec(
        option_strings=("--fp8-amax-history-len",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "fp8_interval": MegatronActionSpec(
        option_strings=("--fp8-interval",), action_type="store", nargs=None, const=None, default=1
    ),
    "fp8_margin": MegatronActionSpec(
        option_strings=("--fp8-margin",), action_type="store", nargs=None, const=None, default=0
    ),
    "fp8_param_gather": MegatronActionSpec(
        option_strings=("--fp8-param-gather",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "fp8_recipe": MegatronActionSpec(
        option_strings=("--fp8-recipe",),
        action_type="store",
        nargs=None,
        const=None,
        default="delayed",
    ),
    "fp8_wgrad": MegatronActionSpec(
        option_strings=("--no-fp8-wgrad",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "fsdp_double_buffer": MegatronActionSpec(
        option_strings=("--fsdp-double-buffer",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "full_validation": MegatronActionSpec(
        option_strings=("--full-validation",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "global_batch_size": MegatronActionSpec(
        option_strings=("--global-batch-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "grad_reduce_in_bf16": MegatronActionSpec(
        option_strings=("--grad-reduce-in-bf16",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "gradient_accumulation_fusion": MegatronActionSpec(
        option_strings=("--no-gradient-accumulation-fusion",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "gradient_reduce_div_fusion": MegatronActionSpec(
        option_strings=("--no-gradient-reduce-div-fusion",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "group_query_attention": MegatronActionSpec(
        option_strings=("--group-query-attention",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "head_lr_mult": MegatronActionSpec(
        option_strings=("--head-lr-mult",), action_type="store", nargs=None, const=None, default=1.0
    ),
    "heterogeneous_layers_config_encoded_json": MegatronActionSpec(
        option_strings=("--heterogeneous-layers-config-encoded-json",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "heterogeneous_layers_config_path": MegatronActionSpec(
        option_strings=("--heterogeneous-layers-config-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "hidden_dropout": MegatronActionSpec(
        option_strings=("--hidden-dropout",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.1,
    ),
    "hidden_size": MegatronActionSpec(
        option_strings=("--hidden-size",), action_type="store", nargs=None, const=None, default=None
    ),
    "hierarchical_context_parallel_sizes": MegatronActionSpec(
        option_strings=("--hierarchical-context-parallel-sizes",),
        action_type="store",
        nargs="+",
        const=None,
        default=None,
    ),
    "high_priority_stream_groups": MegatronActionSpec(
        option_strings=("--high-priority-stream-groups",),
        action_type="store",
        nargs="*",
        const=None,
        default=[],
    ),
    "hybrid_attention_ratio": MegatronActionSpec(
        option_strings=("--hybrid-attention-ratio",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.0,
    ),
    "hybrid_mlp_ratio": MegatronActionSpec(
        option_strings=("--hybrid-mlp-ratio",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.0,
    ),
    "hybrid_override_pattern": MegatronActionSpec(
        option_strings=("--hybrid-override-pattern",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "hysteresis": MegatronActionSpec(
        option_strings=("--hysteresis",), action_type="store", nargs=None, const=None, default=2
    ),
    "ict_head_size": MegatronActionSpec(
        option_strings=("--ict-head-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "ict_load": MegatronActionSpec(
        option_strings=("--ict-load",), action_type="store", nargs=None, const=None, default=None
    ),
    "img_h": MegatronActionSpec(
        option_strings=("--img-h",), action_type="store", nargs=None, const=None, default=224
    ),
    "img_w": MegatronActionSpec(
        option_strings=("--img-w",), action_type="store", nargs=None, const=None, default=224
    ),
    "indexer_batch_size": MegatronActionSpec(
        option_strings=("--indexer-batch-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=128,
    ),
    "indexer_log_interval": MegatronActionSpec(
        option_strings=("--indexer-log-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=1000,
    ),
    "inference_batch_times_seqlen_threshold": MegatronActionSpec(
        option_strings=("--inference-batch-times-seqlen-threshold",),
        action_type="store",
        nargs=None,
        const=None,
        default=-1,
    ),
    "inference_dynamic_batching": MegatronActionSpec(
        option_strings=("--inference-dynamic-batching",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "inference_dynamic_batching_buffer_guaranteed_fraction": MegatronActionSpec(
        option_strings=("--inference-dynamic-batching-buffer-guaranteed-fraction",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.2,
    ),
    "inference_dynamic_batching_buffer_overflow_factor": MegatronActionSpec(
        option_strings=("--inference-dynamic-batching-buffer-overflow-factor",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "inference_dynamic_batching_buffer_size_gb": MegatronActionSpec(
        option_strings=("--inference-dynamic-batching-buffer-size-gb",),
        action_type="store",
        nargs=None,
        const=None,
        default=40.0,
    ),
    "inference_dynamic_batching_chunk_size": MegatronActionSpec(
        option_strings=("--inference-dynamic-batching-chunk-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=256,
    ),
    "inference_dynamic_batching_max_requests_override": MegatronActionSpec(
        option_strings=("--inference-dynamic-batching-max-requests-override",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "inference_dynamic_batching_max_tokens_override": MegatronActionSpec(
        option_strings=("--inference-dynamic-batching-max-tokens-override",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "inference_dynamic_batching_num_cuda_graphs": MegatronActionSpec(
        option_strings=("--inference-dynamic-batching-num-cuda-graphs",),
        action_type="store",
        nargs=None,
        const=None,
        default=16,
    ),
    "inference_max_batch_size": MegatronActionSpec(
        option_strings=("--inference-max-requests",),
        action_type="store",
        nargs=None,
        const=None,
        default=8,
    ),
    "inference_max_seq_length": MegatronActionSpec(
        option_strings=("--inference-max-seq-length",),
        action_type="store",
        nargs=None,
        const=None,
        default=2560,
    ),
    "inference_rng_tracker": MegatronActionSpec(
        option_strings=("--inference-rng-tracker",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "init_method_std": MegatronActionSpec(
        option_strings=("--init-method-std",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.02,
    ),
    "init_method_xavier_uniform": MegatronActionSpec(
        option_strings=("--init-method-xavier-uniform",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "init_model_with_meta_device": MegatronActionSpec(
        option_strings=("--init-model-with-meta-device",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "initial_loss_scale": MegatronActionSpec(
        option_strings=("--initial-loss-scale",),
        action_type="store",
        nargs=None,
        const=None,
        default=4294967296,
    ),
    "inprocess_active_world_size": MegatronActionSpec(
        option_strings=("--inprocess-active-world-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "inprocess_barrier_timeout": MegatronActionSpec(
        option_strings=("--inprocess-barrier-timeout",),
        action_type="store",
        nargs=None,
        const=None,
        default=120,
    ),
    "inprocess_completion_timeout": MegatronActionSpec(
        option_strings=("--inprocess-completion-timeout",),
        action_type="store",
        nargs=None,
        const=None,
        default=120,
    ),
    "inprocess_empty_cuda_cache": MegatronActionSpec(
        option_strings=("--inprocess-empty-cuda-cache",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "inprocess_granularity": MegatronActionSpec(
        option_strings=("--inprocess-granularity",),
        action_type="store",
        nargs=None,
        const=None,
        default="node",
    ),
    "inprocess_hard_timeout": MegatronActionSpec(
        option_strings=("--inprocess-hard-timeout",),
        action_type="store",
        nargs=None,
        const=None,
        default=90,
    ),
    "inprocess_heartbeat_interval": MegatronActionSpec(
        option_strings=("--inprocess-heartbeat-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=30,
    ),
    "inprocess_heartbeat_timeout": MegatronActionSpec(
        option_strings=("--inprocess-heartbeat-timeout",),
        action_type="store",
        nargs=None,
        const=None,
        default=60,
    ),
    "inprocess_last_call_wait": MegatronActionSpec(
        option_strings=("--inprocess-last-call-wait",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "inprocess_max_iterations": MegatronActionSpec(
        option_strings=("--inprocess-max-iterations",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "inprocess_monitor_process_interval": MegatronActionSpec(
        option_strings=("--inprocess-monitor-process-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "inprocess_monitor_thread_interval": MegatronActionSpec(
        option_strings=("--inprocess-monitor-thread-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "inprocess_progress_watchdog_interval": MegatronActionSpec(
        option_strings=("--inprocess-progress-watchdog-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "inprocess_restart": MegatronActionSpec(
        option_strings=("--inprocess-restart",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "inprocess_soft_timeout": MegatronActionSpec(
        option_strings=("--inprocess-soft-timeout",),
        action_type="store",
        nargs=None,
        const=None,
        default=60,
    ),
    "inprocess_termination_grace_time": MegatronActionSpec(
        option_strings=("--inprocess-termination-grace-time",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "is_hybrid_model": MegatronActionSpec(
        option_strings=("--is-hybrid-model",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "iter_per_epoch": MegatronActionSpec(
        option_strings=("--iter-per-epoch",),
        action_type="store",
        nargs=None,
        const=None,
        default=1250,
    ),
    "iterations_to_skip": MegatronActionSpec(
        option_strings=("--iterations-to-skip",),
        action_type="store",
        nargs="+",
        const=None,
        default=[],
    ),
    "keep_fp8_transpose_cache": MegatronActionSpec(
        option_strings=("--keep-fp8-transpose-cache",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "kitchen_config_file": MegatronActionSpec(
        option_strings=("--kitchen-config-file",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "kitchen_recipe_number": MegatronActionSpec(
        option_strings=("--kitchen-recipe-number",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "kv_channels": MegatronActionSpec(
        option_strings=("--kv-channels",), action_type="store", nargs=None, const=None, default=None
    ),
    "kv_lora_rank": MegatronActionSpec(
        option_strings=("--kv-lora-rank",), action_type="store", nargs=None, const=None, default=32
    ),
    "lazy_mpu_init": MegatronActionSpec(
        option_strings=("--lazy-mpu-init",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "load": MegatronActionSpec(
        option_strings=("--load",), action_type="store", nargs=None, const=None, default=None
    ),
    "load_main_params_from_ckpt": MegatronActionSpec(
        option_strings=("--load-main-params-from-ckpt",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=None,
    ),
    "load_model_opt_format": MegatronActionSpec(
        option_strings=("--load-model-opt-format",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "local_rank": MegatronActionSpec(
        option_strings=("--local-rank",), action_type="store", nargs=None, const=None, default=0
    ),
    "log_energy": MegatronActionSpec(
        option_strings=("--log-energy",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_interval": MegatronActionSpec(
        option_strings=("--log-interval",), action_type="store", nargs=None, const=None, default=100
    ),
    "log_loss_scale_to_tensorboard": MegatronActionSpec(
        option_strings=("--no-log-loss-scale-to-tensorboard",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "log_memory_to_tensorboard": MegatronActionSpec(
        option_strings=("--log-memory-to-tensorboard",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_num_zeros_in_grad": MegatronActionSpec(
        option_strings=("--log-num-zeros-in-grad",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_params_norm": MegatronActionSpec(
        option_strings=("--log-params-norm",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_progress": MegatronActionSpec(
        option_strings=("--log-progress",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_straggler": MegatronActionSpec(
        option_strings=("--log-straggler",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_throughput": MegatronActionSpec(
        option_strings=("--log-throughput",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_timers_to_tensorboard": MegatronActionSpec(
        option_strings=("--log-timers-to-tensorboard",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_validation_ppl_to_tensorboard": MegatronActionSpec(
        option_strings=("--log-validation-ppl-to-tensorboard",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "log_world_size_to_tensorboard": MegatronActionSpec(
        option_strings=("--log-world-size-to-tensorboard",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "logging_level": MegatronActionSpec(
        option_strings=("--logging-level",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "loss_scale": MegatronActionSpec(
        option_strings=("--loss-scale",), action_type="store", nargs=None, const=None, default=None
    ),
    "loss_scale_window": MegatronActionSpec(
        option_strings=("--loss-scale-window",),
        action_type="store",
        nargs=None,
        const=None,
        default=1000,
    ),
    "lr": MegatronActionSpec(
        option_strings=("--lr",), action_type="store", nargs=None, const=None, default=None
    ),
    "lr_decay_iters": MegatronActionSpec(
        option_strings=("--lr-decay-iters",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "lr_decay_samples": MegatronActionSpec(
        option_strings=("--lr-decay-samples",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "lr_decay_style": MegatronActionSpec(
        option_strings=("--lr-decay-style",),
        action_type="store",
        nargs=None,
        const=None,
        default="linear",
    ),
    "lr_warmup_fraction": MegatronActionSpec(
        option_strings=("--lr-warmup-fraction",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "lr_warmup_init": MegatronActionSpec(
        option_strings=("--lr-warmup-init",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.0,
    ),
    "lr_warmup_iters": MegatronActionSpec(
        option_strings=("--lr-warmup-iters",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "lr_warmup_samples": MegatronActionSpec(
        option_strings=("--lr-warmup-samples",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "lr_wsd_decay_iters": MegatronActionSpec(
        option_strings=("--lr-wsd-decay-iters",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "lr_wsd_decay_samples": MegatronActionSpec(
        option_strings=("--lr-wsd-decay-samples",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "lr_wsd_decay_style": MegatronActionSpec(
        option_strings=("--lr-wsd-decay-style",),
        action_type="store",
        nargs=None,
        const=None,
        default="exponential",
    ),
    "main_grads_dtype": MegatronActionSpec(
        option_strings=("--main-grads-dtype",),
        action_type="store",
        nargs=None,
        const=None,
        default="fp32",
    ),
    "main_params_dtype": MegatronActionSpec(
        option_strings=("--main-params-dtype",),
        action_type="store",
        nargs=None,
        const=None,
        default="fp32",
    ),
    "make_vocab_size_divisible_by": MegatronActionSpec(
        option_strings=("--make-vocab-size-divisible-by",),
        action_type="store",
        nargs=None,
        const=None,
        default=128,
    ),
    "mamba_head_dim": MegatronActionSpec(
        option_strings=("--mamba-head-dim",),
        action_type="store",
        nargs=None,
        const=None,
        default=64,
    ),
    "mamba_num_groups": MegatronActionSpec(
        option_strings=("--mamba-num-groups",),
        action_type="store",
        nargs=None,
        const=None,
        default=8,
    ),
    "mamba_num_heads": MegatronActionSpec(
        option_strings=("--mamba-num-heads",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "mamba_state_dim": MegatronActionSpec(
        option_strings=("--mamba-state-dim",),
        action_type="store",
        nargs=None,
        const=None,
        default=128,
    ),
    "manual_gc": MegatronActionSpec(
        option_strings=("--manual-gc",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "manual_gc_eval": MegatronActionSpec(
        option_strings=("--no-manual-gc-eval",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "manual_gc_interval": MegatronActionSpec(
        option_strings=("--manual-gc-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "mask_factor": MegatronActionSpec(
        option_strings=("--mask-factor",), action_type="store", nargs=None, const=None, default=1.0
    ),
    "mask_prob": MegatronActionSpec(
        option_strings=("--mask-prob",), action_type="store", nargs=None, const=None, default=0.15
    ),
    "mask_type": MegatronActionSpec(
        option_strings=("--mask-type",),
        action_type="store",
        nargs=None,
        const=None,
        default="random",
    ),
    "masked_softmax_fusion": MegatronActionSpec(
        option_strings=("--no-masked-softmax-fusion",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "max_position_embeddings": MegatronActionSpec(
        option_strings=("--max-position-embeddings",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "max_tokens_to_oom": MegatronActionSpec(
        option_strings=("--max-tokens-to-oom",),
        action_type="store",
        nargs=None,
        const=None,
        default=12000,
    ),
    "memory_snapshot_path": MegatronActionSpec(
        option_strings=("--memory-snapshot-path",),
        action_type="store",
        nargs=None,
        const=None,
        default="snapshot.pickle",
    ),
    "merge_file": MegatronActionSpec(
        option_strings=("--merge-file",), action_type="store", nargs=None, const=None, default=None
    ),
    "micro_batch_size": MegatronActionSpec(
        option_strings=("--micro-batch-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "microbatch_group_size_per_vp_stage": MegatronActionSpec(
        option_strings=("--microbatch-group-size-per-virtual-pipeline-stage",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "mid_level_dataset_surplus": MegatronActionSpec(
        option_strings=("--mid-level-dataset-surplus",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.005,
    ),
    "min_loss_scale": MegatronActionSpec(
        option_strings=("--min-loss-scale",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "min_lr": MegatronActionSpec(
        option_strings=("--min-lr",), action_type="store", nargs=None, const=None, default=0.0
    ),
    "mlp_chunks_for_prefill": MegatronActionSpec(
        option_strings=("--mlp-chunks-for-prefill",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "mmap_bin_files": MegatronActionSpec(
        option_strings=("--no-mmap-bin-files",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "mock_data": MegatronActionSpec(
        option_strings=("--mock-data",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "model_parallel_size": MegatronActionSpec(
        option_strings=("--model-parallel-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_apply_probs_on_input": MegatronActionSpec(
        option_strings=("--moe-apply-probs-on-input",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_aux_loss_coeff": MegatronActionSpec(
        option_strings=("--moe-aux-loss-coeff",),
        action_type="store",
        nargs="+",
        const=None,
        default=0.0,
    ),
    "moe_deepep_num_sms": MegatronActionSpec(
        option_strings=("--moe-deepep-num-sms",),
        action_type="store",
        nargs=None,
        const=None,
        default=20,
    ),
    "moe_enable_deepep": MegatronActionSpec(
        option_strings=("--moe-enable-deepep",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_expert_capacity_factor": MegatronActionSpec(
        option_strings=("--moe-expert-capacity-factor",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_extended_tp": MegatronActionSpec(
        option_strings=("--moe-extended-tp",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_ffn_hidden_size": MegatronActionSpec(
        option_strings=("--moe-ffn-hidden-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_grouped_gemm": MegatronActionSpec(
        option_strings=("--moe-grouped-gemm",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_input_jitter_eps": MegatronActionSpec(
        option_strings=("--moe-input-jitter-eps",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_layer_freq": MegatronActionSpec(
        option_strings=("--moe-layer-freq",), action_type="store", nargs=None, const=None, default=1
    ),
    "moe_layer_recompute": MegatronActionSpec(
        option_strings=("--moe-layer-recompute",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_pad_expert_input_to_capacity": MegatronActionSpec(
        option_strings=("--moe-pad-expert-input-to-capacity",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_per_layer_logging": MegatronActionSpec(
        option_strings=("--moe-per-layer-logging",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_permute_fusion": MegatronActionSpec(
        option_strings=("--moe-permute-fusion",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_router_bias_update_rate": MegatronActionSpec(
        option_strings=("--moe-router-bias-update-rate",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.001,
    ),
    "moe_router_dtype": MegatronActionSpec(
        option_strings=("--moe-router-dtype",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_router_enable_expert_bias": MegatronActionSpec(
        option_strings=("--moe-router-enable-expert-bias",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_router_force_load_balancing": MegatronActionSpec(
        option_strings=("--moe-router-force-load-balancing",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_router_fusion": MegatronActionSpec(
        option_strings=("--moe-router-fusion",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_router_group_topk": MegatronActionSpec(
        option_strings=("--moe-router-group-topk",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_router_load_balancing_type": MegatronActionSpec(
        option_strings=("--moe-router-load-balancing-type",),
        action_type="store",
        nargs="+",
        const=None,
        default="aux_loss",
    ),
    "moe_router_num_groups": MegatronActionSpec(
        option_strings=("--moe-router-num-groups",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_router_padding_for_fp8": MegatronActionSpec(
        option_strings=("--moe-router-padding-for-fp8",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_router_pre_softmax": MegatronActionSpec(
        option_strings=("--moe-router-pre-softmax",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_router_score_function": MegatronActionSpec(
        option_strings=("--moe-router-score-function",),
        action_type="store",
        nargs=None,
        const=None,
        default="softmax",
    ),
    "moe_router_topk": MegatronActionSpec(
        option_strings=("--moe-router-topk",),
        action_type="store",
        nargs=None,
        const=None,
        default=2,
    ),
    "moe_router_topk_scaling_factor": MegatronActionSpec(
        option_strings=("--moe-router-topk-scaling-factor",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_shared_expert_intermediate_size": MegatronActionSpec(
        option_strings=("--moe-shared-expert-intermediate-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "moe_shared_expert_overlap": MegatronActionSpec(
        option_strings=("--moe-shared-expert-overlap",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_token_dispatcher_type": MegatronActionSpec(
        option_strings=("--moe-token-dispatcher-type",),
        action_type="store",
        nargs=None,
        const=None,
        default="allgather",
    ),
    "moe_token_drop_policy": MegatronActionSpec(
        option_strings=("--moe-token-drop-policy",),
        action_type="store",
        nargs=None,
        const=None,
        default="probs",
    ),
    "moe_upcycling_granularity": MegatronActionSpec(
        option_strings=("--moe-upcycling-granularity",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "moe_use_legacy_grouped_gemm": MegatronActionSpec(
        option_strings=("--moe-use-legacy-grouped-gemm",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_use_upcycling": MegatronActionSpec(
        option_strings=("--moe-use-upcycling",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "moe_z_loss_coeff": MegatronActionSpec(
        option_strings=("--moe-z-loss-coeff",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "mrope_section": MegatronActionSpec(
        option_strings=("--mrope-section",),
        action_type="store",
        nargs="+",
        const=None,
        default=None,
    ),
    "mscale": MegatronActionSpec(
        option_strings=("--mscale",), action_type="store", nargs=None, const=None, default=1.0
    ),
    "mscale_all_dim": MegatronActionSpec(
        option_strings=("--mscale-all-dim",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.0,
    ),
    "mtp_loss_scaling_factor": MegatronActionSpec(
        option_strings=("--mtp-loss-scaling-factor",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.1,
    ),
    "mtp_num_layers": MegatronActionSpec(
        option_strings=("--mtp-num-layers",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "multi_latent_attention": MegatronActionSpec(
        option_strings=("--multi-latent-attention",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "multiple_validation_sets": MegatronActionSpec(
        option_strings=("--multiple-validation-sets",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "nccl_all_reduce_for_prefill": MegatronActionSpec(
        option_strings=("--nccl-all-reduce-for-prefill",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "nccl_communicator_config_path": MegatronActionSpec(
        option_strings=("--nccl-communicator-config-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "nccl_ub": MegatronActionSpec(
        option_strings=("--use-nccl-ub",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "no_load_optim": MegatronActionSpec(
        option_strings=("--no-load-optim",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=None,
    ),
    "no_load_rng": MegatronActionSpec(
        option_strings=("--no-load-rng",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=None,
    ),
    "no_persist_layer_norm": MegatronActionSpec(
        option_strings=("--no-persist-layer-norm",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "no_rope_freq": MegatronActionSpec(
        option_strings=("--no-rope-freq",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "no_save_optim": MegatronActionSpec(
        option_strings=("--no-save-optim",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=None,
    ),
    "no_save_rng": MegatronActionSpec(
        option_strings=("--no-save-rng",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=None,
    ),
    "non_persistent_ckpt_type": MegatronActionSpec(
        option_strings=("--non-persistent-ckpt-type",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "non_persistent_global_ckpt_dir": MegatronActionSpec(
        option_strings=("--non-persistent-global-ckpt-dir",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "non_persistent_local_ckpt_algo": MegatronActionSpec(
        option_strings=("--non-persistent-local-ckpt-algo",),
        action_type="store",
        nargs=None,
        const=None,
        default="fully_parallel",
    ),
    "non_persistent_local_ckpt_dir": MegatronActionSpec(
        option_strings=("--non-persistent-local-ckpt-dir",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "non_persistent_save_interval": MegatronActionSpec(
        option_strings=("--non-persistent-save-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "norm_epsilon": MegatronActionSpec(
        option_strings=("--norm-epsilon",),
        action_type="store",
        nargs=None,
        const=None,
        default=1e-05,
    ),
    "normalization": MegatronActionSpec(
        option_strings=("--normalization",),
        action_type="store",
        nargs=None,
        const=None,
        default="LayerNorm",
    ),
    "num_attention_heads": MegatronActionSpec(
        option_strings=("--num-attention-heads",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "num_channels": MegatronActionSpec(
        option_strings=("--num-channels",), action_type="store", nargs=None, const=None, default=3
    ),
    "num_classes": MegatronActionSpec(
        option_strings=("--num-classes",), action_type="store", nargs=None, const=None, default=1000
    ),
    "num_dataset_builder_threads": MegatronActionSpec(
        option_strings=("--num-dataset-builder-threads",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "num_distributed_optimizer_instances": MegatronActionSpec(
        option_strings=("--num-distributed-optimizer-instances",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "num_experts": MegatronActionSpec(
        option_strings=("--num-experts",), action_type="store", nargs=None, const=None, default=None
    ),
    "num_layers": MegatronActionSpec(
        option_strings=("--num-layers",), action_type="store", nargs=None, const=None, default=None
    ),
    "num_layers_at_end_in_bf16": MegatronActionSpec(
        option_strings=("--num-layers-at-end-in-bf16",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "num_layers_at_start_in_bf16": MegatronActionSpec(
        option_strings=("--num-layers-at-start-in-bf16",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "num_layers_per_virtual_pipeline_stage": MegatronActionSpec(
        option_strings=("--num-layers-per-virtual-pipeline-stage",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "num_query_groups": MegatronActionSpec(
        option_strings=("--num-query-groups",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "num_virtual_stages_per_pipeline_rank": MegatronActionSpec(
        option_strings=("--num-virtual-stages-per-pipeline-rank",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "num_workers": MegatronActionSpec(
        option_strings=("--num-workers",), action_type="store", nargs=None, const=None, default=2
    ),
    "object_storage_cache_path": MegatronActionSpec(
        option_strings=("--object-storage-cache-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "one_logger_async": MegatronActionSpec(
        option_strings=("--one-logger-async",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "one_logger_project": MegatronActionSpec(
        option_strings=("--one-logger-project",),
        action_type="store",
        nargs=None,
        const=None,
        default="megatron-lm",
    ),
    "one_logger_run_name": MegatronActionSpec(
        option_strings=("--one-logger-run-name",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "onnx_safe": MegatronActionSpec(
        option_strings=("--onnx-safe",), action_type="store", nargs=None, const=None, default=None
    ),
    "openai_gelu": MegatronActionSpec(
        option_strings=("--openai-gelu",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "optimizer": MegatronActionSpec(
        option_strings=("--optimizer",), action_type="store", nargs=None, const=None, default="adam"
    ),
    "optimizer_cpu_offload": MegatronActionSpec(
        option_strings=("--optimizer-cpu-offload",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "optimizer_offload_fraction": MegatronActionSpec(
        option_strings=("--optimizer-offload-fraction",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "output_bert_embeddings": MegatronActionSpec(
        option_strings=("--output-bert-embeddings",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "overlap_cpu_optimizer_d2h_h2d": MegatronActionSpec(
        option_strings=("--overlap-cpu-optimizer-d2h-h2d",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "overlap_grad_reduce": MegatronActionSpec(
        option_strings=("--overlap-grad-reduce",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "overlap_moe_expert_parallel_comm": MegatronActionSpec(
        option_strings=("--overlap-moe-expert-parallel-comm",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "overlap_p2p_comm": MegatronActionSpec(
        option_strings=("--no-overlap-p2p-communication",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "overlap_p2p_comm_warmup_flush": MegatronActionSpec(
        option_strings=("--overlap-p2p-communication-warmup-flush",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "overlap_param_gather": MegatronActionSpec(
        option_strings=("--overlap-param-gather",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "overlap_param_gather_with_optimizer_step": MegatronActionSpec(
        option_strings=("--overlap-param-gather-with-optimizer-step",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "override_opt_param_scheduler": MegatronActionSpec(
        option_strings=("--override-opt_param-scheduler",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "padded_vocab_size": MegatronActionSpec(
        option_strings=("--padded-vocab-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "patch_dim": MegatronActionSpec(
        option_strings=("--patch-dim",), action_type="store", nargs=None, const=None, default=16
    ),
    "per_split_data_args_path": MegatronActionSpec(
        option_strings=("--per-split-data-args-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "perform_initialization": MegatronActionSpec(
        option_strings=("--no-initialization",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "pin_cpu_grads": MegatronActionSpec(
        option_strings=("--no-pin-cpu-grads",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "pin_cpu_params": MegatronActionSpec(
        option_strings=("--no-pin-cpu-params",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "pipeline_model_parallel_comm_backend": MegatronActionSpec(
        option_strings=("--pipeline-model-parallel-comm-backend",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "pipeline_model_parallel_layout": MegatronActionSpec(
        option_strings=("--pipeline-model-parallel-layout",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "pipeline_model_parallel_size": MegatronActionSpec(
        option_strings=("--pipeline-model-parallel-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "position_embedding_type": MegatronActionSpec(
        option_strings=("--position-embedding-type",),
        action_type="store",
        nargs=None,
        const=None,
        default="learned_absolute",
    ),
    "pretrained_checkpoint": MegatronActionSpec(
        option_strings=("--pretrained-checkpoint",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "profile": MegatronActionSpec(
        option_strings=("--profile",), action_type="store_true", nargs=0, const=True, default=False
    ),
    "profile_ranks": MegatronActionSpec(
        option_strings=("--profile-ranks",), action_type="store", nargs="+", const=None, default=[0]
    ),
    "profile_step_end": MegatronActionSpec(
        option_strings=("--profile-step-end",),
        action_type="store",
        nargs=None,
        const=None,
        default=12,
    ),
    "profile_step_start": MegatronActionSpec(
        option_strings=("--profile-step-start",),
        action_type="store",
        nargs=None,
        const=None,
        default=10,
    ),
    "q_lora_rank": MegatronActionSpec(
        option_strings=("--q-lora-rank",), action_type="store", nargs=None, const=None, default=None
    ),
    "qk_head_dim": MegatronActionSpec(
        option_strings=("--qk-head-dim",), action_type="store", nargs=None, const=None, default=128
    ),
    "qk_l2_norm": MegatronActionSpec(
        option_strings=("--qk-l2-norm",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "qk_layernorm": MegatronActionSpec(
        option_strings=("--qk-layernorm",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "qk_pos_emb_head_dim": MegatronActionSpec(
        option_strings=("--qk-pos-emb-head-dim",),
        action_type="store",
        nargs=None,
        const=None,
        default=64,
    ),
    "query_in_block_prob": MegatronActionSpec(
        option_strings=("--query-in-block-prob",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.1,
    ),
    "rampup_batch_size": MegatronActionSpec(
        option_strings=("--rampup-batch-size",),
        action_type="store",
        nargs="*",
        const=None,
        default=None,
    ),
    "recompute_activations": MegatronActionSpec(
        option_strings=("--recompute-activations",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "recompute_granularity": MegatronActionSpec(
        option_strings=("--recompute-granularity",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "recompute_method": MegatronActionSpec(
        option_strings=("--recompute-method",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "recompute_modules": MegatronActionSpec(
        option_strings=("--recompute-modules",),
        action_type="store",
        nargs="*",
        const=None,
        default=None,
    ),
    "recompute_num_layers": MegatronActionSpec(
        option_strings=("--recompute-num-layers",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "record_memory_history": MegatronActionSpec(
        option_strings=("--record-memory-history",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "relative_attention_max_distance": MegatronActionSpec(
        option_strings=("--relative-attention-max-distance",),
        action_type="store",
        nargs=None,
        const=None,
        default=128,
    ),
    "relative_attention_num_buckets": MegatronActionSpec(
        option_strings=("--relative-attention-num-buckets",),
        action_type="store",
        nargs=None,
        const=None,
        default=32,
    ),
    "replication": MegatronActionSpec(
        option_strings=("--replication",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "replication_factor": MegatronActionSpec(
        option_strings=("--replication-factor",),
        action_type="store",
        nargs=None,
        const=None,
        default=2,
    ),
    "replication_jump": MegatronActionSpec(
        option_strings=("--replication-jump",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "rerun_mode": MegatronActionSpec(
        option_strings=("--rerun-mode",),
        action_type="store",
        nargs=None,
        const=None,
        default="validate_results",
    ),
    "reset_attention_mask": MegatronActionSpec(
        option_strings=("--reset-attention-mask",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "reset_position_ids": MegatronActionSpec(
        option_strings=("--reset-position-ids",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "result_rejected_tracker_filename": MegatronActionSpec(
        option_strings=("--result-rejected-tracker-filename",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "retriever_report_topk_accuracies": MegatronActionSpec(
        option_strings=("--retriever-report-topk-accuracies",),
        action_type="store",
        nargs="+",
        const=None,
        default=[],
    ),
    "retriever_score_scaling": MegatronActionSpec(
        option_strings=("--retriever-score-scaling",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "retriever_seq_length": MegatronActionSpec(
        option_strings=("--retriever-seq-length",),
        action_type="store",
        nargs=None,
        const=None,
        default=256,
    ),
    "retro_add_retriever": MegatronActionSpec(
        option_strings=("--retro-add-retriever",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "retro_attention_gate": MegatronActionSpec(
        option_strings=("--retro-attention-gate",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "retro_cyclic_train_iters": MegatronActionSpec(
        option_strings=("--retro-cyclic-train-iters",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "retro_encoder_attention_dropout": MegatronActionSpec(
        option_strings=("--retro-encoder-attention-dropout",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.1,
    ),
    "retro_encoder_hidden_dropout": MegatronActionSpec(
        option_strings=("--retro-encoder-hidden-dropout",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.1,
    ),
    "retro_encoder_layers": MegatronActionSpec(
        option_strings=("--retro-encoder-layers",),
        action_type="store",
        nargs=None,
        const=None,
        default=2,
    ),
    "retro_num_neighbors": MegatronActionSpec(
        option_strings=("--retro-num-neighbors",),
        action_type="store",
        nargs=None,
        const=None,
        default=2,
    ),
    "retro_num_retrieved_chunks": MegatronActionSpec(
        option_strings=("--retro-num-retrieved-chunks",),
        action_type="store",
        nargs=None,
        const=None,
        default=2,
    ),
    "retro_project_dir": MegatronActionSpec(
        option_strings=("--retro-project-dir",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "retro_verify_neighbor_count": MegatronActionSpec(
        option_strings=("--retro-no-verify-neighbor-count",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "reuse_grad_buf_for_mxfp8_param_ag": MegatronActionSpec(
        option_strings=("--reuse-grad-buf-for-mxfp8-param-ag",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "rope_scaling_factor": MegatronActionSpec(
        option_strings=("--rope-scaling-factor",),
        action_type="store",
        nargs=None,
        const=None,
        default=8.0,
    ),
    "rope_type": MegatronActionSpec(
        option_strings=("--rope-type",), action_type="store", nargs=None, const=None, default=None
    ),
    "rotary_base": MegatronActionSpec(
        option_strings=("--rotary-base",),
        action_type="store",
        nargs=None,
        const=None,
        default=10000,
    ),
    "rotary_interleaved": MegatronActionSpec(
        option_strings=("--rotary-interleaved",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "rotary_percent": MegatronActionSpec(
        option_strings=("--rotary-percent",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "rotary_scaling_factor": MegatronActionSpec(
        option_strings=("--rotary-scaling-factor",),
        action_type="store",
        nargs=None,
        const=None,
        default=1.0,
    ),
    "rotary_seq_len_interpolation_factor": MegatronActionSpec(
        option_strings=("--rotary-seq-len-interpolation-factor",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "run_workload_inspector_server": MegatronActionSpec(
        option_strings=("--run-workload-inspector-server",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "sample_rate": MegatronActionSpec(
        option_strings=("--sample-rate",), action_type="store", nargs=None, const=None, default=1.0
    ),
    "save": MegatronActionSpec(
        option_strings=("--save",), action_type="store", nargs=None, const=None, default=None
    ),
    "save_interval": MegatronActionSpec(
        option_strings=("--save-interval", "--persistent-save-interval"),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "save_retain_interval": MegatronActionSpec(
        option_strings=("--save-retain-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "scatter_gather_tensors_in_pipeline": MegatronActionSpec(
        option_strings=("--no-scatter-gather-tensors-in-pipeline",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "seed": MegatronActionSpec(
        option_strings=("--seed",), action_type="store", nargs=None, const=None, default=1234
    ),
    "seq_length": MegatronActionSpec(
        option_strings=("--seq-length",), action_type="store", nargs=None, const=None, default=None
    ),
    "sequence_parallel": MegatronActionSpec(
        option_strings=("--sequence-parallel",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "sft": MegatronActionSpec(
        option_strings=("--sft",), action_type="store_true", nargs=0, const=True, default=False
    ),
    "sft_tokenizer_prompt_format": MegatronActionSpec(
        option_strings=("--sft-tokenizer-prompt-format",),
        action_type="store",
        nargs=None,
        const=None,
        default="nemotron-h-aligned",
    ),
    "sgd_momentum": MegatronActionSpec(
        option_strings=("--sgd-momentum",), action_type="store", nargs=None, const=None, default=0.9
    ),
    "sharp_enabled_group": MegatronActionSpec(
        option_strings=("--sharp-enabled-group",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "short_seq_prob": MegatronActionSpec(
        option_strings=("--short-seq-prob",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.1,
    ),
    "skip_train": MegatronActionSpec(
        option_strings=("--skip-train",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "spec": MegatronActionSpec(
        option_strings=("--spec",), action_type="store", nargs="*", const=None, default=None
    ),
    "split": MegatronActionSpec(
        option_strings=("--split",), action_type="store", nargs=None, const=None, default=None
    ),
    "squared_relu": MegatronActionSpec(
        option_strings=("--squared-relu",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "start_weight_decay": MegatronActionSpec(
        option_strings=("--start-weight-decay",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "straggler_ctrlr_port": MegatronActionSpec(
        option_strings=("--straggler-ctrlr-port",),
        action_type="store",
        nargs=None,
        const=None,
        default=65535,
    ),
    "straggler_minmax_count": MegatronActionSpec(
        option_strings=("--straggler-minmax-count",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "strict_fsdp_dtensor_load": MegatronActionSpec(
        option_strings=("--no-strict-fsdp-dtensor-load",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "suggested_communication_unit_size": MegatronActionSpec(
        option_strings=("--suggested-communication-unit-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "swiglu": MegatronActionSpec(
        option_strings=("--swiglu",), action_type="store_true", nargs=0, const=True, default=False
    ),
    "swin_backbone_type": MegatronActionSpec(
        option_strings=("--swin-backbone-type",),
        action_type="store",
        nargs=None,
        const=None,
        default="tiny",
    ),
    "symmetric_ar_type": MegatronActionSpec(
        option_strings=("--symmetric-ar-type",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "te_rng_tracker": MegatronActionSpec(
        option_strings=("--te-rng-tracker",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "tensor_model_parallel_size": MegatronActionSpec(
        option_strings=("--tensor-model-parallel-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "tensorboard_dir": MegatronActionSpec(
        option_strings=("--tensorboard-dir",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "tensorboard_log_interval": MegatronActionSpec(
        option_strings=("--tensorboard-log-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=1,
    ),
    "tensorboard_queue_size": MegatronActionSpec(
        option_strings=("--tensorboard-queue-size",),
        action_type="store",
        nargs=None,
        const=None,
        default=1000,
    ),
    "test_data_path": MegatronActionSpec(
        option_strings=("--test-data-path",),
        action_type="store",
        nargs="*",
        const=None,
        default=None,
    ),
    "test_mode": MegatronActionSpec(
        option_strings=("--test-mode",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "tiktoken_num_special_tokens": MegatronActionSpec(
        option_strings=("--tiktoken-num-special-tokens",),
        action_type="store",
        nargs=None,
        const=None,
        default=1000,
    ),
    "tiktoken_pattern": MegatronActionSpec(
        option_strings=("--tiktoken-pattern",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "tiktoken_special_tokens": MegatronActionSpec(
        option_strings=("--tiktoken-special-tokens",),
        action_type="store",
        nargs="+",
        const=None,
        default=None,
    ),
    "timing_log_level": MegatronActionSpec(
        option_strings=("--timing-log-level",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "timing_log_option": MegatronActionSpec(
        option_strings=("--timing-log-option",),
        action_type="store",
        nargs=None,
        const=None,
        default="minmax",
    ),
    "titles_data_path": MegatronActionSpec(
        option_strings=("--titles-data-path",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "tokenizer_model": MegatronActionSpec(
        option_strings=("--tokenizer-model",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "legacy_tokenizer": MegatronActionSpec(
        option_strings=("--legacy-tokenizer",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "tokenizer_type": MegatronActionSpec(
        option_strings=("--tokenizer-type",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "torch_fsdp2_reshard_after_forward": MegatronActionSpec(
        option_strings=("--torch-fsdp2-no-reshard-after-forward",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "tp_comm_bootstrap_backend": MegatronActionSpec(
        option_strings=("--tp-comm-bootstrap-backend",),
        action_type="store",
        nargs=None,
        const=None,
        default="nccl",
    ),
    "tp_comm_bulk_dgrad": MegatronActionSpec(
        option_strings=("--disable-tp-comm-bulk-dgrad",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "tp_comm_bulk_wgrad": MegatronActionSpec(
        option_strings=("--disable-tp-comm-bulk-wgrad",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "tp_comm_overlap": MegatronActionSpec(
        option_strings=("--tp-comm-overlap",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "tp_comm_overlap_ag": MegatronActionSpec(
        option_strings=("--disable-tp-comm-overlap-ag",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "tp_comm_overlap_cfg": MegatronActionSpec(
        option_strings=("--tp-comm-overlap-cfg",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "tp_comm_overlap_rs": MegatronActionSpec(
        option_strings=("--disable-tp-comm-overlap-rs",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "tp_comm_overlap_rs_dgrad": MegatronActionSpec(
        option_strings=("--tp-comm-overlap-rs-dgrad",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "tp_comm_split_ag": MegatronActionSpec(
        option_strings=("--disable-tp-comm-split-ag",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "tp_comm_split_rs": MegatronActionSpec(
        option_strings=("--disable-tp-comm-split-rs",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "train_data_path": MegatronActionSpec(
        option_strings=("--train-data-path",),
        action_type="store",
        nargs="*",
        const=None,
        default=None,
    ),
    "train_iters": MegatronActionSpec(
        option_strings=("--train-iters",), action_type="store", nargs=None, const=None, default=None
    ),
    "train_samples": MegatronActionSpec(
        option_strings=("--train-samples",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "train_sync_interval": MegatronActionSpec(
        option_strings=("--train-sync-interval",),
        action_type="store",
        nargs=None,
        const=None,
        default=None,
    ),
    "transformer_impl": MegatronActionSpec(
        option_strings=("--transformer-impl",),
        action_type="store",
        nargs=None,
        const=None,
        default="transformer_engine",
    ),
    "untie_embeddings_and_output_weights": MegatronActionSpec(
        option_strings=("--untie-embeddings-and-output-weights",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_checkpoint_args": MegatronActionSpec(
        option_strings=("--use-checkpoint-args",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_checkpoint_opt_param_scheduler": MegatronActionSpec(
        option_strings=("--use-checkpoint-opt_param-scheduler",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_cpu_initialization": MegatronActionSpec(
        option_strings=("--use-cpu-initialization",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=None,
    ),
    "use_dist_ckpt_deprecated": MegatronActionSpec(
        option_strings=("--use-dist-ckpt",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_distributed_optimizer": MegatronActionSpec(
        option_strings=("--use-distributed-optimizer",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_flash_attn": MegatronActionSpec(
        option_strings=("--use-flash-attn",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_fused_weighted_squared_relu": MegatronActionSpec(
        option_strings=("--use-fused-weighted-squared-relu",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_legacy_models": MegatronActionSpec(
        option_strings=("--use-legacy-models",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_megatron_fsdp": MegatronActionSpec(
        option_strings=("--use-megatron-fsdp",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_mp_args_from_checkpoint_args": MegatronActionSpec(
        option_strings=("--use-mp-args-from-checkpoint-args",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_one_sent_docs": MegatronActionSpec(
        option_strings=("--use-one-sent-docs",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_persistent_ckpt_worker": MegatronActionSpec(
        option_strings=("--use-persistent-ckpt-worker",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_precision_aware_optimizer": MegatronActionSpec(
        option_strings=("--use-precision-aware-optimizer",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_pytorch_profiler": MegatronActionSpec(
        option_strings=("--use-pytorch-profiler",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_ring_exchange_p2p": MegatronActionSpec(
        option_strings=("--use-ring-exchange-p2p",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_rope_scaling": MegatronActionSpec(
        option_strings=("--use-rope-scaling",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_rotary_position_embeddings": MegatronActionSpec(
        option_strings=("--use-rotary-position-embeddings",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_sharp": MegatronActionSpec(
        option_strings=("--use-sharp",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_tokenizer_model_from_checkpoint_args": MegatronActionSpec(
        option_strings=("--no-use-tokenizer-model-from-checkpoint-args",),
        action_type="store_false",
        nargs=0,
        const=False,
        default=True,
    ),
    "use_torch_fsdp2": MegatronActionSpec(
        option_strings=("--use-torch-fsdp2",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_torch_optimizer_for_cpu_offload": MegatronActionSpec(
        option_strings=("--use-torch-optimizer-for-cpu-offload",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "use_tp_pp_dp_mapping": MegatronActionSpec(
        option_strings=("--use-tp-pp-dp-mapping",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "v_head_dim": MegatronActionSpec(
        option_strings=("--v-head-dim",), action_type="store", nargs=None, const=None, default=128
    ),
    "valid_data_path": MegatronActionSpec(
        option_strings=("--valid-data-path",),
        action_type="store",
        nargs="*",
        const=None,
        default=None,
    ),
    "vision_backbone_type": MegatronActionSpec(
        option_strings=("--vision-backbone-type",),
        action_type="store",
        nargs=None,
        const=None,
        default="vit",
    ),
    "vision_pretraining": MegatronActionSpec(
        option_strings=("--vision-pretraining",),
        action_type="store_true",
        nargs=0,
        const=True,
        default=False,
    ),
    "vision_pretraining_type": MegatronActionSpec(
        option_strings=("--vision-pretraining-type",),
        action_type="store",
        nargs=None,
        const=None,
        default="classify",
    ),
    "vocab_extra_ids": MegatronActionSpec(
        option_strings=("--vocab-extra-ids",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "vocab_file": MegatronActionSpec(
        option_strings=("--vocab-file",), action_type="store", nargs=None, const=None, default=None
    ),
    "vocab_size": MegatronActionSpec(
        option_strings=("--vocab-size",), action_type="store", nargs=None, const=None, default=None
    ),
    "wandb_exp_name": MegatronActionSpec(
        option_strings=("--wandb-exp-name",),
        action_type="store",
        nargs=None,
        const=None,
        default="",
    ),
    "wandb_project": MegatronActionSpec(
        option_strings=("--wandb-project",), action_type="store", nargs=None, const=None, default=""
    ),
    "wandb_save_dir": MegatronActionSpec(
        option_strings=("--wandb-save-dir",),
        action_type="store",
        nargs=None,
        const=None,
        default="",
    ),
    "warmup": MegatronActionSpec(
        option_strings=("--warmup",), action_type="store", nargs=None, const=None, default=None
    ),
    "weight_decay": MegatronActionSpec(
        option_strings=("--weight-decay",),
        action_type="store",
        nargs=None,
        const=None,
        default=0.01,
    ),
    "weight_decay_incr_style": MegatronActionSpec(
        option_strings=("--weight-decay-incr-style",),
        action_type="store",
        nargs=None,
        const=None,
        default="constant",
    ),
    "wgrad_deferral_limit": MegatronActionSpec(
        option_strings=("--wgrad-deferral-limit",),
        action_type="store",
        nargs=None,
        const=None,
        default=0,
    ),
    "yaml_cfg": MegatronActionSpec(
        option_strings=("--yaml-cfg",), action_type="store", nargs=None, const=None, default=None
    ),
}

__all__ = ["MEGATRON_ARG_METADATA", "MEGATRON_ACTION_SPECS", "MegatronActionSpec"]
