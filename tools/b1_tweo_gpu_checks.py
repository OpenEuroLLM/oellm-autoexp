"""Small real-Megatron TWEO gradient test: explicit scalar reference vs injection.

Four GPUs, TP2, first PP1 then PP2. No training data/checkpoints/optimizer steps.
The PP1 scalar-loss reference is partitioned into PP2 by copying identical weights.
Tests normal and sequence-parallel layouts with one and four microbatches.
"""

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import \
    get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import \
    model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.tweo import TWEOState, attach


def observed_scale_checks():
    """Exercise the measured 266k activation range with normal/tiny coefficients."""
    for coeff in (1e-32, 1e-45):
        TWEOState.reset()
        TWEOState.scale = 1.
        x = torch.full((1024,), 5e9, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        cfg = SimpleNamespace(tweo_loss_coeff=coeff, _tweo_current_coeff=coeff, tweo_tau=3.,
                              num_layers=64, sequence_parallel=True, tensor_model_parallel_size=4)
        attach(x, cfg, 64).backward(torch.zeros_like(x))
        expected = (4*coeff*x.detach().double().pow(3)/(64*1024*4*(3.+1e-6)**4)).bfloat16()
        assert torch.isfinite(TWEOState.moments[64]).all() and (x.grad != 0).all()
        torch.testing.assert_close(x.grad, expected, rtol=.008, atol=0)


def global_name(name, pp):
    if name.startswith('decoder.layers.'):
        fields = name.split('.')
        fields[2] = str(int(fields[2])+parallel_state.get_pipeline_model_parallel_rank()*(2//pp))
        return '.'.join(fields)
    return name


def run_case(pp, sp, batches, mode, weights=None, precision='fp32', recompute=False,
             coefficient=.01, ce_multiplier=1.):
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    FP8GlobalStateManager.reset()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=pp)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    cfg = TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=4, ffn_hidden_size=256,
        tensor_model_parallel_size=2, pipeline_model_parallel_size=pp,
        sequence_parallel=sp, hidden_dropout=0., attention_dropout=0.,
        params_dtype=torch.float32 if precision == 'fp32' else torch.bfloat16,
        pipeline_dtype=torch.float32 if precision == 'fp32' else torch.bfloat16,
        bf16=precision != 'fp32', fp8='hybrid' if precision == 'fp8' else None,
        fp8_amax_history_len=1024, fp8_amax_compute_algo='max',
        recompute_granularity='selective' if recompute else None,
        recompute_modules=['core_attn'],
        use_cpu_initialization=True, embedding_init_method_std=1.0, gradient_accumulation_fusion=False,
        normalization='RMSNorm', add_bias_linear=False, qk_layernorm=True,
        attention_backend=AttnBackend.unfused,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        tweo_loss_coeff=coefficient if mode == 'injected' else 0.,
    )
    cfg._tweo_current_coeff = coefficient
    # Non-unit loss scale tests that auxiliary gradients obey schedule scaling.
    cfg.grad_scale_func = lambda value: value * 8.
    model = GPTModel(
        config=cfg, transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
        vocab_size=256, max_sequence_length=32, position_embedding_type='rope',
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
        share_embeddings_and_output_weights=False,
    ).cuda().train()
    if weights is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(weights[global_name(name, pp)].to(param.device))
    initial = {global_name(n, pp): p.detach().cpu().clone() for n, p in model.named_parameters()}
    moments = []
    handles = []
    if mode == 'scalar':
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # SP partitions tokens; divide the local mean by TP for the global objective.
            moments.append((hidden.float()/(3.+1e-6)).pow(4).mean()/2/(2 if sp else 1))
        handles = [layer.register_forward_hook(capture) for layer in model.decoder.layers]

    def forward(data_iterator, model):
        batch = next(data_iterator)
        moments.clear()
        token = (torch.arange(32, device='cuda')[None, :] + batch*7) % 256
        positions = torch.arange(32, device='cuda')[None, :]
        attention = torch.triu(torch.ones(1, 1, 32, 32, device='cuda', dtype=torch.bool), diagonal=1)
        output = model(token, positions, attention, labels=(token+1) % 256)
        # Capture this microbatch's reference graph before later pipeline forwards.
        penalty = sum(moments)
        def loss_func(value):
            ce = value.float().mean()
            return ce_multiplier*ce + (coefficient*penalty if mode == 'scalar' else 0), {'ce': ce.detach()}
        return output, loss_func

    TWEOState.reset()
    get_forward_backward_func()(
        forward_step_func=forward, data_iterator=iter(range(batches)), model=model,
        num_microbatches=batches, seq_length=32, micro_batch_size=1,
        forward_only=False,
    )
    torch.cuda.synchronize()
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError('Missing gradient: '+name)
        gradients[global_name(name, pp)] = param.grad.detach().float().cpu().clone()
    for handle in handles:
        handle.remove()
    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return initial, gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group('nccl')
    rank = torch.distributed.get_rank()
    observed_scale_checks()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for sp in (False, True):
        for batches in (1, 4):
            if rank == 0:
                print(f'TWEO_CHECK_BEGIN sequence_parallel={sp} microbatches={batches}', flush=True)
            weights, ref = run_case(1, sp, batches, 'scalar')
            # DP replicas have identical tokens/weights. TP rank stays rank % 2 under PP2.
            _, baseline = run_case(1, sp, batches, 'disabled', weights)
            signal = sum((ref[n]-baseline[n]).double().square().sum() for n in ref).sqrt().item()
            if signal <= 1e-8:
                raise RuntimeError('TWEO test signal too small to validate')
            for pp in (1, 2):
                _, actual = run_case(pp, sp, batches, 'injected', weights)
                squared_error, squared_ref, squared_signal = 0., 0., 0.
                for name, grad in actual.items():
                    torch.testing.assert_close(grad, ref[name], rtol=3e-4, atol=3e-6, msg=name)
                    squared_error += (grad-ref[name]).double().square().sum().item()
                    squared_ref += ref[name].double().square().sum().item()
                    squared_signal += (ref[name]-baseline[name]).double().square().sum().item()
                if squared_error > .05**2 * squared_signal:
                    raise RuntimeError('Gradient error exceeds 5% of the actual TWEO gradient signal')
                results.append(dict(pp=pp, sequence_parallel=sp, microbatches=batches,
                                    relative_l2=(squared_error/max(squared_ref, 1e-30))**.5,
                                    penalty_gradient_l2=squared_signal**.5,
                                    absolute_error_l2=squared_error**.5, parameters=len(actual)))
                (args.output/f'progress-{rank}.json').write_text(json.dumps(results, indent=2)+'\n')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/f'rank-{rank}.json').write_text(json.dumps(dict(
        passed=True, rank=rank, torch_version=torch.__version__, cases=results,
        observed_scale_checks_passed=True,
        scope='Small TE FP32 GPT, TP2, PP1/2, SP on/off; not FP8/BF16 or optimizer validation',
    ), indent=2)+'\n')
    torch.distributed.barrier()
    if rank == 0:
        print('TWEO_GPU_CHECKS_PASSED', flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
