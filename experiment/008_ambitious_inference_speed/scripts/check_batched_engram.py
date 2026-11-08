"""Compare the opt-in batched Engram path with real loaded checkpoint modules."""
import importlib
import time
from types import SimpleNamespace


def check_modules(model):
    import torch
    port = importlib.import_module('vllm.model_executor.models.oellm_ambitious')
    results = []
    original_metadata = port._layer_metadata
    try:
        with torch.inference_mode():
            for name, module in model.named_modules():
                if not isinstance(module, port.OellmEngram):
                    continue
                original_cache, original_mode = module.kv_cache, module.batched_decode
                try:
                    for batch in [1, 8, 32, 64, 128, 256]:
                        for mixed in [False, True]:
                            torch.manual_seed(2026 + batch)
                            device = module.key_proj.weight.device
                            count = batch + (14 if mixed else 0)
                            pool = 2 * (batch + 2) + 5
                            slots = torch.randperm(pool, device=device)[:batch + 2]
                            conv = torch.randn(pool, module.h, module.pad, device=device,
                                               dtype=module.key_proj.weight.dtype) * 0.1
                            vocab = module.model_config.hf_config.vocab_size
                            hist = torch.randint(vocab + 1, (pool, module.hist), device=device)
                            ids = torch.randint(vocab, (count,), device=device)
                            x = torch.randn(count, module.h, device=device,
                                            dtype=module.key_proj.weight.dtype) * 0.2
                            meta = SimpleNamespace(num_decode_tokens=batch,
                                num_prefills=2 if mixed else 0,
                                state_indices_tensor_d=slots[:batch],
                                state_indices_tensor_p=slots[batch:batch + 2],
                                query_start_loc_p=torch.tensor([0, 5, 14], device=device),
                                has_initial_states_p=torch.tensor([True, False], device=device))
                            port._layer_metadata = lambda prefix: meta
                            mapped = module.token_map[ids] if module.use_token_map else ids
                            expected = [module._seq_rows(mapped[i:i+1], hist[slots[i]])[0]
                                        for i in range(batch)]
                            shifted = torch.cat([hist[slots[:batch]], mapped[:batch, None] + 1], dim=1)
                            actual = module._rows_decode(shifted, module.hash_mult,
                                                         module.sizes, module.offsets)
                            assert torch.equal(torch.cat(expected), actual)

                            def run(batched, timed=False):
                                module.batched_decode = batched
                                module.kv_cache = (conv.clone(), hist.clone())
                                torch.cuda.synchronize()
                                begin = time.perf_counter()
                                y = module(x, ids)
                                torch.cuda.synchronize()
                                elapsed = time.perf_counter() - begin
                                if timed:
                                    return elapsed
                                return y.clone(), tuple(t.clone() for t in module.kv_cache)

                            reference, ref_state = run(False)
                            candidate, new_state = run(True)
                            difference = (reference.float() - candidate.float()).abs()
                            torch.testing.assert_close(candidate, reference, rtol=0.01, atol=2e-4)
                            assert torch.equal(ref_state[0], new_state[0])
                            assert torch.equal(ref_state[1], new_state[1])
                            record = {'module': name, 'batch': batch, 'mixed': mixed,
                                'rows_equal': True, 'conv_state_equal': True, 'id_state_equal': True,
                                'output_equal': torch.equal(reference, candidate),
                                'max_abs_output_difference': difference.max().item(),
                                'rms_output_difference': difference.square().mean().sqrt().item()}
                            if not mixed:
                                record['reference_seconds'] = [run(False, True) for _ in range(3)]
                                record['batched_seconds'] = [run(True, True) for _ in range(3)]
                            results.append(record)
                finally:
                    module.kv_cache, module.batched_decode = original_cache, original_mode
    finally:
        port._layer_metadata = original_metadata
    assert results, 'No Engram modules found'
    return results
