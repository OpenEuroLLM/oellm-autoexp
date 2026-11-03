"""Actual loaded CKDA weights: multi-step graph replay with changing state slots."""
from types import SimpleNamespace


def check(runner):
    import torch
    from vllm.forward_context import set_forward_context,get_forward_context
    from vllm.model_executor.layers.oellm_graph import DecodeGraph
    from vllm.model_executor.models.oellm_ambitious import OellmAmbitiousLayer
    layer=next(m for m in runner.model.modules() if isinstance(m,OellmAmbitiousLayer) and m.kind=='ckda')
    original_cache=layer.mixer.kv_cache
    original_graph=layer.ckda_graph
    config=runner.vllm_config
    results=[]
    try:
        with torch.inference_mode():
            for batch in [1,8,64]:
                torch.manual_seed(2026+batch)
                pool=batch*2+5
                initial=tuple(torch.randn((pool,*t.shape[1:]),device=t.device,dtype=t.dtype)*.01 for t in original_cache)
                xs=[torch.randn(batch,4,2048,device=initial[0].device,dtype=torch.bfloat16)*.1 for _ in range(5)]
                slots=[torch.randperm(pool,device=initial[0].device)[:batch].to(torch.int32) for _ in xs]
                def run(graph):
                    layer.mixer.kv_cache=tuple(t.clone() for t in initial)
                    layer.ckda_graph=DecodeGraph()
                    outputs=[]
                    for x,idx in zip(xs,slots):
                        meta=SimpleNamespace(num_actual_tokens=batch,num_prefills=0,num_decode_tokens=batch,
                            spec_sequence_masks=None,non_spec_state_indices_tensor=idx)
                        with set_forward_context({layer.mixer.prefix:meta},config,num_tokens=batch,
                            num_tokens_across_dp=torch.full((config.parallel_config.data_parallel_size,),batch,dtype=torch.int32)):
                            get_forward_context().additional_kwargs["oellm_decode_graph"]=graph
                            outputs.append(layer(x,None).clone())
                    return outputs,tuple(t.clone() for t in layer.mixer.kv_cache),layer.ckda_graph
                eager,ref,_=run(False)
                actual,new,graph=run(True)
                for a,b in zip(actual,eager):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                for a,b in zip(new,ref):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                assert graph.captures==1 and graph.replays==4
                results.append({'batch':batch,'outputs_exact':True,'states_exact':True,'captures':graph.captures,'replays':graph.replays})
    finally:
        layer.mixer.kv_cache=original_cache
        layer.ckda_graph=original_graph
    return results
