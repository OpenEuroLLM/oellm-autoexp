"""Numerical rounding and graph state/slot regression checks on a real GPU."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.model_executor.models.oellm_ambitious import OellmMHC, OellmRMSNorm
from vllm.model_executor.layers.oellm_graph import DecodeGraph


@torch.inference_mode()
def main():
    torch.manual_seed(2026)
    records=[]
    config=SimpleNamespace(mhc_expansion_rate=4,hidden_size=2048,mhc_sinkhorn_iters=20,mhc_eps=1e-6)
    model=OellmMHC(config).cuda().to(torch.bfloat16)
    for p in model.parameters():
        p.copy_(torch.randn_like(p)*0.08)
    for batch in [1,7,32,64,256]:
        for scale in [0.,1e-4,1.,50.]:
            x=torch.randn(batch,4,2048,device='cuda',dtype=torch.bfloat16)*scale
            out=torch.randn(batch,2048,device='cuda',dtype=torch.bfloat16)
            model.fused=False
            ref,pending=model.read(x)
            wr=model.write(x,out,pending)
            model.fused=True
            new,pn=model.read(x)
            wn=model.write(x,out,pending)
            torch.testing.assert_close(new,ref,atol=.03125,rtol=.01)
            torch.testing.assert_close(pn[0],pending[0],atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(pn[1],pending[1],atol=2e-5,rtol=2e-5)
            assert torch.equal(wn,wr), 'mHC write rounding changed'
            records.append({'batch':batch,'scale':scale,'read_max_abs':(new.float()-ref.float()).abs().max().item(),'write_exact':True})
    for dtype in [torch.bfloat16,torch.float16,torch.float32]:
        for batch in [1,7,64]:
            for zc in [False,True]:
                norm=OellmRMSNorm(2048,1e-6,zc).cuda().to(dtype)
                norm.weight.copy_(torch.randn_like(norm.weight)*.1)
                x=torch.randn(batch,2048,device='cuda',dtype=dtype)
                norm.fused=False; ref=norm(x)
                norm.fused=True; new=norm(x)
                torch.testing.assert_close(new,ref,atol=2e-3 if dtype!=torch.float32 else 2e-6,rtol=.01 if dtype!=torch.float32 else 2e-6)
    # First capture must update state once, and every replay must refresh slots.
    for batch in [1,8,64]:
        ref=torch.randn(batch*3,16,device='cuda')
        state=ref.clone()
        graph=DecodeGraph()
        def step(x,slots):
            value=state[slots]*.5+x
            state[slots]=value
            return value
        for i in range(5):
            slots=torch.randperm(batch*3,device='cuda')[:batch]
            x=torch.randn(batch,16,device='cuda')
            expected=ref[slots]*.5+x
            ref[slots]=expected
            actual=graph(step,x,slots)
            assert torch.equal(actual,expected)
            assert torch.equal(state,ref), 'capture advanced state or replay used stale indices'
        assert graph.captures==1 and graph.replays==4
    target=Path('experiment/010_ambitious_fusion_graphs/results/kernel_checks.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'passed':True,'mhc':records,'rms_cases':18,'graph_slot_cases':15},indent=2)+'\n')
    print('Fusion rounding and graph state/slot checks passed',flush=True)

if __name__=='__main__':
    main()
