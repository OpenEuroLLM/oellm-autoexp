"""Direct adjoint/moment checks and same-GPU component timings before GPT oracle."""
import argparse
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from megatron.core.transformer import tweo


def config(backend):
    return SimpleNamespace(tweo_loss_coeff=1.,_tweo_current_coeff=1.,tweo_tau=3.,
        num_layers=64,sequence_parallel=True,tensor_model_parallel_size=4,
        tweo_implementation=backend)


def evaluate(x,g,coeff,backend,collect=True):
    c=config(backend);c._tweo_current_coeff=coeff;c.tweo_loss_coeff=coeff
    tweo.TWEOState.reset(collect=collect);tweo.TWEOState.scale=torch.tensor(.5,device=x.device)
    v=x.detach().requires_grad_(True)
    output=tweo.attach(v,c,64)
    assert torch.equal(output,x)
    output.backward(g)
    return v.grad,tweo.TWEOState.moments.get(64)


def correctness():
    torch.manual_seed(1234)
    rows=[]
    for dtype in (torch.float32,torch.bfloat16,torch.float64):
        for amplitude,coeff in ((1.,.01),(1e4,1e-14),(5e9,1.3348774420985196e-26),(5e9,1e-45)):
            for noncontiguous in (False,True):
                x=(torch.randn(129,257,device='cuda')*amplitude).to(dtype)
                if noncontiguous:x=x.T
                g=torch.zeros_like(x)
                ref,m0=evaluate(x,g,coeff,'reference')
                fast,m1=evaluate(x,g,coeff,'fused')
                sparse,m2=evaluate(x,g,coeff,'fused',False)
                assert m2 is None
                torch.testing.assert_close(sparse,fast,rtol=0,atol=0)
                torch.testing.assert_close(m1,m0,rtol=2e-12,atol=1e-30)
                oracle=(4*coeff*(x.double()/(3+1e-6)).pow(3)/(64*x.numel()*4)/(3+1e-6)*.5)
                tol=.008 if dtype==torch.bfloat16 else 2e-6 if dtype==torch.float32 else 1e-12
                torch.testing.assert_close(fast.double(),oracle,rtol=tol,atol=1e-38)
                torch.testing.assert_close(fast,ref,rtol=tol,atol=1e-38)
                rows.append(dict(dtype=str(dtype),amplitude=amplitude,coeff=coeff,
                    noncontiguous=noncontiguous,max_error=float((fast.double()-oracle).abs().max()),passed=True))
    # Cancellation test against the same FP64 rounding envelope used previously.
    x=(torch.randn(200003,device='cuda')*1e8).bfloat16()
    extra=4*1e-26*(x.double()/(3+1e-6)).pow(3)/(64*x.numel()*4)/(3+1e-6)*.5
    g=(-extra*.9).bfloat16()
    result,_=evaluate(x,g,1e-26,'fused',False)
    expected=g.double()+extra
    bound=.004*expected.abs()+4*torch.finfo(torch.float32).eps*(g.double().abs()+extra.abs())
    assert ((result.double()-expected).abs()<=bound+1e-38).all()
    return rows


def timing():
    x=(torch.randn(1024,2,5120,device='cuda')*1e6).bfloat16().requires_grad_(True)
    g=torch.randn_like(x)*1e-4
    tweo.TWEOState.scale=torch.tensor(1/16,device='cuda')
    rows=[]
    for mode,collect in [('reference',True),('reference',False),('fused',True),('fused',False)]:
        cfg=config(mode);cfg._tweo_current_coeff=1.3348774420985196e-26
        def step():
            x.grad=None
            tweo.TWEOState.reset(collect);tweo.TWEOState.scale=scale
            tweo.attach(x,cfg,64).backward(g)
        scale=torch.tensor(1/16,device='cuda')
        for _ in range(5):step()
        times=[]
        for _ in range(5):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();begin.record()
            for i in range(10):step()
            end.record();end.synchronize();times.append(begin.elapsed_time(end)/10)
        rows.append(dict(backend=mode,full_diagnostics=collect,milliseconds=statistics.median(times),replicates=times))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    rank=int(os.environ['RANK']);a.output.mkdir(parents=True,exist_ok=True)
    result=dict(correctness=correctness(),component_timing=timing())
    (a.output/f'components-{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TWEO_FUSED_COMPONENTS_PASS',rank,result['component_timing'],flush=True)
    # Existing BF16/FP8 TP/PP/SP/recompute scalar-loss oracle; audit fast adjoint too.
    import b1_tweo_precision_checks as oracle
    ref=tweo._TWEOIdentity.backward
    original_fast=tweo._TWEOFusedIdentity.backward
    # The inherited audit wraps the reference class. Wrap the fused class with
    # the same independent FP64 bound, without changing fallback dispatch.
    def audited(ctx,incoming):
        answer=original_fast(ctx,incoming)
        value=ctx.saved_tensors[0].double()
        extra=4*ctx.coeff*(value/ctx.tau).pow(3)/ctx.normalizer/ctx.tau*tweo.TWEOState.scale
        expected=incoming.double()+extra
        bound=.004*expected.abs()+4*torch.finfo(torch.float32).eps*(incoming.double().abs()+extra.abs())
        if not ((answer[0].double()-expected).abs()<=bound+1e-38).all():
            raise RuntimeError('Fused adjoint failed independent FP64 bound')
        fused_checks.append(float(((answer[0].double()-expected).abs()/(bound+1e-38)).max()))
        return answer
    fused_checks=[];tweo._TWEOFusedIdentity.backward=staticmethod(audited)
    # The old helper expects nonempty reference-adjoint audit records; add
    # fused records to that same log without replacing the reference fallback.
    old_audit=oracle.audit_injected_adjoints
    def all_audits():
        old_audit()
        return fused_checks
    oracle.audit_injected_adjoints=all_audits
    oracle.main()
    result['distributed_result']=json.loads((a.output/f'rank-{rank}.json').read_text())
    result['fused_adjoint_checks']=len(fused_checks);result['passed']=True
    (a.output/f'rank-{rank}.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
