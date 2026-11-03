"""Named RPCs for same-loaded-model fusion and CUDA graph validation."""
from scripts.korbi.vllm.batch_memory import BatchMemoryProbe


class FusionProbe(BatchMemoryProbe):
    def configure_fusion(self, fused=False, graphs=False):
        import os
        os.environ['OELLM_DECODE_GRAPHS']='1' if graphs else '0'
        count=0
        for module in self.model_runner.model.modules():
            if type(module).__name__ in ['OellmMHC','OellmRMSNorm']:
                module.fused=fused
                count+=1
        return {'modules':count,'fused':fused,'graphs':graphs}

    def graph_stats(self):
        from collections import defaultdict
        from vllm.model_executor.layers.oellm_graph import DecodeGraph
        stats=defaultdict(lambda:{'captures':0,'replays':0,'fallbacks':0})
        for module in self.model_runner.model.modules():
            for name,value in vars(module).items():
                if isinstance(value,DecodeGraph):
                    for k in stats[name]:
                        stats[name][k]+=getattr(value,k)
        return dict(stats)

    def check_ckda_graphs(self):
        import importlib
        check=importlib.import_module('experiment.010_ambitious_fusion_graphs.scripts.check_ckda')
        return check.check(self.model_runner)
