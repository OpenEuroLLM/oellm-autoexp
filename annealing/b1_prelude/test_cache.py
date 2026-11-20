import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch
import cache

class Cache(unittest.TestCase):
    def target(self,steps=1193):
        return dict(train_samples=(416000+steps)*4096,global_batch_size=4096,
                    phase_transition_iterations='416000,416003',eval_iters=1,eval_global_batch_size=16,
                    eval_interval=10000,start_eval_at_iter=416000,split='998,2,0')

    def test_shape_matches_pinned_megatron_and_normalization(self):
        source=Path(__file__).resolve().parents[2]/'submodules/Megatron-LM/megatron/training/training.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_train_valid_test_num_samples')
        for steps in (1193,119210):
            target=self.target(steps);target['phase_transition_iterations']=f'416000,{416000+steps}'
            namespace={'get_args':lambda:SimpleNamespace(**dict(target,train_iters=416000+steps,iteration=416000,
                phase_transition_iterations=[416000,416000+steps],full_validation=False,skip_train=False))}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
            self.assertEqual(cache.shape(target,416000),namespace[node.name]())
            normalized=cache.normalized(target,416000)
            actual=cache.shape(normalized,0);expected=cache.shape(target,416000)
            self.assertEqual(actual[:2],expected[:2])
            self.assertGreater(actual[1],0)

    def test_zero_validation_is_rejected_before_any_submission(self):
        target=self.target();target['eval_iters']=0
        with self.assertRaisesRegex(ValueError,'positive validation'):cache.normalized(target,416000)

    def test_nonzero_test_split_is_not_silently_missized(self):
        target=self.target();target['split']='998,1,1'
        with self.assertRaisesRegex(ValueError,'test split'):cache.normalized(target,416000)

    def test_submission_preserves_uncertain_intent_and_refuses_source_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'cache-preparation';folder.mkdir()
            script=folder/'job.sbatch';script.write_text('#!/bin/bash\ntrue\n')
            cache.save(folder/'render.json',dict(script=str(script),script_sha256='wrong'))
            with patch.object(cache.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'script changed'):cache.submit(root)
                run.assert_not_called()
            cache.save(folder/'render.json',dict(script=str(script),script_sha256=cache.sha(script)))
            cache.save(folder/'submission-intent.json',dict(comment='uncertain'))
            with self.assertRaisesRegex(ValueError,'intent'):cache.submit(root)

if __name__=='__main__':unittest.main()
