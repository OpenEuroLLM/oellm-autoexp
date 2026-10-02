"""CPU reference checks for the isolated TWEO candidate; no Megatron/GPU imports.

Run in the pinned training image, with --source pointing at the isolated clone.
These establish local algebra, not real Megatron TP/PP gradient correctness.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint


def config(**overrides):
    values = dict(
        tweo_loss_coeff=0.01, tweo_tau=3.0, tweo_start_step=100,
        tweo_warmup_steps=50, _tweo_current_coeff=0.01,
        num_layers=3, tensor_model_parallel_size=1, sequence_parallel=False,
        context_parallel_size=1, calculate_per_token_loss=False,
        num_moe_experts=None, mtp_num_layers=None, cuda_graph_impl='none',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class AlgebraChecks(unittest.TestCase):
    """Compare injected gradients to an independently differentiated scalar loss."""

    def setUp(self):
        torch.manual_seed(42)
        tweo.TWEOState.reset()

    def test_zero_is_exact_identity(self):
        x = torch.randn(4, 7, requires_grad=True)
        out = tweo.attach(x, config(tweo_loss_coeff=0), 1)
        self.assertIs(out, x)
        out.square().sum().backward()
        torch.testing.assert_close(x.grad, 2*x.detach(), rtol=0, atol=0)
        self.assertEqual(tweo.TWEOState.moments, {})

    def test_training_cli_and_config_overrides(self):
        from megatron.training.arguments import (
            add_megatron_arguments, core_transformer_config_from_args)

        parser = add_megatron_arguments(argparse.ArgumentParser())
        defaults = parser.parse_args([])
        self.assertEqual(defaults.tweo_loss_coeff, 0.)
        self.assertEqual(defaults.tweo_tau, 3.)
        self.assertEqual(defaults.tweo_start_step, 0)
        self.assertEqual(defaults.tweo_warmup_steps, 0)
        args = parser.parse_args([
            '--num-layers', '2', '--hidden-size', '128', '--num-attention-heads', '4',
            '--tweo-loss-coeff', '1e-32', '--tweo-tau', '7',
            '--tweo-start-step', '266000', '--tweo-warmup-steps', '100',
        ])
        args.params_dtype = torch.float32  # normally supplied by validate_args
        cfg = core_transformer_config_from_args(args)
        self.assertEqual(cfg.tweo_loss_coeff, 1e-32)
        self.assertEqual(cfg.tweo_tau, 7.)
        self.assertEqual(cfg.tweo_start_step, 266000)
        self.assertEqual(cfg.tweo_warmup_steps, 100)

    def test_no_grad_is_identity(self):
        x = torch.randn(4, 7)
        with torch.no_grad():
            self.assertIs(tweo.attach(x, config(), 1), x)

    def test_absolute_ramp_resume(self):
        expected = {99: 0., 100: 0., 125: .005, 150: .01, 200: .01}
        for step, value in expected.items():
            self.assertEqual(tweo.coefficient(.01, step, 100, 50), value)
        self.assertEqual(tweo.coefficient(.01, 100, 100, 0), .01)
        for target in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                tweo.coefficient(target, 100, 100, 50)

    def test_missing_coefficient_and_scale_fail(self):
        x = torch.ones(2, requires_grad=True)
        c = config()
        del c._tweo_current_coeff
        with self.assertRaises(RuntimeError):
            tweo.attach(x, c, 1)
        with self.assertRaises(RuntimeError):
            tweo.attach(x, config(), 1).sum().backward()

    def test_unsupported_modes_fail(self):
        for override in (
            dict(context_parallel_size=2), dict(calculate_per_token_loss=True),
            dict(num_moe_experts=2), dict(mtp_num_layers=1),
            dict(cuda_graph_impl='local'), dict(tweo_tau=0),
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                tweo.validate(config(**override))
        tweo.validate(config())

    def test_scalar_reference_multiple_layers_and_microbatches(self):
        for batches in (1, 4):
            for scale in (1., 128.):
                with self.subTest(batches=batches, scale=scale):
                    weights = [torch.randn(5, 5, dtype=torch.double)*.1 for _ in range(3)]
                    inputs = torch.randn(batches, 7, 5, dtype=torch.double)
                    results = []
                    for injected in (False, True):
                        params = [w.clone().requires_grad_() for w in weights]
                        tweo.TWEOState.reset()
                        tweo.TWEOState.scale = scale / batches
                        for x in inputs:
                            raw = 0
                            for layer, w in enumerate(params, 1):
                                x = x + torch.tanh(x @ w)
                                if injected:
                                    x = tweo.attach(x, config(), layer)
                                else:
                                    raw = raw + (x/(3.+1e-6)).pow(4).mean()/3
                            loss = x.square().mean()
                            if not injected:
                                loss = loss + .01*raw
                            (loss*scale/batches).backward()
                        results.append([p.grad.clone() for p in params])
                    for ref, actual in zip(*results):
                        torch.testing.assert_close(ref, actual, rtol=2e-12, atol=2e-12)

    def test_partitioned_sequence_matches_full_mean(self):
        # Algebra only: real TP synchronization is a separate GPU gate.
        x = torch.randn(12, 5, dtype=torch.double)
        full = x.clone().requires_grad_()
        (.01*(full/(3.+1e-6)).pow(4).mean()/3).backward()
        parts = []
        for chunk in x.chunk(4):
            shard = chunk.clone().requires_grad_()
            tweo.TWEOState.scale = 1.
            out = tweo.attach(shard, config(sequence_parallel=True, tensor_model_parallel_size=4), 1)
            out.backward(torch.zeros_like(out))
            parts.append(shard.grad)
        torch.testing.assert_close(torch.cat(parts), full.grad, rtol=2e-12, atol=2e-12)

    def test_finite_difference(self):
        x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        tweo.TWEOState.scale = 1.
        tweo.attach(x, config(), 1).square().mean().backward()
        eps = 1e-5
        def objective(v):
            return v.square().mean()+.01*(v/(3.+1e-6)).pow(4).mean()/3
        for index in ((0, 0), (2, 3)):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus[index] += eps
            minus[index] -= eps
            numeric = (objective(plus)-objective(minus))/(2*eps)
            torch.testing.assert_close(numeric, x.grad[index], rtol=1e-8, atol=1e-10)

    def test_reentrant_recomputation(self):
        x = torch.randn(3, 5, dtype=torch.double)
        grads = []
        for recompute in (False, True):
            tweo.TWEOState.reset()
            tweo.TWEOState.scale = 1.
            inp = x.clone().requires_grad_()
            def block(v):
                return tweo.attach(v + v.sin(), config(), 1)
            out = checkpoint(block, inp, use_reentrant=True) if recompute else block(inp)
            out.square().mean().backward()
            grads.append(inp.grad)
            self.assertEqual(tweo.TWEOState.moments[1][1].item(), x.numel())
        torch.testing.assert_close(*grads, rtol=0, atol=0)

    def test_bf16_outlier_derivative_uses_fp32(self):
        x = torch.tensor([-10000., 10000., 0.], dtype=torch.bfloat16, requires_grad=True)
        tweo.TWEOState.scale = 1.
        tweo.attach(x, config(), 1).backward(torch.zeros_like(x))
        expected = (4*.01*x.detach().float().pow(3)/(3*3*(3.+1e-6)**4)).bfloat16()
        self.assertTrue(torch.isfinite(x.grad).all())
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)

    def test_observed_266k_scale_and_tiny_coefficient(self):
        # The measured residual maximum exceeds 5e9. GPUs flush FP32
        # subnormal intermediates, so folding lambda/N into one tiny constant
        # can silently zero a perfectly representable final derivative.
        torch.set_flush_denormal(True)
        try:
            x = torch.full((1024,), 5e9, dtype=torch.bfloat16, requires_grad=True)
            cfg = config(tweo_loss_coeff=1e-32, _tweo_current_coeff=1e-32,
                         num_layers=64, sequence_parallel=True, tensor_model_parallel_size=4)
            tweo.TWEOState.scale = 1.
            tweo.attach(x, cfg, 64).backward(torch.zeros_like(x))
            expected = (4*1e-32*x.detach().double().pow(3)/(64*1024*4*(3.+1e-6)**4)).bfloat16()
            self.assertTrue(torch.isfinite(tweo.TWEOState.moments[64]).all())
            self.assertTrue((x.grad != 0).all())
            torch.testing.assert_close(x.grad, expected, rtol=.008, atol=0)
        finally:
            torch.set_flush_denormal(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.source/'megatron/core/transformer/tweo.py'
    sys.path.insert(0, str(args.source.resolve()))
    spec = importlib.util.spec_from_file_location('tweo_candidate', path)
    global tweo
    tweo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tweo)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AlgebraChecks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        passed=result.wasSuccessful(), tests=result.testsRun,
        source=str(args.source), module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__, scope='CPU algebra only; GPU TP/PP integration remains unvalidated',
    ), indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
