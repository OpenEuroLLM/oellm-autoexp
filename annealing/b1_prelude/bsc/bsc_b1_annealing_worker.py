"""Run inside the pinned BSC training container under torchrun."""
import json
import os
from pathlib import Path
import runpy
import sys


def check_device():
    import torch

    expected = int(os.environ["B1_GPUS_PER_NODE"])
    if torch.cuda.device_count() != expected or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected {expected} visible H100s, got {torch.cuda.device_count()}")
    marker = Path("/tmp/write-probe")
    if marker.read_text() != "write-probe\n":
        raise RuntimeError("Host-backed /tmp probe missing")
    Path(f"/tmp/rank-{os.environ['LOCAL_RANK']}").touch()


def probe():
    import torch
    import torch.distributed as dist
    import transformer_engine
    from transformer_engine import pytorch as te
    # Import the exact native training module needed by both stages.
    from megatron.training import training
    assert callable(training.setup_model_and_optimizer)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        x = torch.ones(32, 32, device="cuda", dtype=torch.bfloat16)
        layer = te.Linear(32, 32, params_dtype=torch.bfloat16).cuda()
        with te.fp8_autocast(enabled=True):
            y = layer(x)
        y.float().mean().backward()
        total = torch.ones(1, device="cuda")
        dist.all_reduce(total)
        if total.item() != dist.get_world_size() or not torch.isfinite(layer.weight.grad).all().item():
            raise RuntimeError("FP8 or NCCL runtime probe failed")
        if dist.get_rank() == 0:
            print(json.dumps({"BSC_B1_RUNTIME": "passed", "world": dist.get_world_size(),
                              "torch": torch.__version__, "cuda": torch.version.cuda,
                              "transformer_engine": transformer_engine.__version__,
                              "device": torch.cuda.get_device_name(local_rank)}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    mode, root, source = sys.argv[1:]
    check_device()
    if mode == "probe":
        probe()
        return
    if mode not in ("cache", "smoke"):
        raise ValueError(f"Unexpected worker mode: {mode}")
    repo = Path(source)
    entry = (repo / "submodules/Megatron-LM/pretrain_gpt.py" if mode == "cache"
             else repo / "annealing/b1_prelude/training.py")
    sys.argv = [str(entry), *json.loads((Path(root) / f"{mode}-argv.json").read_text())]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
