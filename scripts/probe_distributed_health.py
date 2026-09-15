#!/usr/bin/env python3
"""GPU and RCCL collective health probe for one process per allocated GPU."""

import argparse
import os
import socket
import sys
from datetime import timedelta

# Make collective failures surface instead of waiting indefinitely where supported.
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

import torch
import torch.distributed as dist


def _required_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is not set")
    return int(value)


def _check_close(actual: torch.Tensor, expected: torch.Tensor, operation: str) -> None:
    if not torch.allclose(actual, expected):
        raise RuntimeError(
            f"{operation} returned unexpected values: "
            f"actual={actual.flatten()[:8].cpu().tolist()}, "
            f"expected={expected.flatten()[:8].cpu().tolist()}"
        )


def _announce(rank: int, operation: str) -> None:
    if rank == 0:
        print(f"HEALTH PROBE: testing {operation}", flush=True)


def _test_collectives(
    rank: int,
    world_size: int,
    device: torch.device,
    total_elements: int,
    large_reduce_scatter_mib: int,
) -> None:
    rank_sum = world_size * (world_size + 1) / 2
    total_elements = max(world_size, (total_elements // world_size) * world_size)
    chunk_elements = total_elements // world_size

    _announce(rank, "barrier")
    dist.barrier()

    _announce(rank, "broadcast")
    value = torch.full((total_elements,), float(rank + 1), device=device)
    dist.broadcast(value, src=0)
    _check_close(value, torch.ones_like(value), "broadcast")

    _announce(rank, "reduce")
    value.fill_(float(rank + 1))
    dist.reduce(value, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        _check_close(value, torch.full_like(value, rank_sum), "reduce")

    _announce(rank, "all-reduce")
    value.fill_(float(rank + 1))
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    _check_close(value, torch.full_like(value, rank_sum), "all-reduce")

    _announce(rank, "all-gather")
    gather_input = torch.full((chunk_elements,), float(rank), device=device)
    gather_output = torch.empty(total_elements, device=device)
    dist.all_gather_into_tensor(gather_output, gather_input)
    gather_expected = torch.arange(
        world_size, dtype=gather_output.dtype, device=device
    ).repeat_interleave(chunk_elements)
    _check_close(gather_output, gather_expected, "all-gather")

    _announce(rank, "reduce-scatter")
    scatter_input = torch.full((total_elements,), float(rank + 1), device=device)
    scatter_output = torch.empty(chunk_elements, device=device)
    dist.reduce_scatter_tensor(scatter_output, scatter_input, op=dist.ReduceOp.SUM)
    _check_close(
        scatter_output,
        torch.full_like(scatter_output, rank_sum),
        "reduce-scatter",
    )

    _announce(rank, "coalesced reduce-scatter")
    from torch.distributed.distributed_c10d import _coalescing_manager

    coalesced_outputs = [
        torch.empty(chunk_elements, device=device),
        torch.empty(chunk_elements, device=device),
    ]
    coalesced_inputs = [
        torch.full((total_elements,), float(rank + 1), device=device),
        torch.full((total_elements,), float(2 * (rank + 1)), device=device),
    ]
    with _coalescing_manager(dist.group.WORLD, async_ops=False):
        for output, input_tensor in zip(coalesced_outputs, coalesced_inputs):
            dist.reduce_scatter_tensor(
                output, input_tensor, op=dist.ReduceOp.SUM
            )
    _check_close(
        coalesced_outputs[0],
        torch.full_like(coalesced_outputs[0], rank_sum),
        "coalesced reduce-scatter[0]",
    )
    _check_close(
        coalesced_outputs[1],
        torch.full_like(coalesced_outputs[1], 2 * rank_sum),
        "coalesced reduce-scatter[1]",
    )
    del coalesced_inputs, coalesced_outputs

    if large_reduce_scatter_mib > 0:
        element_size = torch.tensor([], dtype=torch.float32).element_size()
        large_elements = large_reduce_scatter_mib * 1024 * 1024 // element_size
        large_elements = (large_elements // world_size) * world_size
        if large_elements == 0:
            raise RuntimeError("large reduce-scatter payload is smaller than WORLD_SIZE")
        large_chunk_elements = large_elements // world_size

        _announce(
            rank,
            f"{large_reduce_scatter_mib} MiB coalesced all-reduce",
        )
        large_all_reduce = torch.full(
            (large_elements,), float(rank + 1), device=device
        )
        with _coalescing_manager(dist.group.WORLD, async_ops=False):
            dist.all_reduce(large_all_reduce, op=dist.ReduceOp.SUM)
        _check_close(
            large_all_reduce,
            torch.full_like(large_all_reduce, rank_sum),
            "large coalesced all-reduce",
        )
        del large_all_reduce

        _announce(
            rank,
            f"{large_reduce_scatter_mib} MiB coalesced reduce-scatter",
        )
        large_input = torch.full(
            (large_elements,), float(rank + 1), device=device
        )
        large_output = torch.empty(large_chunk_elements, device=device)
        with _coalescing_manager(dist.group.WORLD, async_ops=False):
            dist.reduce_scatter_tensor(
                large_output, large_input, op=dist.ReduceOp.SUM
            )
        _check_close(
            large_output,
            torch.full_like(large_output, rank_sum),
            "large coalesced reduce-scatter",
        )
        del large_input, large_output

    _announce(rank, "all-to-all")
    all_to_all_input = (
        rank * world_size
        + torch.arange(world_size, dtype=torch.float32, device=device)
    ).repeat_interleave(chunk_elements)
    all_to_all_output = torch.empty_like(all_to_all_input)
    dist.all_to_all_single(all_to_all_output, all_to_all_input)
    all_to_all_expected = (
        torch.arange(world_size, dtype=torch.float32, device=device) * world_size
        + rank
    ).repeat_interleave(chunk_elements)
    _check_close(all_to_all_output, all_to_all_expected, "all-to-all")

    _announce(rank, "point-to-point ring")
    previous_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size
    send = torch.tensor([float(rank)], device=device)
    receive = torch.empty_like(send)
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send, next_rank),
            dist.P2POp(dist.irecv, receive, previous_rank),
        ]
    )
    for request in requests:
        request.wait()
    _check_close(
        receive,
        torch.tensor([float(previous_rank)], device=device),
        "point-to-point ring",
    )

    dist.barrier()
    torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Process-group operation timeout (default: 120)",
    )
    parser.add_argument(
        "--master-port-offset",
        type=int,
        default=1,
        help="Offset from MASTER_PORT to avoid colliding with training (default: 1)",
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=262144,
        help="Approximate FP32 elements per collective (default: 262144, or 1 MiB)",
    )
    parser.add_argument(
        "--large-reduce-scatter-mib",
        type=int,
        default=4096,
        help="Large coalesced all-reduce/reduce-scatter input per rank in MiB (default: 4096)",
    )
    args = parser.parse_args()

    rank = _required_int("RANK")
    world_size = _required_int("WORLD_SIZE")
    local_rank = _required_int("LOCAL_RANK")
    hostname = socket.gethostname()
    training_master_port = _required_int("MASTER_PORT")
    os.environ["MASTER_PORT"] = str(training_master_port + args.master_port_offset)

    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("PyTorch sees no ROCm devices")
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but PyTorch sees only {device_count} devices"
            )

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # Exercise allocation and a small kernel before involving other ranks.
        local = torch.tensor([float(device_count)], device=device)
        local.add_(1).sub_(1)
        torch.cuda.synchronize(device)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=args.timeout_seconds),
        )

        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        expected_device_sum = world_size * device_count
        if local.item() != float(expected_device_sum):
            raise RuntimeError(
                "inconsistent GPU visibility across ranks: "
                f"observed device-count sum={local.item()}, "
                f"expected={expected_device_sum}"
            )

        _test_collectives(
            rank,
            world_size,
            device,
            args.elements,
            args.large_reduce_scatter_mib,
        )

        if rank == 0:
            print(
                f"HEALTH PROBE PASSED: {world_size} ranks, "
                f"{device_count} visible GPUs per task, all RCCL tests successful",
                flush=True,
            )
        return 0
    except Exception as error:
        print(
            f"HEALTH PROBE FAILED on host={hostname} rank={rank} "
            f"local_rank={local_rank}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ["MASTER_PORT"] = str(training_master_port)


if __name__ == "__main__":
    raise SystemExit(main())
