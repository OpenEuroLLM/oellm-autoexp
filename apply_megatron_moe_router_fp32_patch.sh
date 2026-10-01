#!/bin/bash
# moe_router_dtype: "fp32" fails on Lumi because of a hipBLASLt GEMM failure Unable to find any suitable algorithms, use torch.mm instead.

FILE="./submodules/Megatron-LM/megatron/core/transformer/moe/moe_utils.py"

sed -i \
    -e 's/te_general_gemm is not None and router_dtype != torch.float64:/te_general_gemm is not None and router_dtype not in (torch.float32, torch.float64):/' \
    "$FILE"
