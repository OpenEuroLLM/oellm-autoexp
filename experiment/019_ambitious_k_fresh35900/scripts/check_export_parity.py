#!/usr/bin/env python3
"""Fail closed on actual-export HF/vLLM parity, including chunked prefill."""
import json
import os
import sys
from pathlib import Path

def main():
    from oellm_autoexp.hf_export.vllm_parity import check
    from scripts.evals.spec import ModelCode

    # Same arithmetic/load paths as the recall workers; eager execution for the
    # explicit prefill/chunked-prefill comparisons in the parity harness.
    os.environ.update(ModelCode(enforce_eager=True, vllm_cudagraph="none").env())
    export = Path(sys.argv[1])
    report = check(export, tp=4, hf_device_map="auto")
    (export / "hf_vllm_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise SystemExit("HF/vLLM parity failed; recall evaluations remain gated")


if __name__ == "__main__":
    main()
