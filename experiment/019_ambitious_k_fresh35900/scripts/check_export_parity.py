#!/usr/bin/env python3
"""Fail closed on actual-export HF/vLLM parity, including chunked prefill."""
import json
import sys
from pathlib import Path

from oellm_autoexp.hf_export.vllm_parity import check

export = Path(sys.argv[1])
report = check(export, tp=4, hf_device_map="auto")
(export / "hf_vllm_parity.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
if not report["pass"]:
    raise SystemExit("HF/vLLM parity failed; recall evaluations remain gated")
