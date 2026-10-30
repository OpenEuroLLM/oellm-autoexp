"""Moved into the package: oellm_autoexp/node_health/bench.py (+ verdict.py), the preflight node
check of the pools (config/backend/node_health.yaml, experiments/korbi/pool/node_health_mn5).
Kept so the old invocation still works::

    torchrun ... scripts/korbi/pool/fsdp_comm_bench.py   ==   torchrun ... -m oellm_autoexp.node_health.bench
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from oellm_autoexp.node_health.bench import main  # noqa: E402

if __name__ == "__main__":
    main()
