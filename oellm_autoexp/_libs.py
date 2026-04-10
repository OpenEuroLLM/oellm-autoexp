"""Setup library paths for hydra_staged_sweep, slurm_gen, monitor,
argparse_schema.

This module ensures the standalone libraries are importable by adding
their src directories to sys.path. Import this module before importing
from the libraries.
"""

import sys
from pathlib import Path

# Get the root directory (where oellm_autoexp is)
_ROOT = Path(__file__).parent.parent

# Add library source directories to sys.path
_LIBS = [
    _ROOT / "hydra_staged_sweep" / "src",
    _ROOT / "slurm_gen" / "src",
    _ROOT / "monitor" / "src",
    _ROOT / "argparse_schema" / "src",
]

for lib_path in _LIBS:
    if lib_path.exists() and str(lib_path) not in sys.path:
        sys.path.insert(0, str(lib_path))
