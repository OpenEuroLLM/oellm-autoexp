#!/usr/bin/env python3
"""Wrapper to generate Megatron config with mocked transformer_engine."""

import sys
from unittest.mock import MagicMock

# Mock all transformer_engine submodules before any imports
te_mock = MagicMock()
sys.modules['transformer_engine'] = te_mock
sys.modules['transformer_engine.pytorch'] = MagicMock()
sys.modules['transformer_engine.pytorch.distributed'] = MagicMock()
sys.modules['transformer_engine.pytorch.tensor'] = MagicMock()

# Now import and run the actual script
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'submodules' / 'Megatron-LM'))

from scripts.generate_megatron_config import main

if __name__ == '__main__':
    main()
