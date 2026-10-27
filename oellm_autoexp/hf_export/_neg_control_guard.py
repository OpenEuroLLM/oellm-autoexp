"""Negative control for flash-linear-attention/tests/layers/test_off_current_device.py: the same
tests with _SignedGate's input_guard removed; they must FAIL (else the test does not see the bug)."""

import sys

import pytest

from fla.ops.kda import signed_gate

sg = signed_gate._SignedGate
sg.forward = staticmethod(sg.forward.__wrapped__)
sg.backward = staticmethod(sg.backward.__wrapped__)
sys.exit(pytest.main(["-q", "flash-linear-attention/tests/layers/test_off_current_device.py", "-k", "complex_kda"]))
