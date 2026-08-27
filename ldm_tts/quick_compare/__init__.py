"""Reusable fixed-budget LDM, BO, and direct-LLM comparison workflow."""

from ldm_tts.quick_compare.config import load_quick_compare_spec
from ldm_tts.quick_compare.execution import run_comparison

__all__ = ["load_quick_compare_spec", "run_comparison"]
