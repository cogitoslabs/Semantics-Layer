"""
tests/test_trace_log_ui.py — Unit tests for Trace Log UI helpers and formatting logic
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from lib.utils.trace_logger import format_file_label, parse_eval_num_from_filename


def test_format_file_label():
    path1 = Path("logs/traces/cloze/20260814_114500_eval_0001.csv")
    label1 = format_file_label(path1)
    assert "2026-08-14" in label1
    assert "11:45:00" in label1
    assert "eval_0001" in label1

    path2 = Path("logs/traces/qa/custom_trace.csv")
    label2 = format_file_label(path2)
    assert label2 == "custom_trace.csv"


def test_parse_eval_num_from_filename():
    assert parse_eval_num_from_filename("20260814_114500_eval_0001.csv") == 1
    assert parse_eval_num_from_filename("20260814_114500_eval_0.csv") == 0
    assert parse_eval_num_from_filename("custom.csv") == 999999
