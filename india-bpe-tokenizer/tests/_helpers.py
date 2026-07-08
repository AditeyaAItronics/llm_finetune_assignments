"""Shared paths + skip helper for the test suite."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
TOK = DIST / "tokenizer" / "tokenizer.json"
MANIFEST = DIST / "tokenizer" / "manifest.json"
CORPUS = ROOT / "corpus"


def require_build():
    """Skip a test cleanly if ./build.sh hasn't produced artifacts yet."""
    if not TOK.exists():
        pytest.skip("build artifacts not found — run ./build.sh first")
