"""Shared constants and paths for the India BPE tokenizer project.

Every other script imports from here so there is a single source of truth for
the language set, the vocab budget, the fertility gate, and where files live.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode Devanagari/Telugu/Bengali.
# Every script imports common, so force UTF-8 stdout/stderr here once, for all of them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"            # pinned, post-preprocessing text + manifest
DIST_DIR = ROOT / "dist"                # deployable bundle
TOKENIZER_DIR = DIST_DIR / "tokenizer"  # tokenizer.json, vocab.json, merges.txt, ...
DIST_CORPUS_DIR = DIST_DIR / "corpus"   # corpus copied into the bundle for the widget
WIDGET_DIR = ROOT / "widget"            # static SPA source

# --- constants -------------------------------------------------------------
# Order matters only for reproducible iteration/printing.
LANGS: list[str] = ["en", "hi", "te", "bn"]

VOCAB_SIZE = 10_000       # total tokenizer vocabulary (byte base + specials + merges)
FERTILITY_GATE = 1.2      # every X_i must be <= this
BYTE_BASE = 256           # byte-level alphabet size (always in the vocab)

# Wikipedia "India" article per language: {lang: (api_host, article_title)}.
# fetch_corpus.py resolves redirects, so slight title variants are fine.
ARTICLES: dict[str, tuple[str, str]] = {
    "en": ("en.wikipedia.org", "India"),
    "hi": ("hi.wikipedia.org", "भारत"),
    "te": ("te.wikipedia.org", "భారత దేశం"),
    "bn": ("bn.wikipedia.org", "ভারত"),
}
