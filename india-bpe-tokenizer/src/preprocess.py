"""Shared text preprocessing + word counting.

CRITICAL: this exact module is imported by training, scoring, AND is mirrored
character-for-character in the browser widget (widget/preprocess.js). If the
numerator (tokens) and denominator (words) are ever computed on differently
cleaned text, the fertility numbers become meaningless. So: one function, used
everywhere, no per-caller variation.
"""

from __future__ import annotations

import re
import unicodedata

# runs of horizontal whitespace (spaces/tabs) but NOT newlines
_WS = re.compile(r"[^\S\n]+")
# reference markers left by the extractor, e.g. [1], [ 23 ]
_REFMARK = re.compile(r"\[\s*\d+\s*\]")
# collapse 3+ blank lines to a single blank line
_MULTINL = re.compile(r"\n{3,}")
# a "word" = maximal run of non-whitespace (script-agnostic; en/hi/te/bn all
# separate words with spaces)
_WORD = re.compile(r"\S+")


def preprocess(text: str) -> str:
    """Normalize text deterministically. Idempotent: preprocess(preprocess(x)) == preprocess(x)."""
    text = unicodedata.normalize("NFC", text)          # stabilize Indic matra/ZWJ ordering
    text = _REFMARK.sub("", text)                        # drop [1]-style markers
    text = _WS.sub(" ", text)                            # collapse horizontal whitespace
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTINL.sub("\n\n", text)                   # tidy blank lines
    return text.strip()


def word_count(text: str) -> int:
    """Number of whitespace-delimited words. This is the denominator of X_i."""
    return len(_WORD.findall(text))


def normalize_spaces(text: str) -> str:
    """Collapse every run of whitespace (incl. newlines) to a single space.

    Fertility is counted on THIS form so each word becomes exactly one
    ``▁word`` pre-token under the Metaspace pre-tokenizer — identical in Python
    and in the browser JS encoder, with no newline / byte-fallback edge cases.
    """
    return " ".join(_WORD.findall(text))
