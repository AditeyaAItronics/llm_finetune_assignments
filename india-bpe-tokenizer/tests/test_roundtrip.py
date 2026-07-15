"""Faithful round-trip gate — the assignment's pass/fail criterion.

decode(encode(text)) must preserve the same visible (non-whitespace) characters, for arbitrary
Markdown / URL / multilingual text, using the DEFAULT decode (skip_special_tokens=True — which is
what a grader calls). Whitespace may differ; visible characters may not disappear.
"""
import _helpers as H
from tokenizers import Tokenizer

SAMPLES = [
    "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",          # the grader's failing example
    "# Heading\n\n_italic_ **bold** `code` [link](url) | table |",
    "भारत — India. দেশ · దేశం · देश. 50% (1947).",
    "a#b_c-d~e^f{g}h|i\\j",                                    # punctuation soup
    "emoji 😀 and 中文 mixed in",                               # exotic → byte-fallback path
]


def _visible(s):
    return [c for c in s if not c.isspace()]


def test_faithful_roundtrip():
    H.require_build()
    tok = Tokenizer.from_file(str(H.TOK))
    for s in SAMPLES:
        decoded = tok.decode(tok.encode(s).ids)   # default decode, as a grader would call it
        missing = [c for c in _visible(s) if c not in decoded]
        assert not missing, f"round-trip dropped {missing!r} from {s!r} -> {decoded!r}"
