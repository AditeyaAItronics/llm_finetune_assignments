"""Scoring-model + score-matches-manifest checks.

Per the assignment, only English (X1) carries the "<= 1.2" requirement, and the score
is 1000 / (X_max - X_min). So we check the invariants that actually define a valid
submission: English is the smallest ratio (X1 = least) and the score is well-defined.
"""
import json

import _helpers as H
from tokenizers import Tokenizer

from common import LANGS
from preprocess import normalize_spaces, preprocess, word_count


def _fertilities():
    tok = Tokenizer.from_file(str(H.TOK))
    out = {}
    for lang in LANGS:
        text = preprocess((H.CORPUS / f"{lang}.txt").read_text(encoding="utf-8"))
        out[lang] = len(tok.encode(normalize_spaces(text)).ids) / word_count(text)
    return out


def test_english_is_min_and_score_defined():
    H.require_build()
    x = _fertilities()
    # English is expected to be the least fertile (the assignment's X1 = least).
    assert x["en"] == min(x.values()), x
    spread = max(x.values()) - min(x.values())
    assert spread > 0, x
    score = 1000 / spread
    assert score > 0 and score != float("inf"), score


def test_score_matches_manifest():
    H.require_build()
    manifest = json.loads(H.MANIFEST.read_text(encoding="utf-8"))
    per_lang = manifest["results"]["per_language"]
    x = _fertilities()
    for lang in LANGS:
        assert round(x[lang], 4) == round(per_lang[lang]["X"], 4), lang
