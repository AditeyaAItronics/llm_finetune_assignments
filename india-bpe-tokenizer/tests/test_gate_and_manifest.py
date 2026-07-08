"""Fertility gate + score-matches-manifest checks."""
import json

import _helpers as H
from tokenizers import Tokenizer

from common import FERTILITY_GATE, LANGS
from preprocess import preprocess, word_count


def _fertilities():
    tok = Tokenizer.from_file(str(H.TOK))
    out = {}
    for lang in LANGS:
        text = preprocess((H.CORPUS / f"{lang}.txt").read_text(encoding="utf-8"))
        out[lang] = len(tok.encode(text).ids) / word_count(text)
    return out


def test_gate_all_under_1_2():
    H.require_build()
    x = _fertilities()
    assert all(v <= FERTILITY_GATE for v in x.values()), x


def test_score_matches_manifest():
    H.require_build()
    manifest = json.loads(H.MANIFEST.read_text(encoding="utf-8"))
    per_lang = manifest["results"]["per_language"]
    x = _fertilities()
    for lang in LANGS:
        assert round(x[lang], 4) == round(per_lang[lang]["X"], 4), lang
