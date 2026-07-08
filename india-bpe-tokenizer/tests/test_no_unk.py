"""Byte-level base => zero UNK. Every corpus must round-trip cleanly."""
import _helpers as H
from tokenizers import Tokenizer

from common import LANGS
from preprocess import preprocess


def test_round_trip_no_unk():
    H.require_build()
    tok = Tokenizer.from_file(str(H.TOK))
    vocab_size = tok.get_vocab_size()
    for lang in LANGS:
        text = preprocess((H.CORPUS / f"{lang}.txt").read_text(encoding="utf-8"))
        enc = tok.encode(text)
        # all ids are real vocab entries (no out-of-range / UNK sentinel)
        assert all(0 <= i < vocab_size for i in enc.ids), lang
        decoded = tok.decode(enc.ids)
        assert "�" not in decoded, f"{lang}: replacement char in round-trip"
        assert decoded.strip() == text.strip(), f"{lang}: round-trip mismatch"
