"""The tokenizer vocabulary must be exactly 10,000."""
import _helpers as H
from tokenizers import Tokenizer

from common import VOCAB_SIZE


def test_vocab_size_is_exactly_10000():
    H.require_build()
    tok = Tokenizer.from_file(str(H.TOK))
    assert tok.get_vocab_size() == VOCAB_SIZE
