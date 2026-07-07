"""Optional analysis — per-language fertility vs. vocab-size curves.

For each language, trains a STANDALONE byte-level BPE at several vocab sizes and
prints the resulting fertility (tokens/word). This tells us how "expensive" each
script is (Indic scripts need more merges than English to reach the same fertility),
which is useful for seeding the balancing weights in train_balance.py.

This does not produce a shipped artifact; it's a diagnostic you run by hand.
"""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from common import CORPUS_DIR, LANGS
from preprocess import preprocess, word_count

SIZES = [500, 1000, 2000, 4000, 6000, 8000]


def _train(text: str, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tok.train_from_iterator(
        (p for p in text.split("\n") if p.strip()), trainer=trainer
    )
    return tok


def main() -> None:
    for lang in LANGS:
        text = preprocess((CORPUS_DIR / f"{lang}.txt").read_text(encoding="utf-8"))
        words = word_count(text)
        cells = []
        for n in SIZES:
            tok = _train(text, n)
            fert = len(tok.encode(text).ids) / words
            cells.append(f"{n}:{fert:.3f}")
        print(f"[{lang}] " + "  ".join(cells))


if __name__ == "__main__":
    main()
