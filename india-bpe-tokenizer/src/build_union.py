"""Step 2 (FALLBACK path) — union of per-language BPEs.

Use this only if train_balance.py cannot get the spread small enough. Logic:
  1. Train four STANDALONE byte-level BPEs (one per language) to a per-language size.
  2. Read each one's ordered merge list (rank = line order in merges.txt).
  3. Interleave the four lists round-robin by rank (rank-0 of each lang, then rank-1, ...),
     deduplicating, until we hit the merge budget (10000 - 256 byte tokens). Round-robin
     gives every language roughly equal representation, which is what balances fertility.
  4. Rebuild a single BPE from the byte alphabet + the merged, ordered merge list.

Note: this is a heuristic. Round-robin ranks are not globally frequency-optimal, so
encoding can be slightly less tight than the joint path. Keep whichever path yields the
smaller *reproducible* spread.
"""

from __future__ import annotations

import tempfile

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from common import BYTE_BASE, CORPUS_DIR, LANGS, TOKENIZER_DIR, VOCAB_SIZE
from preprocess import preprocess

PER_LANG_SIZE = 6000  # standalone target per language before the union


def _train_standalone(text: str, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tok.train_from_iterator((p for p in text.split("\n") if p.strip()), trainer=trainer)
    return tok


def _read_merges(tok: Tokenizer) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        tok.model.save(d)
        lines = (open(f"{d}/merges.txt", encoding="utf-8").read().splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main() -> None:
    corpora = {
        lang: preprocess((CORPUS_DIR / f"{lang}.txt").read_text(encoding="utf-8"))
        for lang in LANGS
    }
    per_lang_merges = {
        lang: _read_merges(_train_standalone(corpora[lang], PER_LANG_SIZE))
        for lang in LANGS
    }

    budget = VOCAB_SIZE - BYTE_BASE
    merged: list[str] = []
    seen: set[str] = set()
    max_len = max(len(m) for m in per_lang_merges.values())
    rank = 0
    while len(merged) < budget and rank < max_len:
        for lang in LANGS:
            merges = per_lang_merges[lang]
            if rank < len(merges) and merges[rank] not in seen:
                seen.add(merges[rank])
                merged.append(merges[rank])
                if len(merged) >= budget:
                    break
        rank += 1

    # Rebuild vocab from the byte alphabet + merges, in rank order.
    vocab: dict[str, int] = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    merges_pairs: list[tuple[str, str]] = []
    for m in merged:
        a, b = m.split(" ", 1)
        merges_pairs.append((a, b))
        combined = a + b
        if combined not in vocab:
            vocab[combined] = len(vocab)

    tok = Tokenizer(BPE(vocab=vocab, merges=merges_pairs, unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tok.decoder = ByteLevelDecoder()

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOKENIZER_DIR / "tokenizer.json"))
    print(f"union tokenizer vocab={len(vocab)} merges={len(merges_pairs)} saved.")


if __name__ == "__main__":
    main()
