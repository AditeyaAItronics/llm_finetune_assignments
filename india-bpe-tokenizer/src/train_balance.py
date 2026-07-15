"""Step 2 (primary path) — train ONE joint 10k CHARACTER-level BPE, English-biased.

Design (see design doc §6-§7 and the assignment):
  * FAITHFUL ROUND-TRIP GATE (the pass/fail criterion): decode(encode(text)) must preserve the
    same visible, non-whitespace characters, for arbitrary Markdown/URL text. We guarantee this by
    (a) seeding printable ASCII + Latin-1 into the alphabet as normal characters, so Markdown/URL
    punctuation (# _ ` * [ ] etc.) is always in-vocab, and (b) marking the byte-fallback tokens
    NON-special (see _save_faithful) so decode() doesn't skip them. Without both, characters absent
    from the Wikipedia corpus get silently dropped on decode and the whole submission scores 0.
  * The score is 1000 / (X_max - X_min) over the four per-language ratios X_i = tokens_i/words_i.
    Only ENGLISH (X1) must be <= 1.2; the other three just feed the spread.
  * BASE ALPHABET = characters, not bytes. Byte-level BPE charges every Indic glyph 3 UTF-8
    bytes before any merge, which is the whole reason byte-level fertility is 3-6x for Indic.
    Character-level BPE (the ORIGINAL BPE, Sennrich 2016) removes that penalty, collapsing the
    Indic fertilities into the same band as English and shrinking the spread dramatically.
  * BYTE-FALLBACK keeps the zero-UNK guarantee: any character never seen in training is emitted
    as its UTF-8 byte tokens (<0x00>..<0xFF>, which live inside the 10k vocab) instead of UNK.
  * ENGLISH-BIASED weighting: char-level makes the spread small but pushes English slightly ABOVE
    1.2 at flat weights. So we replicate the English corpus more heavily, giving English more
    merge budget until X_en <= 1.2, and keep the smallest-spread config that still clears the gate.

Fertility is measured on whitespace-normalized text (preprocess.normalize_spaces) so every word is
exactly one ``▁word`` pre-token — identical in Python and in the browser JS encoder.

Integer replication (not random sampling) keeps training bit-reproducible.
"""

from __future__ import annotations

import json

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

from common import CORPUS_DIR, FERTILITY_GATE, LANGS, TOKENIZER_DIR, VOCAB_SIZE
from preprocess import normalize_spaces, preprocess, word_count

GATE_LANG = "en"                              # the one language the assignment gates at <= 1.2
BYTE_FALLBACK_TOKENS = [f"<0x{i:02X}>" for i in range(256)]  # UTF-8 byte tokens -> zero UNK
# Printable ASCII (0x20-0x7E) + Latin-1 supplement (0xA0-0xFF), seeded as NORMAL alphabet chars so
# Markdown/URL punctuation absent from the Wikipedia corpus (# _ ` * [ ] …) is in-vocab and round-trips.
INIT_ALPHABET = [chr(c) for c in range(0x20, 0x7F)] + [chr(c) for c in range(0xA0, 0x100)]
# English replication factors to sweep (others pinned at 1). Rising en-weight pushes X_en down;
# the first factor that clears the 1.2 gate is also the smallest-spread passing config.
EN_WEIGHTS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]


def load_corpora() -> dict[str, str]:
    return {
        lang: preprocess((CORPUS_DIR / f"{lang}.txt").read_text(encoding="utf-8"))
        for lang in LANGS
    }


def _paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n") if p.strip()]


def _weighted_iter(corpora: dict[str, str], weights: dict[str, int]):
    """Yield each language's paragraphs `weights[lang]` times (integer replication)."""
    for lang in LANGS:
        paras = _paragraphs(corpora[lang])
        for _ in range(max(1, weights[lang])):
            yield from paras


def train_joint(corpora: dict[str, str], weights: dict[str, int],
                vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = Metaspace()                       # char-level; marks word starts with ▁
    tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Metaspace()])
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=INIT_ALPHABET,                   # ASCII + Latin-1 always in-vocab
        special_tokens=BYTE_FALLBACK_TOKENS,              # 256 byte tokens counted inside the 10k
        show_progress=False,
    )
    tok.train_from_iterator(_weighted_iter(corpora, weights), trainer=trainer)
    return tok


def _save_faithful(tok: Tokenizer, path) -> None:
    """Save tokenizer.json, but first mark the byte-fallback tokens NON-special.

    Trainers register special_tokens as special, and Tokenizer.decode() skips special tokens by
    default — which would silently drop any byte-fallback character on decode and fail the faithful
    round-trip gate. Flipping their `special` flag to false keeps them in decode output while leaving
    them in the model vocab so byte-fallback at encode time still works.
    """
    j = json.loads(tok.to_str())
    for t in j.get("added_tokens", []):
        c = t.get("content", "")
        if len(c) == 6 and c.startswith("<0x") and c.endswith(">"):
            t["special"] = False
    path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")


def fertility(tok: Tokenizer, text: str) -> float:
    return len(tok.encode(normalize_spaces(text)).ids) / word_count(text)


def evaluate(tok: Tokenizer, corpora: dict[str, str]) -> dict[str, float]:
    return {lang: fertility(tok, corpora[lang]) for lang in LANGS}


def _spread(x: dict[str, float]) -> float:
    return max(x.values()) - min(x.values())


def main() -> None:
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    corpora = load_corpora()

    best: dict | None = None       # smallest-spread config that clears the English gate
    fallback: dict | None = None   # lowest X_en seen, used only if nothing clears the gate

    for en_w in EN_WEIGHTS:
        weights = {lang: 1 for lang in LANGS}
        weights[GATE_LANG] = en_w
        tok = train_joint(corpora, weights)
        x = evaluate(tok, corpora)
        en_ok = x[GATE_LANG] <= FERTILITY_GATE
        delta = _spread(x)
        print(
            f"en_w={en_w:2d}  X={ {k: round(v, 4) for k, v in x.items()} }  "
            f"Δ={delta:.4f}  [{GATE_LANG} {'≤1.2 ok' if en_ok else '>1.2'}]"
        )

        meta = {
            "weights": dict(weights),
            "spread": delta,
            "X": {k: round(v, 6) for k, v in x.items()},
            "vocab_size": tok.get_vocab_size(),
            "english_gate_pass": en_ok,
        }
        if fallback is None or x[GATE_LANG] < fallback["X"][GATE_LANG]:
            fallback = {**meta, "_tok": tok}
        if en_ok and (best is None or delta < best["spread"]):
            best = {**meta, "_tok": tok}

    chosen = best if best is not None else fallback
    assert chosen is not None, "no configuration trained — were the corpora fetched?"
    _save_faithful(chosen["_tok"], TOKENIZER_DIR / "tokenizer.json")
    meta_out = {k: v for k, v in chosen.items() if k != "_tok"}
    (TOKENIZER_DIR / "balance_meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nchosen weights={meta_out['weights']}  Δ={meta_out['spread']:.4f}  "
        f"score={1000 / meta_out['spread']:.1f}  vocab={meta_out['vocab_size']}"
    )
    if best is None:
        print(
            f"WARNING: no config cleared X_en ≤ 1.2; shipped the closest "
            f"(X_en={meta_out['X'][GATE_LANG]:.4f}). Widen EN_WEIGHTS or check the corpus."
        )
    print(f"saved {TOKENIZER_DIR / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
