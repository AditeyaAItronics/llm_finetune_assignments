"""Step 3 — export downloadable artifacts + write the results manifest.

Logic:
  1. Load dist/tokenizer/tokenizer.json.
  2. Assert vocab size == 10000 (warn loudly otherwise).
  3. Emit vocab.json + merges.txt (via tok.model.save) and a human-friendly tokens.csv
     (id, token, decoded-UTF8, script tag).
  4. Recompute per-language tokens/words/X, the spread, and the score, and merge them
     into corpus/manifest.json -> dist/tokenizer/manifest.json (adds vocab_size, library
     versions, and the results block).
"""

from __future__ import annotations

import csv
import json
import platform

import tokenizers
from tokenizers import Tokenizer

from common import CORPUS_DIR, FERTILITY_GATE, LANGS, TOKENIZER_DIR, VOCAB_SIZE
from preprocess import normalize_spaces, preprocess, word_count


def decode_token(token: str) -> str:
    """Render a char-level token for human reading.

    Char-level tokens are already Unicode text; we just turn the Metaspace word-start
    marker ``▁`` back into a leading space. Byte-fallback tokens (``<0xE2>`` …) are byte
    fragments of some multi-byte glyph and aren't printable on their own, so we leave them
    verbatim.
    """
    if token.startswith("<0x") and token.endswith(">"):
        return token
    return token.replace("▁", " ")


def script_tag(s: str) -> str:
    for ch in s:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "Devanagari"
        if 0x0C00 <= o <= 0x0C7F:
            return "Telugu"
        if 0x0980 <= o <= 0x09FF:
            return "Bengali"
    if any(c.isascii() and c.isalpha() for c in s):
        return "Latin"
    return "shared"


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    vocab = tok.get_vocab()
    size = len(vocab)
    print(f"vocab size = {size}")
    if size != VOCAB_SIZE:
        print(f"WARNING: vocab size {size} != target {VOCAB_SIZE}")

    # vocab.json + merges.txt
    tok.model.save(str(TOKENIZER_DIR))

    # tokens.csv (sorted by id)
    with open(TOKENIZER_DIR / "tokens.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "token", "decoded", "script"])
        for token, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
            dec = decode_token(token)
            writer.writerow([tid, token, dec, script_tag(dec)])

    # results
    corpora = {
        lang: preprocess((CORPUS_DIR / f"{lang}.txt").read_text(encoding="utf-8"))
        for lang in LANGS
    }
    per_language = {}
    for lang in LANGS:
        toks = len(tok.encode(normalize_spaces(corpora[lang])).ids)
        words = word_count(corpora[lang])
        per_language[lang] = {"tokens": toks, "words": words, "X": toks / words}

    xs = {lang: per_language[lang]["X"] for lang in LANGS}
    xmax, xmin = max(xs.values()), min(xs.values())
    spread = xmax - xmin
    results = {
        "interpretation": "tokens_per_total_word",
        "per_language": per_language,
        "X_max": xmax,
        "X_min": xmin,
        "spread": spread,
        "score": (1000 / spread) if spread > 1e-9 else None,
        "english_gate_pass": xs["en"] <= FERTILITY_GATE,  # only English carries the <=1.2 rule
    }

    manifest_path = CORPUS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["vocab_size"] = size
    manifest["library"] = {
        "python": platform.python_version(),
        "tokenizers": tokenizers.__version__,
    }
    manifest["results"] = results

    (TOKENIZER_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
