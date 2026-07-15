"""Step 4 — the canonical scorer. THIS is what the grader runs.

Deliberately short and dependency-light. Loads the tokenizer + the shipped corpus,
applies the SAME preprocessing as training, and prints per-language tokens/words/X,
the spread, and the score = 1000 / (X_max - X_min).

Per the assignment, only ENGLISH (X1) carries the "<= 1.2" requirement; the other three
languages just feed the spread. So we report the English gate but ALWAYS print a score
and exit 0 (the score is defined regardless of the other languages' ratios).

Its output must match dist/tokenizer/manifest.json['results'] and the widget, to 4 dp.
"""

from __future__ import annotations

import json

from tokenizers import Tokenizer

from common import CORPUS_DIR, FERTILITY_GATE, LANGS, TOKENIZER_DIR
from preprocess import normalize_spaces, preprocess, word_count


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))

    per_language = {}
    for lang in LANGS:
        text = preprocess((CORPUS_DIR / f"{lang}.txt").read_text(encoding="utf-8"))
        toks = len(tok.encode(normalize_spaces(text)).ids)
        words = word_count(text)
        per_language[lang] = {"tokens": toks, "words": words, "X": toks / words}

    xs = {lang: per_language[lang]["X"] for lang in LANGS}
    xmax, xmin = max(xs.values()), min(xs.values())
    spread = xmax - xmin

    out = {
        "per_language": per_language,
        "X_max": xmax,
        "X_min": xmin,
        "spread": spread,
        "score": (1000 / spread) if spread > 1e-9 else None,
        "english_gate_pass": xs["en"] <= FERTILITY_GATE,  # only English carries the <=1.2 rule
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
