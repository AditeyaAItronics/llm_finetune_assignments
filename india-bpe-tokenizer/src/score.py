"""Step 4 — the canonical scorer. THIS is what the grader runs.

Deliberately short and dependency-light. Loads the tokenizer + the shipped corpus,
applies the SAME preprocessing as training, and prints per-language tokens/words/X,
the spread, and the score. Exits non-zero if any X_i > 1.2 (gate fail).

Its output must match dist/tokenizer/manifest.json['results'] and the widget, to 4 dp.
"""

from __future__ import annotations

import json

from tokenizers import Tokenizer

from common import CORPUS_DIR, FERTILITY_GATE, LANGS, TOKENIZER_DIR
from preprocess import preprocess, word_count


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))

    per_language = {}
    for lang in LANGS:
        text = preprocess((CORPUS_DIR / f"{lang}.txt").read_text(encoding="utf-8"))
        toks = len(tok.encode(text).ids)
        words = word_count(text)
        per_language[lang] = {"tokens": toks, "words": words, "X": toks / words}

    xs = {lang: per_language[lang]["X"] for lang in LANGS}
    xmax, xmin = max(xs.values()), min(xs.values())
    spread = xmax - xmin
    gate_pass = all(v <= FERTILITY_GATE for v in xs.values())

    out = {
        "per_language": per_language,
        "X_max": xmax,
        "X_min": xmin,
        "spread": spread,
        "score": (1000 / spread) if spread > 1e-9 else None,
        "gate_pass": gate_pass,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if not gate_pass:
        raise SystemExit("GATE FAILED: some X_i > 1.2")


if __name__ == "__main__":
    main()
