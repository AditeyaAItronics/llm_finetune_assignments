"""Step 2 (primary path) — train ONE joint 10k byte-level BPE, balanced across languages.

Core idea (see design doc §7):
  The score is 1000 / (X_max - X_min), so we don't just want low fertility, we want
  EQUAL fertility across en/hi/te/bn. We train a single joint BPE, but replicate each
  language's corpus by an integer factor `weights[lang]` so harder scripts exert more
  influence on merge selection. A small control loop nudges the weights:
      - measure the four fertilities
      - give the WORST (highest fertility) language more weight
      - ease off the BEST (lowest fertility) language
      - anneal the step size when the spread stops improving
  We keep the tokenizer with the smallest spread that also passes the <=1.2 gate.

Integer replication (not random sampling) keeps training bit-reproducible.
"""

from __future__ import annotations

import json

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from common import CORPUS_DIR, FERTILITY_GATE, LANGS, TOKENIZER_DIR, VOCAB_SIZE
from preprocess import preprocess, word_count

MAX_ITERS = 20      # control-loop iterations (each is one full retrain)
PATIENCE = 3        # iters without improvement before we anneal the step
INIT_STEP = 3       # initial weight increment


def load_corpora() -> dict[str, str]:
    """Load and (idempotently) re-preprocess each corpus."""
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
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=ByteLevel.alphabet(),  # all 256 bytes -> zero UNK
        special_tokens=[],
        show_progress=False,
    )
    tok.train_from_iterator(_weighted_iter(corpora, weights), trainer=trainer)
    return tok


def fertility(tok: Tokenizer, text: str) -> float:
    return len(tok.encode(text).ids) / word_count(text)


def evaluate(tok: Tokenizer, corpora: dict[str, str]) -> dict[str, float]:
    return {lang: fertility(tok, corpora[lang]) for lang in LANGS}


def _spread(x: dict[str, float]) -> float:
    return max(x.values()) - min(x.values())


def main() -> None:
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    corpora = load_corpora()
    weights = {lang: 1 for lang in LANGS}
    step = INIT_STEP
    stale = 0
    best: dict | None = None

    for it in range(1, MAX_ITERS + 1):
        tok = train_joint(corpora, weights)
        x = evaluate(tok, corpora)
        gated = all(v <= FERTILITY_GATE for v in x.values())
        delta = _spread(x)
        print(
            f"iter {it:2d}  w={weights}  "
            f"X={ {k: round(v, 4) for k, v in x.items()} }  "
            f"Δ={delta:.4f}  [{'ok' if gated else 'GATE-FAIL'}]"
        )

        if gated and (best is None or delta < best["spread"]):
            best = {"weights": dict(weights), "spread": delta, "X": dict(x)}
            tok.save(str(TOKENIZER_DIR / "tokenizer.json"))
            stale = 0
        else:
            stale += 1

        # nudge weights: reward the worst language, relax the best
        worst = max(x, key=x.get)
        best_lang = min(x, key=x.get)
        weights[worst] += step
        if weights[best_lang] > 1:
            weights[best_lang] = max(1, weights[best_lang] - step)

        if stale >= PATIENCE:
            step //= 2
            stale = 0
        if step == 0:
            break

    if best is None:
        raise SystemExit(
            "No configuration passed the <=1.2 gate. "
            "Check that corpora were fetched and are large enough for 10k merges."
        )

    (TOKENIZER_DIR / "balance_meta.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nbest Δ={best['spread']:.4f}  weights={best['weights']}")
    print(f"saved {TOKENIZER_DIR / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
