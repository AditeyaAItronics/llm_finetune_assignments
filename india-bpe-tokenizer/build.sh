#!/usr/bin/env bash
# End-to-end build: scrape -> train+balance -> export -> score -> assemble bundle.
# Requires `uv` (https://docs.astral.sh/uv/). Run from the repo root: ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 0/5  sync environment (uv)"
uv sync

echo "==> 1/5  fetch corpus (needs network)"
uv run python src/fetch_corpus.py

# Optional diagnostic — uncomment to see per-language fertility curves:
# uv run python src/curve_sweep.py

echo "==> 2/5  train + balance (primary path)"
uv run python src/train_balance.py
# Fallback path if the spread is unsatisfactory:
# uv run python src/build_union.py

echo "==> 3/5  export artifacts (vocab.json, merges.txt, tokens.csv, manifest)"
uv run python src/export_artifacts.py

echo "==> 4/5  score (verify <=1.2 gate + print numbers)"
uv run python src/score.py

echo "==> 5/5  assemble dist/ bundle"
uv run python src/build_widget.py

echo
echo "Done. Deploy the widget + tokenizer:"
echo "    netlify deploy --dir=dist --prod"
echo "Tokenizer download URL will be:  https://<site>.netlify.app/tokenizer/tokenizer.json"
