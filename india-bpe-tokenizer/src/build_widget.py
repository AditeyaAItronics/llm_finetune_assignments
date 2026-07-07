"""Step 5 — assemble the deployable dist/ bundle.

Logic:
  - copy corpus/*.txt into dist/corpus/ so the widget can score the exact same text
  - copy widget/* static files into dist/
  - copy the results manifest to dist/manifest.json for the widget to display

After this, `dist/` is a self-contained static site: deploy with
    netlify deploy --dir=dist --prod
The tokenizer is already at dist/tokenizer/ (written by train + export).
"""

from __future__ import annotations

import shutil

from common import (
    CORPUS_DIR,
    DIST_CORPUS_DIR,
    DIST_DIR,
    LANGS,
    TOKENIZER_DIR,
    WIDGET_DIR,
)


def main() -> None:
    DIST_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        shutil.copy(CORPUS_DIR / f"{lang}.txt", DIST_CORPUS_DIR / f"{lang}.txt")

    if WIDGET_DIR.exists():
        for path in WIDGET_DIR.iterdir():
            if path.is_file():
                shutil.copy(path, DIST_DIR / path.name)
    else:
        print(f"NOTE: {WIDGET_DIR} not found — widget files not copied yet.")

    manifest = TOKENIZER_DIR / "manifest.json"
    if manifest.exists():
        shutil.copy(manifest, DIST_DIR / "manifest.json")

    print(f"assembled bundle at {DIST_DIR}")
    print("deploy with:  netlify deploy --dir=dist --prod")


if __name__ == "__main__":
    main()
