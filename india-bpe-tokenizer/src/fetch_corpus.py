"""Step 1 — scrape the four Wikipedia 'India' articles and pin them.

Logic:
  1. For each language, resolve the CURRENT revision id (oldid) + timestamp via the
     MediaWiki Action API (prop=revisions). That oldid is what we "pin" — we record it
     so the grader can align to the exact same source revision.
  2. Pull the plain-text extract of that article (prop=extracts&explaintext=1). We use
     the extracts API, NOT rendered HTML, so there is no nav chrome / skin noise.
  3. Preprocess (NFC + cleanup) with the SHARED preprocess module.
  4. Write corpus/<lang>.txt and record oldid, timestamp, sha256(text), chars, words
     in corpus/manifest.json.

The shipped corpus/*.txt files are the source of truth for training and scoring;
sha256 lets anyone confirm byte-identical input.
"""

from __future__ import annotations

import hashlib
import json
import time

import requests

from common import ARTICLES, CORPUS_DIR, LANGS
from preprocess import preprocess, word_count

# Wikipedia asks for a descriptive User-Agent; requests without one may be blocked.
USER_AGENT = "india-bpe-tokenizer/0.1 (assignment; amit.kayal@expeditecommerce.com)"


def _api(host: str, params: dict) -> dict:
    resp = requests.get(
        f"https://{host}/w/api.php",
        params={**params, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _current_revision(host: str, title: str) -> tuple[str, int, str]:
    """Return (resolved_title, oldid, timestamp) of the current revision."""
    data = _api(host, {
        "action": "query", "prop": "revisions", "titles": title,
        "rvprop": "ids|timestamp", "rvlimit": 1, "redirects": 1,
    })
    page = next(iter(data["query"]["pages"].values()))
    rev = page["revisions"][0]
    return page["title"], rev["revid"], rev["timestamp"]


def _extract_text(host: str, title: str) -> str:
    """Plain-text extract of the current article."""
    data = _api(host, {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "redirects": 1,
    })
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"]


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extractor": "mediawiki action api prop=extracts explaintext=1",
        "articles": {},
    }

    for lang in LANGS:
        host, title = ARTICLES[lang]
        resolved_title, oldid, ts = _current_revision(host, title)
        raw = _extract_text(host, resolved_title)
        text = preprocess(raw)

        (CORPUS_DIR / f"{lang}.txt").write_text(text, encoding="utf-8")
        manifest["articles"][lang] = {
            "host": host,
            "title": resolved_title,
            "oldid": oldid,
            "revision_timestamp": ts,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chars": len(text),
            "words": word_count(text),
        }
        info = manifest["articles"][lang]
        print(f"[{lang}] {resolved_title} oldid={oldid} words={info['words']} chars={info['chars']}")

    (CORPUS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {CORPUS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
