"""
Pull a real sample of AI4Bharat Sangraha (verified split) via the HuggingFace
datasets-server /rows API. No huggingface_hub / datasets lib needed - this is
plain REST, so it stays fast and dependency-light.

Output: data/raw_sample.jsonl (one JSON object per row) + data/dataset_stats.json
(the real, whole-dataset size numbers pulled from the /size endpoint, so the
widget can honestly show "sample used" vs "full population").
"""
import json
import time
import urllib.request
from pathlib import Path

BASE = "https://datasets-server.huggingface.co"
DATASET = "ai4bharat/sangraha"
CONFIG = "verified"
LANGS = ["eng", "hin", "tel", "ben", "mal", "tam", "guj", "mar", "urd", "pan", "kan", "asm"]
ROWS_PER_LANG = 300
PAGE = 100

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5)


def fetch_rows(lang, n):
    out = []
    offset = 0
    while len(out) < n:
        length = min(PAGE, n - len(out))
        url = f"{BASE}/rows?dataset={DATASET}&config={CONFIG}&split={lang}&offset={offset}&length={length}"
        try:
            data = fetch_json(url)
        except Exception as e:
            print(f"  ! {lang}: stopped early at offset {offset} ({e})")
            break
        rows = data.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            row["_lang_claimed"] = lang
            out.append(row)
        offset += length
    return out


def fetch_size():
    return fetch_json(f"{BASE}/size?dataset={DATASET}")


def main():
    print("Fetching real whole-dataset size stats ...")
    size_info = fetch_size()
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(size_info, indent=2), encoding="utf-8")

    all_rows = []
    for lang in LANGS:
        print(f"Fetching {ROWS_PER_LANG} rows for verified/{lang} ...")
        rows = fetch_rows(lang, ROWS_PER_LANG)
        print(f"  -> got {len(rows)}")
        all_rows.extend(rows)

    out_path = OUT_DIR / "raw_sample.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nTotal sample rows fetched: {len(all_rows)}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
