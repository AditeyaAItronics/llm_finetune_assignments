# India BPE Tokenizer (en · hi · te · bn)

A single **byte-level BPE tokenizer** with a **10,000-token vocabulary**, trained on the Wikipedia "India" article in **English, Hindi, Telugu, and Bengali**. Built for one goal: keep per-language token efficiency (*fertility*) not just low, but **equal across all four languages**.

- **Live widget:** `<NETLIFY_URL>` — computes all ratios, token stats, and the self-score in your browser.
- **Direct tokenizer download:** `<NETLIFY_URL>/tokenizer/tokenizer.json`
- **Design doc:** [`BPE_Tokenizer_Design_Spec.md`](./BPE_Tokenizer_Design_Spec.md)

> Numbers below are computed by the pipeline and are **not** hand-edited. Anyone can reproduce them with `python src/score.py` against the shipped tokenizer and corpus.

---

## Results

| Language | Words | Tokens | `X = tokens/words` | ≤ 1.2 |
|---|---:|---:|---:|:---:|
| English (en) | `<W_EN>` | `<T_EN>` | `<X_EN>` | ✅ |
| Hindi (hi)   | `<W_HI>` | `<T_HI>` | `<X_HI>` | ✅ |
| Telugu (te)  | `<W_TE>` | `<T_TE>` | `<X_TE>` | ✅ |
| Bengali (bn) | `<W_BN>` | `<T_BN>` | `<X_BN>` | ✅ |

- **Spread** `Δ = X_max − X_min = <DELTA>`
- **Self-score** `S = 1000 / Δ = <SCORE>`
- **Vocab size:** 10,000 (exact) · **UNK:** none (byte-level base)

*(Placeholders are filled in automatically by the build; see [Reproduce](#reproduce-the-numbers).)*

---

## How the score works

For each language `i`:

```
X_i   = tokens_i / words_i            # fertility: tokens the encoder emits ÷ word count
gate  : every X_i <= 1.2              # required for all four languages
Δ     = max(X_i) - min(X_i)           # spread across the four languages
score = 1000 / Δ
```

The score depends **only on the spread Δ**, not on absolute fertility (as long as the ≤ 1.2 gate holds). So the design optimizes for *equal* fertility across scripts, not just low fertility. See the [design doc](./BPE_Tokenizer_Design_Spec.md) §2 and §7 for the full reasoning and the balancing algorithm.

---

## How it works (short version)

1. **Byte-level BPE** (HuggingFace `tokenizers`) → zero `UNK` for any script, and a 256-byte base shared by all four languages.
2. **Corpus-mixture weighting** — one joint tokenizer is trained, but each language's corpus is replicated by an integer factor so harder scripts (the Indic ones) get more influence. A small control loop nudges the weights to minimize the fertility spread Δ.
3. **One shared preprocessing function** (NFC normalization + whitespace cleanup) is used identically by training, scoring, and the browser widget — so numerator and denominator always see the same text.
4. **Three independent checks agree to 4 dp**: the build's `manifest.json`, `score.py`, and the widget's in-browser tokenizer (`bpe.js`, verified to match Python exactly). Confirm before deploying.

---

## Repo layout

```
india-bpe/
├── README.md
├── BPE_Tokenizer_Design_Spec.md   # full design doc
├── build.sh                        # end-to-end: fetch → train → export → score → widget
├── pyproject.toml                  # deps + Python pin (uv)
├── uv.lock                         # exact locked versions (reproducibility)
├── corpus/                         # pinned, post-preprocessing text (shipped)
│   ├── en.txt  hi.txt  te.txt  bn.txt
│   └── manifest.json               # oldids, hashes, versions
├── src/
│   ├── fetch_corpus.py             # resolve oldids, pull plain text from the Wikipedia API
│   ├── preprocess.py               # shared NFC + cleanup + word_count (imported everywhere)
│   ├── curve_sweep.py              # optional: per-language fertility curves
│   ├── train_balance.py            # balancing control loop → tokenizer.json
│   ├── build_union.py              # fallback: union of per-language BPEs
│   ├── export_artifacts.py         # vocab.json, merges.txt, tokens.csv; asserts vocab==10000
│   ├── score.py                    # canonical scorer (run this to verify)
│   └── build_widget.py             # assemble dist/
├── widget/                         # static SPA source
│   ├── index.html  app.js  bpe.js  preprocess.js  styles.css
├── tests/                          # parity, vocab_size, no_unk, gate, score-matches-manifest
└── dist/                           # deployable bundle (widget + tokenizer/ + corpus/)
    └── tokenizer/
        ├── tokenizer.json          # source of truth for encoding
        ├── vocab.json              # {token: id}
        ├── merges.txt              # ordered merge rules
        └── tokens.csv              # id, token, decoded, lang_tag (human-friendly)
```

---

## Quickstart

### Requirements

- [`uv`](https://docs.astral.sh/uv/) (manages Python + deps)
- Python 3.11+ (uv installs it automatically if missing)

```bash
uv sync            # creates .venv and installs pinned deps from pyproject.toml / uv.lock
```

Dependencies (`tokenizers`, `regex`, `requests`) and the exact versions are pinned in `pyproject.toml` + `uv.lock` — the lockfile is what makes re-runs byte-identical.

### Build everything

```bash
./build.sh
# uv sync → fetch_corpus → train_balance → export_artifacts → score → build_widget
```

Only `fetch_corpus.py` needs network access; everything else runs offline from the shipped `corpus/`.

**What the build produces:** the deployable site in **`dist/`**. This is the key point — you *edit* `widget/`, but you *deploy* `dist/`. The build copies the `widget/` files into `dist/` and adds the generated data the page needs:

```
dist/
├── index.html  app.js  bpe.js  preprocess.js  styles.css   # copied from widget/
├── tokenizer/   tokenizer.json  vocab.json  merges.txt  tokens.csv
├── corpus/      en.txt  hi.txt  te.txt  bn.txt
└── manifest.json
```

The `widget/` folder on its own is **not** deployable — the page fetches `tokenizer/*` and `corpus/*` at runtime, and those only exist after the build.

### Reproduce the numbers

```bash
uv run python src/score.py
```

This loads `dist/tokenizer/tokenizer.json` and the corpus, applies the **same** preprocessing as training, and prints per-language `tokens`, `words`, `X_i`, the spread, and the score. Its output matches `corpus/manifest.json` and the widget exactly (to 4 dp).

### Preview the widget locally

```bash
cd dist && uv run python -m http.server 8000
# open http://localhost:8000
```

### Deploy (Netlify)

Deploy the **`dist/`** folder (not `widget/`):

```bash
netlify deploy --dir=dist --prod
```

Or drag-and-drop the `dist/` folder into the Netlify dashboard ("Add new site → Deploy manually"). After deploy:

- **Widget link:** `https://<site>.netlify.app`
- **Tokenizer.json (direct download):** `https://<site>.netlify.app/tokenizer/tokenizer.json`

Verify both in an incognito window before submitting.

---

## Using the tokenizer

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("dist/tokenizer/tokenizer.json")
ids = tok.encode("भारत एक देश है।", add_special_tokens=False).ids
print(len(ids), ids)
print(tok.decode(ids))          # round-trips with zero UNK
```

Download the raw tokenizer directly (no auth, not Google Drive):
`<NETLIFY_URL>/tokenizer/tokenizer.json`

---

## Reproducibility

Wikipedia changes daily, so the corpus is **pinned by revision id (`oldid`)** and the extracted text is **shipped** in `corpus/`. The bundle is the source of truth: running `python src/score.py` on it reproduces the widget's numbers. The `oldid`s, timestamps, library versions, and text hashes are recorded in `corpus/manifest.json` so a grader can align to the exact input if they re-fetch.

---

## Interpretation & assumptions

The assignment wording ("*Total English tokens / Total English Vocab, say 5000 words*") is read as follows (also shown in the widget):

- **`X_i` = tokens ÷ total words** (fertility). The widget has a toggle for the tokens-÷-*unique*-words reading.
- **Eval text = the full pinned article** per language (switchable to a fixed word slice via config).
- **10,000 = one shared tokenizer's `vocab_size`**, each entry counted once (shared subwords not double-counted).

Locked assumptions and their alternatives are listed in the [design doc](./BPE_Tokenizer_Design_Spec.md) Appendix E (Assumptions & clarifications).

---

## Submission

| Field | Link |
|---|---|
| **Widget** | `<NETLIFY_URL>` |
| **Tokenizer.json** | `<NETLIFY_URL>/tokenizer/tokenizer.json` |

Both are public static files on Netlify — verified accessible in an incognito window (no login, not a Google Drive link).

---

## License / attribution

Source text: Wikipedia "India" articles (en/hi/te/bn), CC BY-SA; revision ids recorded in `corpus/manifest.json`. Code in this repo is provided for the assignment.
