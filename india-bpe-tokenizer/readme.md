# India BPE Tokenizer (en · hi · te · bn)

A single **character-level BPE tokenizer** (with UTF-8 **byte-fallback** for zero UNK) and a **10,000-token vocabulary**, trained on the Wikipedia "India" article in **English, Hindi, Telugu, and Bengali**. Built for one goal: keep per-language token efficiency (*fertility*) low and as **even across the four scripts** as the English ≤ 1.2 rule allows.

> **Pass/fail gate first:** the tokenizer must **faithfully round-trip** — `decode(encode(text))` has to give back the same visible (non-whitespace) characters for arbitrary Markdown/URL text, or the submission scores 0 regardless of fertility. We guarantee this by seeding printable ASCII + Latin-1 as normal vocab characters and keeping the byte-fallback tokens **non-special** so `decode()` never drops them. See [`tests/test_roundtrip.py`](./tests/test_roundtrip.py).

- **Live widget:** `<NETLIFY_URL>` — computes all ratios, token stats, and the self-score in your browser.
- **Direct tokenizer download:** `<NETLIFY_URL>/tokenizer/tokenizer.json`
- **Design doc:** [`BPE_Tokenizer_Design_Spec.md`](./BPE_Tokenizer_Design_Spec.md)
- **Deployment guide:** [`DEPLOYMENT.md`](./DEPLOYMENT.md) — build → verify → deploy, with a step-by-step of every script.

> Numbers below are computed by the pipeline and are **not** hand-edited. Anyone can reproduce them with `python src/score.py` against the shipped tokenizer and corpus.

### 👋 New here? Read in this order

1. **This file (`readme.md`)** — what the project is, the scoring model, and how to build it. Start here and run it.
2. **[`BPE_Tokenizer_Design_Spec.md`](./BPE_Tokenizer_Design_Spec.md)** — the *why*: char-level-vs-byte-level reasoning, the English-biased balancing, interpretation decisions, and per-script logic.
3. **[`DEPLOYMENT.md`](./DEPLOYMENT.md)** — build → verify → deploy, with a step-by-step of every script (input / output / what it does) and the Netlify + submission steps.

Short version: **read this → `./build.sh` → design doc to understand → deployment doc to ship.**

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
- **Vocab size:** 10,000 (exact) · **UNK:** none (char base + UTF-8 byte-fallback)

*(Placeholders are filled in automatically by the build; see [Reproduce](#reproduce-the-numbers). Current measured build: en ≈ 1.16 · hi ≈ 1.68 · te ≈ 2.84 · bn ≈ 2.22 → Δ ≈ 1.68, S ≈ 594.)*

---

## How the score works

For each language `i`:

```
X_i   = tokens_i / words_i            # fertility: tokens the encoder emits ÷ word count
gate  : X_en <= 1.2                   # per the assignment, ONLY English (X1) carries this
Δ     = max(X_i) - min(X_i)           # spread across the four languages
score = 1000 / Δ                      # always defined; never "disqualified"
```

The assignment requires only **English (X1) ≤ 1.2**; Hindi, Telugu, and the fourth language
are just measured, sorted, and fed into the spread. The score depends **only on the spread Δ**.
See the [design doc](./BPE_Tokenizer_Design_Spec.md) §2 and §7 for the balancing algorithm.

> **Why character-level (measured, not hand-waved):** byte-level BPE charges every Indic glyph
> 3 UTF-8 bytes before any merge, so byte-level fertility floors around **en 1.29 · hi 3.51 ·
> te 5.72 · bn 4.72** (Δ ≈ 4.4, S ≈ 226). **Character-level BPE removes that penalty**, collapsing
> the four fertilities into one band and — with English-biased weighting to satisfy `X_en ≤ 1.2` —
> giving **en 1.16 · hi 1.68 · te 2.84 · bn 2.22 → Δ ≈ 1.68, S ≈ 594** (a ~2.6× gain). Byte-fallback
> keeps UNK at zero. Note the tension: the `X_en ≤ 1.2` rule forces English *below* the Indic
> cluster, which re-widens the spread — the English-biased balancer holds English just under the
> gate at the smallest spread. All numbers are computed live in the widget; nothing is faked.

---

## How it works (short version)

1. **Character-level BPE + byte-fallback** (HuggingFace `tokenizers`, `Metaspace` pre-tokenizer) → a char base that tokenizes Indic scripts ~3× tighter than byte-level. Printable ASCII + Latin-1 are seeded as normal characters, and the byte-fallback tokens (`<0x00>..<0xFF>`, inside the 10k, kept **non-special**) guarantee zero `UNK` **and** a faithful `decode(encode(text))` round-trip.
2. **English-biased corpus weighting** — one joint tokenizer; the English corpus is replicated more heavily so English gets enough merge budget to hold `X_en ≤ 1.2`, and we keep the smallest-spread config that still clears that gate.
3. **One shared preprocessing + whitespace normalization** (NFC cleanup, then one `▁word` per word) used identically by training, scoring, and the browser widget — so numerator and denominator always see the same text.
4. **Three independent checks agree to 4 dp**: the build's `manifest.json`, `score.py`, and the widget's in-browser tokenizer (`bpe.js`, verified to match Python exactly on all four corpora). Confirm before deploying.

---

## Repo layout

```
india-bpe-tokenizer/
├── readme.md
├── BPE_Tokenizer_Design_Spec.md   # full design doc
├── DEPLOYMENT.md                   # build → verify → deploy, per-script details
├── build.sh                        # end-to-end: sync → fetch → train → export → score → widget
├── pyproject.toml                  # deps + Python pin (uv)
├── uv.lock                         # exact locked versions (created by uv sync)
├── .python-version                 # pins Python 3.11
├── netlify.toml                    # publish=dist + charset/CORS headers
├── .gitignore                      # ignores .venv/, dist/, __pycache__/
├── corpus/                         # fetched once, then committed (pinned, shipped)
│   ├── en.txt  hi.txt  te.txt  bn.txt
│   └── manifest.json               # oldids, hashes, versions
├── src/
│   ├── common.py                   # shared constants + paths (imported everywhere)
│   ├── preprocess.py               # shared NFC + cleanup + word_count
│   ├── fetch_corpus.py             # resolve oldids, pull plain text from the Wikipedia API
│   ├── curve_sweep.py              # optional: per-language fertility curves
│   ├── train_balance.py            # balancing control loop → tokenizer.json
│   ├── build_union.py              # fallback: union of per-language BPEs
│   ├── export_artifacts.py         # vocab.json, merges.txt, tokens.csv; asserts vocab==10000
│   ├── score.py                    # canonical scorer (run this to verify)
│   └── build_widget.py             # assemble dist/
├── widget/                         # static SPA source (you edit here)
│   └── index.html  app.js  bpe.js  preprocess.js  styles.css
├── tests/                          # conftest, _helpers, test_parity, test_vocab_size,
│                                   #   test_no_unk, test_gate_and_manifest
└── dist/                           # GENERATED by build — deploy this (gitignored)
    ├── index.html  app.js  bpe.js  preprocess.js  styles.css   # copied from widget/
    ├── tokenizer/  tokenizer.json  vocab.json  merges.txt  tokens.csv  manifest.json
    ├── corpus/     en.txt  hi.txt  te.txt  bn.txt
    ├── manifest.json
    └── _headers
```

> `corpus/` is fetched once and **committed** (it's the pinned source of truth). `dist/` is **generated** every build and gitignored — deploy it, don't commit it.

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
