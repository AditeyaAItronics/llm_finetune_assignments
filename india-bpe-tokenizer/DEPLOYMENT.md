# Deployment guide — build → verify → deploy

This is the operational companion to [`readme.md`](./readme.md) (what the project is) and
[`BPE_Tokenizer_Design_Spec.md`](./BPE_Tokenizer_Design_Spec.md) (why it's built this way).
It walks through every script, how to verify the numbers, and how to ship the two submission links.

The tokenizer is a **character-level BPE with UTF-8 byte-fallback** (zero UNK), 10,000-token vocab,
**English-biased** so `X_en ≤ 1.2`. It also passes the assignment's **faithful round-trip gate**
(`decode(encode(text))` preserves visible characters). Current measured build:
`en 1.16 · hi 1.68 · te 2.84 · bn 2.22` → Δ 1.68, **score ≈ 594**.

---

## 0. Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (manages Python + dependencies). `uv sync` installs everything.
- Python 3.11+ (uv installs it automatically if missing).
- **Windows note:** set `PYTHONUTF8=1` before running any script, or the Wikipedia fetch/print will
  crash on the Devanagari/Telugu/Bengali text (Windows consoles default to cp1252, not UTF-8).
  - PowerShell: `$env:PYTHONUTF8=1`
  - Git Bash: prefix commands with `PYTHONUTF8=1 ...`, or `export PYTHONUTF8=1` once.
- Optional: **Node.js** — only needed for the JS↔Python parity *test* (`tests/test_parity.py`);
  it skips cleanly if Node is absent. Parity is also confirmed live by the widget.
- Optional: **Netlify CLI** (`npm install -g netlify-cli`) — only for CLI deploys. Drag-and-drop
  needs no install.

```bash
uv sync            # create .venv and install pinned deps from pyproject.toml / uv.lock
```

---

## 1. Build

`build.sh` runs the whole pipeline (from the repo root, in Git Bash):

```bash
PYTHONUTF8=1 ./build.sh
# uv sync → fetch_corpus → train_balance → export_artifacts → score → build_widget
```

Or run the steps by hand (PowerShell shown; only step 1 needs the network):

```powershell
$env:PYTHONUTF8=1
uv run python src/fetch_corpus.py       # 1. download + pin the 4 Wikipedia articles
uv run python src/train_balance.py      # 2. train the char-level BPE, English-biased
uv run python src/export_artifacts.py   # 3. write vocab.json, merges.txt, tokens.csv, manifest
uv run python src/score.py              # 4. print the per-language ratios + score
uv run python src/build_widget.py       # 5. assemble the deployable dist/ bundle
```

### What each script does

| Script | Input | Output | What it does |
|---|---|---|---|
| `fetch_corpus.py` | Wikipedia API | `corpus/*.txt` + `corpus/manifest.json` | Resolves the current revision id (`oldid`) of each India article, pulls plain text, preprocesses, and records oldids + sha256 hashes so the corpus is pinned and reproducible. **Only step needing network.** |
| `preprocess.py` | — (imported) | — | Shared cleaning: NFC normalization, whitespace cleanup, `word_count`, and `normalize_spaces` (one `▁word` per word). Used identically by training, scoring, and the widget. |
| `train_balance.py` | `corpus/*.txt` | `dist/tokenizer/tokenizer.json`, `balance_meta.json` | Trains one joint **char-level `BPE(byte_fallback=True)`** with a `Metaspace` pre-tokenizer. Sweeps the English replication weight until `X_en ≤ 1.2`, keeping the smallest-spread config. Never disqualifies. |
| `export_artifacts.py` | `tokenizer.json` | `vocab.json`, `merges.txt`, `tokens.csv`, `dist/tokenizer/manifest.json` | Asserts vocab == 10,000; writes the human-readable token list; recomputes per-language `tokens/words/X`, spread, score into the manifest. |
| `score.py` | `tokenizer.json` + `corpus/` | prints JSON | **The canonical scorer — the grader runs this.** Prints per-language `tokens/words/X`, spread, score, and `english_gate_pass`. Always exits 0. |
| `build_widget.py` | `dist/tokenizer/`, `corpus/`, `widget/` | `dist/` | Copies the widget files, corpus, `_headers`, and manifest into `dist/` — a self-contained static site. |
| `build_union.py` | `corpus/*.txt` | `tokenizer.json` | Fallback path (not used by default): union of per-language char-level BPEs. |
| `curve_sweep.py` | `corpus/*.txt` | prints | Optional diagnostic: per-language fertility at several vocab sizes. |

> `corpus/` is fetched once and **committed** (pinned source of truth). `dist/` is **generated**
> every build — deploy it, don't commit it.

---

## 2. Verify (before deploying)

**Reproduce the score:**
```powershell
uv run python src/score.py
```
Expect `english_gate_pass: true` and `score ≈ 594`. This must match `dist/tokenizer/manifest.json`.

**Run the test suite:**
```powershell
uv run --with pytest pytest -q
```
Expect **5 passed, 1 skipped** (parity skips without Node). Tests check: the faithful round-trip
gate, vocab == 10,000, zero UNK round-trip, English is the minimum ratio + score defined, and
`score.py` == manifest to 4 dp.

**Preview the widget locally:**
```powershell
cd dist
uv run python -m http.server 8000
# open http://localhost:8000  — the page recomputes everything live in the browser
```
Confirm: the score card shows ~594, the badge reads "English X₁ ≤ 1.2 ✓", the Indic text renders,
and the download buttons work. The widget's live numbers must equal `score.py` to 4 dp (they do —
this is the JS↔Python parity guarantee).

---

## 3. Deploy (Netlify)

Deploy the **`dist/`** folder (not `widget/` — the page fetches `tokenizer/*` and `corpus/*`, which
only exist after a build). `netlify.toml` already sets `publish = "dist"`, and `dist/_headers`
serves the Indic text and tokenizer as UTF-8 with CORS.

**Drag-and-drop (no install):**
1. app.netlify.com → log in → **Add new site → Deploy manually**.
2. Drag the `dist/` folder onto the drop zone.
3. Copy the generated `https://<site>.netlify.app` URL.

**Or CLI:**
```bash
netlify deploy --dir=dist --prod
```

Vercel / Cloudflare Pages / GitHub Pages also work with the same `dist/` folder.

---

## 4. Submission

| Field | Link |
|---|---|
| **Widget** | `https://<site>.netlify.app` |
| **Tokenizer.json** | `https://<site>.netlify.app/tokenizer/tokenizer.json` |

**Pre-submit checklist (tick truthfully):**

- [ ] Widget URL opens in a **fresh incognito window** and shows live numbers (score ≈ 594, English ✓).
- [ ] Tokenizer.json URL opens in incognito → raw JSON, no login, **not a Google Drive link**.
- [ ] Indic text (Hindi/Telugu/Bengali) renders correctly, not as boxes.
- [ ] Both captions filled in.

---

## 5. Reproducibility

Wikipedia changes daily, so the corpus is **pinned by revision id** and the extracted text is
**shipped** in `corpus/`. Running `uv run python src/score.py` against the shipped bundle reproduces
the widget's numbers exactly. The `oldid`s, timestamps, library versions, and text hashes are in
`corpus/manifest.json` and `dist/tokenizer/manifest.json`, so a grader can align to the same input.
`uv.lock` pins exact dependency versions so re-runs are byte-identical.
