# India BPE Tokenizer — Design Doc

_Author: Aditeya Kayal · v3 · 2026-07-08 · status: design, about to build_

This is my working design doc for the multilingual BPE assignment. It's written the way I'd hand it to another engineer picking up the repo: what I'm building, the decisions I've already made and why, the parts that will bite us, and the exact steps to ship. If you're skimming, read §2 (the scoring game), §7 (how I balance the languages), and §11 (why the numbers are trustworthy) — those are the parts that actually matter.

**TL;DR** — One byte-level BPE tokenizer, 10k vocab, trained on the "India" Wikipedia article in `en / hi / te / bn`. Score is `1000 / (X_max - X_min)` where `X_i` = tokens-per-word for language `i`. So the whole game is: get all four fertilities under 1.2 *and dead-even with each other*. I do that with corpus-mixture weighting and a tiny control loop. Ship a static widget that recomputes everything in the browser so the grader can't catch me hand-waving numbers.

---

## 0. Submission — the two boxes I actually have to fill (1000 pts)

The submit form has exactly **two graded fields, 500 pts each**. Everything else in this doc exists to make these two links correct. Both must survive an **incognito-window test** (publicly accessible, not logged-in/private), and there's a checkbox confirming I did that test — so I *will* actually open both in a private window before submitting.

| Field | Points | What goes in it | Caption I'll use |
|---|---|---|---|
| **Widget Link** | 500 | The public Netlify URL of the widget | `Live BPE tokenizer widget — ratios, token stats & self-score (en/hi/te/bn), computed in-browser` |
| **Tokenizer.json** | 500 | A **direct-download** URL to `tokenizer.json` | `Direct download of the 10,000-token tokenizer.json (byte-level BPE)` |

**Hard rules from the form, and how I satisfy each:**

- **Tokenizer.json must NOT be a Google Drive link.** Drive links silently default to private / "request access," which the form explicitly warns against. **My fix: host `tokenizer.json` on the same Netlify site as a static file**, so its link is just `https://<site>.netlify.app/tokenizer/tokenizer.json` — a plain public direct download, no auth, no Drive. (GitHub raw, `raw.githubusercontent.com/.../tokenizer.json`, is an acceptable backup — also direct and public.)
- **Both links public.** Because it's all static on Netlify, there's no login wall. I verify in incognito before ticking the boxes.
- **The tokenizer link is a direct file, not a page.** Clicking it downloads/serves the raw JSON, not an HTML viewer. The widget's in-page "download" button points at this *same* URL, so the file the grader downloads is byte-identical to the one the widget scores.

**Pre-submit gate (I don't tick a checkbox until all pass):**

- [ ] Open Widget Link in a fresh incognito window → widget loads and shows live numbers.
- [ ] Open Tokenizer.json link in incognito → raw JSON downloads/renders, no login, not Drive.
- [ ] `tokenizer.json` at that URL is the exact file `score.py` uses (same sha256).
- [ ] Both captions filled in (form marks caption "required").
- [ ] Both "tested in incognito" checkboxes ticked truthfully.

> Note on points: these two fields are **1000 submission points** and are separate from the tokenizer *quality* score `S = 1000/Δ`. I need both — a great `Δ` is worthless if the links are private or the tokenizer.json isn't downloadable.

---

## 1. What we're actually shipping

Four things, in priority order:

1. **A static widget** that loads the tokenizer + the four corpora and computes the ratios, token stats, and self-score *live* in the browser. No baked-in numbers — if I hard-code a score, I've already lost, because the grader re-runs it.
2. **The tokenizer, viewable + downloadable** from the widget (`vocab.json`, `merges.txt`, a friendly `tokens.csv`, and the full `tokenizer.json`).
3. **A public URL** (Netlify; Vercel/CF Pages are fine too — it's a static folder).
4. **A repro bundle** — pinned corpora, the tokenizer, and a `score.py` the grader can run to get the exact same numbers.

Explicit non-goals: this isn't a production tokenizer, it's tuned to four specific articles on purpose. No model training downstream. No backend — everything is static so hosting is trivial and there's nothing to break.

---

## 2. The scoring game (read this twice)

Per language:

```
X_i = tokens_i / words_i          # tokens the encoder emits ÷ word count
```

Rules:

```
gate:   every X_i <= 1.2          # miss this on any language and you're out
Δ   =   max(X_i) - min(X_i)       # the spread across the 4 languages
score = 1000 / Δ
```

```mermaid
flowchart LR
    A[Encode each corpus] --> B["X_i = tokens_i / words_i<br/>(en, hi, te, bn)"]
    B --> C{All X_i ≤ 1.2?}
    C -- no --> D[DISQUALIFIED<br/>score = 0]
    C -- yes --> E[sort X_i]
    E --> F["Δ = X_max − X_min"]
    F --> G["S = 1000 / Δ"]
```

Here's the thing everyone misses on first read: **the score doesn't care how low your fertility is, only how *even* it is.** Absolute fertility only matters for clearing the 1.2 gate. After that, it's all about Δ.

So the objective is not "compress hard," it's "compress *equally* hard across four scripts." Two very different problems.

And Δ is brutal because it's in the denominator:

| Δ | score |
|---|---|
| 0.10 | 10,000 |
| 0.05 | 20,000 |
| 0.02 | 50,000 |
| 0.01 | 100,000 |

Shaving 0.01 off the spread doubles the score. That's why half this doc is about squeezing Δ *and* proving the squeeze is real and not reproducibility noise. A huge score I can't reproduce is worthless — the grader runs it themselves.

Target I'm aiming for: all four `X_i` parked around **1.00–1.10** (comfortable margin under the gate), with Δ in the low hundredths. I'll report everything to 4 decimals.

---

## 3. Interpreting the vague bits

The assignment says "*Total English tokens / Total English Vocab, say 5000 words*." That's underspecified, so I'm pinning decisions and surfacing them so the grader can correct me cheaply.

- **`X_i` = tokens ÷ total words (fertility).** This is the only reading where "≤ 1.2" is both meaningful and achievable — a decent BPE lands ~1.0–1.1 tokens/word. If they meant tokens ÷ *unique* words, the number blows past 1.2 for everyone, which tells me that's not it. I still put a **toggle in the widget** for the unique-words reading so it's one click away.
- **Word = whitespace-delimited unit**, counted with `len(re.findall(r"\S+", text))` after NFC. Script-agnostic; `hi/te/bn` all space-separate words, so this Just Works.
- **10,000 = one tokenizer's `vocab_size`, counted once.** Shared subwords (punctuation, digits, Latin bits) don't get double-counted. This is the harder, honest reading of "10k overall for all languages" — one tokenizer, not four.

Three assumptions I'm flagging (they're printed in the widget + README):

- **A1:** eval text = the full pinned article (not a "first 5000 words" slice). One config flag flips this if they want a slice.
- **A2:** one shared 10k tokenizer computes all four `X_i`. Not four separate tokenizers.
- **A3:** grader may use my shipped corpus or their own fetch — I ship the `oldid`s so both line up.

If any of these is wrong, it's a 5-minute change, not a redesign. See Appendix E for the locked assumptions and their alternatives.

---

## 4. Getting the data (and pinning it so it doesn't move)

The four articles:

| lang | title | url |
|---|---|---|
| en | India | en.wikipedia.org/wiki/India |
| hi | भारत | hi.wikipedia.org/wiki/भारत |
| te | భారత దేశం | te.wikipedia.org/wiki/భారత_దేశం |
| bn | ভারত | bn.wikipedia.org/wiki/ভারত |

Wikipedia changes every day, so an unpinned corpus = my numbers won't match the grader's. The fix:

1. Grab the current revision id (`oldid`) at build time and freeze it.
2. Pull **plain text from that exact `oldid`** via the Action API — *not* scraped HTML, which drags in nav chrome and shifts with skin changes:

```
GET https://en.wikipedia.org/w/api.php
    ?action=query&prop=extracts&explaintext=1&format=json&revids=<OLDID>
```

3. **Ship the extracted text files.** The widget and `score.py` read *those files*, so the bundle is the source of truth. Anyone running my bundle gets my numbers to the token.

I record `oldid`, timestamp, and the sha256 of the *post-preprocessing* text in `corpus/manifest.json` — so drift is detectable, not silent.

Gotcha: English is a much longer article than the others. Raw length skews joint BPE toward whoever has the most bytes, so I do **not** feed raw-proportional data — the mixture weights in §7 handle that.

---

## 5. Preprocessing (one function, used everywhere)

The #1 way to get train/score mismatch is to clean text differently in two places. So there's exactly **one** `preprocess()` and both training and scoring (and the JS widget) call it.

```python
import re, unicodedata

_WS      = re.compile(r"[^\S\n]+")       # runs of horizontal whitespace
_REFMARK = re.compile(r"\[\s*\d+\s*\]")  # [1], [ 23 ]
_MULTINL = re.compile(r"\n{3,}")

def preprocess(text: str) -> str:
    text = unicodedata.normalize("NFC", text)   # matra/ZWJ ordering -> stable
    text = _REFMARK.sub("", text)
    text = _WS.sub(" ", text)
    text = "\n".join(l.strip() for l in text.split("\n"))
    return _MULTINL.sub("\n\n", text).strip()

_WORD = re.compile(r"\S+")
def word_count(text): return len(_WORD.findall(text))
```

Decisions baked in: **NFC** (Indic scripts encode the same glyph multiple ways — normalize or your counts wobble), **no lowercasing** (would change English fertility and diverge from the grader's raw text), no stemming/translit/stopword nonsense. The JS port uses the identical regexes + `.normalize("NFC")`, and there's a test that diffs JS vs Python output on Indic fixtures (§10).

---

## 6. Why byte-level BPE (and not the alternatives)

BPE = start from a base alphabet, repeatedly merge the most frequent adjacent pair, record each merge. `vocab = base + merges (+ specials)`. Standard.

The real decision is the **base alphabet**:

| base | UNK risk | shares across scripts | port to JS | verdict |
|---|---|---|---|---|
| char-level | high — unseen glyph = UNK | bad, each script needs its chars | easy | ✗ |
| SentencePiece unigram | low w/ byte fallback | good | annoying to match exactly | backup |
| **byte-level BPE** | **zero** | **great — 256-byte base shared by all** | straightforward | ✓ |

Going byte-level because:

- **Zero UNK, guaranteed.** Every Unicode char is UTF-8 bytes, all 256 are in the base. The grader literally cannot feed input that blows up fertility with an out-of-vocab char. That removes a whole category of "gotcha" input.
- **256-byte base covers all four scripts** for 256 of my 10k budget. Cheap.
- **Merges serialize cleanly** and I can re-implement the encoder in ~150 lines of JS for the in-browser verifier.

Here's the whole reason Indic needs more merge budget than English — one Telugu word, before and after merges:

```mermaid
flowchart LR
    W["భారత<br/>(1 word)"] --> U["UTF-8 bytes<br/>9 bytes → 9 base tokens"]
    U --> M["apply learned merges<br/>(byte pairs → syllables → word)"]
    M --> T["1–2 tokens<br/>fertility ≈ 1.0"]
    M -.->|too few merges<br/>for this script| T2["6–9 tokens<br/>fertility ≫ 1.2 ✗"]
```

The balancing loop (§7) exists precisely to buy each script *enough* merges to land in the good box, not the bad one. One thing to keep honest about: byte-level turns each multi-byte Indic char into a *sequence* of byte-tokens before merges kick in. So Indic languages need more merges than English to claw fertility back down to ~1.0. That imbalance is exactly what §7 fixes. I'll set `add_prefix_space=True` and document it, since word-initial token identity affects counts, and the JS port replicates the same byte→visible-byte table and prefix rule.

Library: HuggingFace `tokenizers`, exact version pinned. It's deterministic given fixed input + vocab size, and its `tokenizer.json` loads in both Python and a WASM build I can run in the browser.

---

## 7. The interesting part: balancing the four languages

Budget math first:

- 256 byte tokens (shared, mandatory)
- 0–1 special tokens (keep it minimal)
- ~9,743 learned merges — this is what I spread across languages
- **= 10,000 exactly** (build asserts it; if dedup lands me at 9,999/10,001 I nudge the merge target by ±1 and rebuild — deterministic)

The tension: English hits ~1.0 fertility with few merges; Indic scripts need way more. Train a naive joint BPE on raw text and English gets over-served, Indic under-served, and Δ is ugly.

**My fix — corpus-mixture weighting.** Train *one* joint BPE, but replicate each language's corpus by an integer factor `r_i` so the harder languages pull more weight in merge selection. Integer replication (not random sampling) keeps it bit-reproducible.

```
train_corpus = corpus_en*r_en + corpus_hi*r_hi + corpus_te*r_te + corpus_bn*r_bn
tok = train_byte_level_bpe(train_corpus, vocab_size=10000)
```

Then a dead-simple control loop nudges the weights until the spread stops shrinking:

```python
r = [1, 1, 1, 1]                     # or seed from a quick curve sweep
best, step, stale = None, 2, 0
for _ in range(MAX_ITERS):           # ~20 is plenty
    tok = train_joint(r, vocab_size=10000)
    X   = {L: fertility(tok, corpus[L]) for L in LANGS}
    if all(x <= 1.2 for x in X.values()):
        Δ = max(X.values()) - min(X.values())
        if best is None or Δ < best.Δ:
            best, stale = (r[:], tok, X, Δ), 0
        else:
            stale += 1
    r[argmax(X)] += step             # give the worst language more pull
    if r[argmin(X)] > 1: r[argmin(X)] -= step
    if stale >= PATIENCE: step //= 2  # anneal
    if step == 0: break
export(best.tok)
```

```mermaid
flowchart TD
    S["seed weights r = (1,1,1,1)<br/>(or from curve sweep)"] --> T[train joint 10k BPE on<br/>weighted corpus mix]
    T --> M["measure X_en, X_hi, X_te, X_bn"]
    M --> G{all X_i ≤ 1.2<br/>and Δ improved?}
    G -- yes --> K[keep as best]
    G -- no --> N[ ]
    K --> U["r[worst] += step<br/>r[best] −= step"]
    N --> U
    U --> P{Δ stalled?}
    P -- yes --> A["step //= 2 (anneal)"]
    P -- no --> T
    A --> Q{step == 0?}
    Q -- no --> T
    Q -- yes --> E[export best tokenizer]
```

Only 3 free ratios (pin one language at 1), so the search is tiny and each retrain on four short articles is seconds-to-minutes. Optional accelerator: sweep standalone vocab sizes per language first to sketch each fertility curve `f_i(n)`, then seed `r` smartly (Indic > 1) instead of starting flat.

**Anti-overfit / honesty check (optional, not yet wired in):** the intended enhancement is to split each corpus 90/10 by paragraph, tune only on the 90%, and *report* fertility on the held-out 10%. If held-out Δ ≈ train Δ, the balance is real. (By assignment the eval text *is* the article, so train≈eval is expected and fine — the held-out number is a trust signal, not a second eval.) The current build does **not** compute this; it's a straightforward add to `train_balance.py` + `export_artifacts.py` if we want the extra signal.

**Fallback** if mixture weighting can't tighten Δ enough: train four standalone BPEs sized so each hits a common fertility, then union them into one 10k tokenizer (union the byte base, concatenate merges under a single global rank, dedup, trim/pad to 10k). More direct control, but rank-reconciliation makes encoding slightly less globally optimal. I keep whichever path gives the smaller *reproducible* Δ.

---

## 8. Build pipeline

Pinned env: managed with **`uv`** — `pyproject.toml` + `uv.lock` pin Python 3.11 and exact versions of `tokenizers`, `regex`, `requests`. The lockfile is what guarantees the grader (and future me) rebuild against identical deps. Run everything via `uv run ...`. No RNG anywhere — mixture is integer replication, BPE merge selection is deterministic given the pinned lib. `manifest.json` also records resolved versions + a preprocessing hash so env drift is visible.

Scripts (one command wires them together via `build.sh`):

```
fetch_corpus.py     # resolve oldids, hit extracts API -> corpus/*.txt + manifest
preprocess.py       # the shared function from §5 (imported everywhere)
curve_sweep.py      # optional: per-language fertility curves to seed r
train_balance.py    # §7 control loop -> dist/tokenizer/tokenizer.json
build_union.py      # fallback path
export_artifacts.py # vocab.json, merges.txt, tokens.csv; ASSERT size==10000
score.py            # canonical scorer (the grader runs this)
build_widget.py     # drop artifacts+corpus into dist/, stamp provenance
```

`build.sh`: `fetch → preprocess → (sweep) → train_balance → export → score → build_widget`. Only `fetch` needs network; everything else runs offline from the shipped files.

```mermaid
flowchart LR
    subgraph net [needs network]
        F["fetch_corpus.py<br/>pin oldids, extracts API"]
    end
    F --> C[("corpus/*.txt<br/>+ manifest")]
    C --> P["preprocess.py<br/>NFC + cleanup (shared)"]
    P --> SW["curve_sweep.py<br/>(optional seed)"]
    SW --> TB["train_balance.py<br/>control loop §7"]
    TB --> EX["export_artifacts.py<br/>assert vocab == 10000"]
    EX --> AR[("tokenizer.json, vocab.json,<br/>merges.txt, tokens.csv")]
    AR --> SC["score.py<br/>compute X_i, Δ, S"]
    AR --> BW["build_widget.py"]
    P --> SC
    BW --> D[("dist/ → Netlify")]
    SC --> D
```

---

## 8a. Environment with `uv` (exact commands)

The whole project is a `uv` app. `pyproject.toml` declares deps + the Python floor; `uv.lock` pins exact resolved versions; `.python-version` pins the interpreter. That trio is the reproducibility backbone — the grader gets the same bytes because they resolve the same lockfile.

```bash
# one-time: create the project (already done in this repo)
uv init --app --python 3.11
uv add "tokenizers>=0.20,<0.22" "regex>=2024.5.15" "requests>=2.32"
uv add --dev ruff

# day-to-day
uv sync                       # make .venv match pyproject + uv.lock
uv run python src/score.py    # run anything inside the locked env
uv run ./build.sh             # full pipeline
uv lock --upgrade             # only when intentionally bumping versions
```

`pyproject.toml` essentials:

```toml
[project]
name = "india-bpe-tokenizer"
requires-python = ">=3.11"
dependencies = ["tokenizers>=0.20,<0.22", "regex>=2024.5.15", "requests>=2.32"]

[dependency-groups]
dev = ["ruff>=0.6"]

[tool.uv]
package = false               # it's an app, not an installable library
```

Rules of thumb: **commit `uv.lock`** (it's the pin), never `pip install` into the venv by hand, and always invoke scripts through `uv run` so they use the locked interpreter + deps.

---

## 8b. Script-by-script logic

Each `src/*.py` is small and single-purpose. Here's exactly what each one does and the non-obvious decisions inside.

### `common.py`
Single source of truth for constants + paths: `LANGS = [en, hi, te, bn]`, `VOCAB_SIZE = 10000`, `FERTILITY_GATE = 1.2`, `BYTE_BASE = 256`, the `ARTICLES` map (`lang → (api_host, title)`), and all directory paths. Everything imports from here so there's no drift.

### `preprocess.py` (shared, imported everywhere)
`preprocess(text)`: NFC-normalize → strip `[1]`-style ref markers → collapse horizontal whitespace → strip per-line → tidy blank lines. **Idempotent** (running it twice changes nothing), so it's safe that `fetch` writes preprocessed text *and* `train`/`score` re-apply it. `word_count(text)` = `len(re.findall(r"\S+", text))` — the denominator of every `X_i`. The JS widget mirrors these two functions exactly (same regexes + `normalize("NFC")`); a test diffs them.

### `fetch_corpus.py` (step 1 — the scrape)
For each language: (1) `prop=revisions` to resolve the **current `oldid`** + timestamp — the revision we pin; (2) `prop=extracts&explaintext=1` to pull **plain text** (not HTML); (3) `preprocess()`; (4) write `corpus/<lang>.txt` and record `oldid`, timestamp, `sha256(text)`, char/word counts in `corpus/manifest.json`. Sends a descriptive `User-Agent` (Wikipedia requires it) and follows redirects (`redirects=1`) so title variants resolve. The `extracts` API returns the current revision, and we log which `oldid` that was — the shipped text file is the actual source of truth.

### `curve_sweep.py` (optional diagnostic)
Trains a standalone byte-level BPE per language at sizes `[500…8000]` and prints fertility at each. Shows how "expensive" each script is → used to seed the balancing weights. Produces no shipped artifact.

### `train_balance.py` (step 2 — primary path)
The core. Loads corpora; starts weights `{en:1, hi:1, te:1, bn:1}`. Each iteration: build the training stream by **replicating each language's paragraphs `weights[lang]` times** (integer replication = deterministic), train one joint `ByteLevelBPE(vocab_size=10000, initial_alphabet=ByteLevel.alphabet())` (the full-byte alphabet guarantees zero UNK), then measure the four fertilities. If the gate passes and the spread `Δ` is a new best, **save `tokenizer.json`** and record `balance_meta.json`. Then nudge: `weights[worst] += step`, ease `weights[best] -= step`; when `Δ` stalls for `PATIENCE` iters, halve `step` (anneal); stop when `step` hits 0 or `MAX_ITERS`. Raises if nothing ever passes the gate (signals corpus/vocab problem).

> Gotcha baked in: with a small corpus the trainer can stop below 10,000 merges (not enough distinct pairs). The real "India" articles are large enough to reach 10k; `export` warns loudly if not.

### `build_union.py` (step 2 — fallback)
Only if the joint path can't tighten `Δ`. Trains four standalone BPEs, reads each ordered `merges.txt`, **round-robin interleaves by rank** (rank-0 of every language, then rank-1, …) with dedup up to the `10000−256` merge budget, then rebuilds one BPE from the byte alphabet + merged ordered merges. Round-robin gives each language equal footing (good for balance) at the cost of globally-optimal frequency ordering. Keep whichever path gives the smaller *reproducible* `Δ`.

### `export_artifacts.py` (step 3)
Loads `tokenizer.json`; **asserts vocab == 10000** (warns otherwise); writes `vocab.json` + `merges.txt` (`tok.model.save`) and a friendly `tokens.csv` (`id, token, decoded-UTF8, script`). The `decoded` column inverts the byte-level visible-byte map back to real UTF-8 so Indic tokens are readable; `script` is inferred from Unicode ranges (Devanagari/Telugu/Bengali/Latin/shared). Then recomputes per-language tokens/words/`X`, `Δ`, and `score`, and merges them into `dist/tokenizer/manifest.json` along with resolved library versions.

### `score.py` (step 4 — the canonical scorer, grader runs this)
Minimal + dependency-light. Loads tokenizer + shipped corpus, applies the **same** `preprocess`, computes `tokens/words/X` per language, `Δ`, `score`; prints JSON; exits non-zero if any `X_i > 1.2`. Output must match `manifest.results` and the widget to 4 dp.

### `build_widget.py` (step 5)
Copies `corpus/*.txt` → `dist/corpus/`, the `widget/*` static files → `dist/`, and the results manifest → `dist/manifest.json`. After it runs, `dist/` is a self-contained static site ready for `netlify deploy --dir=dist --prod`, with the tokenizer already at `dist/tokenizer/`.

### `build.sh`
`uv sync` → `fetch_corpus` → `train_balance` → `export_artifacts` → `score` → `build_widget`, each via `uv run`. Only step 1 needs network. Commented-out lines enable the curve sweep and the union fallback.

---

## 9. Artifacts (what lands in `dist/`)

`dist/tokenizer/`:

- `tokenizer.json` — the real thing, loads in Python and WASM. Source of truth. `vocab_size == 10000` or the build dies.
- `vocab.json` — `{token: id}`, the "list of all tokens" for view/download.
- `merges.txt` — ordered merge rules, rank = line order.
- `tokens.csv` — `id,token,decoded,script` — human-readable, so the token viewer isn't gibberish. `script` = which script the token mostly serves (Latin/Devanagari/Telugu/Bengali/shared).
- `manifest.json` — oldids, hashes, lib versions, mixture weights, and the computed results block:

```json
"results": {
  "interpretation": "tokens_per_total_word",
  "per_language": { "en": {"tokens":0,"words":0,"X":0.0}, "hi":{}, "te":{}, "bn":{} },
  "X_max":0.0, "X_min":0.0, "spread":0.0, "score":0.0, "gate_pass":true
}
```

(An optional `held_out` block — §8.5 — is not emitted by the current build.)

`dist/corpus/{en,hi,te,bn}.txt` — the post-preprocessing text, so widget and grader score identical bytes.

---

## 10. `score.py` — the number that counts

This is the file the grader runs. It's deliberately boring and short (paths come from `common.py`; it reads the committed `corpus/` and the built `dist/tokenizer/`):

```python
import json
from tokenizers import Tokenizer
from preprocess import preprocess, word_count      # SAME module as training

LANGS = ["en","hi","te","bn"]
tok = Tokenizer.from_file("dist/tokenizer/tokenizer.json")

res = {}
for L in LANGS:
    text  = preprocess(open(f"corpus/{L}.txt", encoding="utf-8").read())
    words = word_count(text)
    toks  = len(tok.encode(text, add_special_tokens=False).ids)
    res[L] = {"tokens": toks, "words": words, "X": toks/words}

X = {L: res[L]["X"] for L in LANGS}
spread = max(X.values()) - min(X.values())
assert all(v <= 1.2 for v in X.values()), "GATE FAILED"
print(json.dumps({"per_language":res, "X_max":max(X.values()),
                  "X_min":min(X.values()), "spread":spread,
                  "score":1000/spread}, indent=2, ensure_ascii=False))
```

Its output must match `manifest.results` *and* the widget, to 4 dp. If they disagree, something's wrong and I fix it before shipping — that's what §11 enforces.

---

## 11. Making the numbers trustworthy (JS ↔ Python parity)

The widget has to *compute* the numbers in the browser, not display constants. So I need the browser and Python to agree exactly. Two options were considered:

- **Chosen: hand-rolled JS BPE** (`widget/bpe.js`, ~150 lines): UTF-8 → HF's visible-byte table, `add_prefix_space=True`, greedy-merge by `merges.txt` rank, map via `vocab.json`. Zero dependencies, trivial to host statically. **Verified to match Python `tokenizers` exactly** on mixed Hindi/Telugu/Bengali/English strings (`tests/test_parity.py`).
- **Alternative: WASM.** Load the official `tokenizers` WASM build and `Tokenizer.from_file(tokenizer.json)` — identical to Python by construction, but heavier to bundle. Kept as a fallback if the hand-rolled path ever drifts.

Verification gate before deploy — the parity test over sample corpora (run via `uv run pytest`):

```
for L in langs:
    assert js_tokens(corpus[L]) == py_tokens(corpus[L])   # exact
    assert js_words(corpus[L])  == py_words(corpus[L])
```

One token of mismatch on any language means don't deploy until it's fixed. That's what lets me claim, honestly, that what the widget shows is what the grader's Python will produce. Same idea for `preprocess()` — the JS port (`widget/preprocess.js`) uses the identical regexes + `normalize("NFC")`.

So three independent things compute the same numbers: the build's `manifest.json`, `score.py`, and the browser (`bpe.js`). Run `uv run pytest` to confirm they agree before deploying. That's the whole trust story.

```mermaid
flowchart TD
    TOK[("tokenizer.json<br/>+ corpus/*.txt")] --> B["build → manifest.json.results"]
    TOK --> S["score.py (grader runs this)"]
    TOK --> W["browser widget (bpe.js)"]
    B --> CMP{agree to 4 dp?}
    S --> CMP
    W --> CMP
    CMP -- no --> BLK["block deploy — fix mismatch"]
    CMP -- yes --> OK["deploy ✓ numbers are trustworthy"]
```

---

## 12. The widget

Static SPA, vanilla JS is fine, no backend. On load it fetches the tokenizer + corpora and computes everything. Components:

1. **Self-score card** — big `S = 1000/Δ`, plus Δ and a gate badge ("all X_i ≤ 1.2 ✓/✗"; if it fails, show DISQUALIFIED, don't fake a score).
2. **Per-language table** — lang · words · tokens · `X_i` (4 dp) · ≤1.2? · rank. Highlight the max and min rows and draw the spread bracket between them.
3. **Calculation panel** — the literal arithmetic per language and the final `1000/(X_max−X_min)`, so nothing's a black box.
4. **Interpretation toggle** — total-words (default) vs unique-words denominator, recomputes live; shows assumptions A1–A3.
5. **Token viewer** — searchable table of the 10k tokens (id · token · decoded), search by substring. Loads Indic fonts so tokens render.
6. **Downloads** — buttons for `tokenizer.json`, `vocab.json`, `merges.txt`, `tokens.csv`, `manifest.json`.
7. **Provenance footer** — oldids, vocab size, lib versions (from `manifest.json`).

_(Optional/future: a held-out robustness row per §8.5, once the build emits `held_out`.)_

Perf: tokenize the four corpora once, cache in memory. Works offline after first load. Implemented in `widget/index.html` + `app.js` + `bpe.js` + `preprocess.js` + `styles.css`.

Data flow: `load → preprocess_js(corpus) → tokens=encode_js(corpus).length, words=word_count_js(corpus) → X, sort, Δ, S → render`.

---

## 13. Ship it (Netlify)

1. `build.sh` produces `dist/` (index.html + JS/CSS + `tokenizer/` + `corpus/` + manifest).
2. **Drag-drop:** Netlify → Add new site → Deploy manually → drop `dist/` → grab `https://<name>.netlify.app`.
   **Or CLI:** `netlify deploy --dir=dist --prod`.
3. Add a `_headers` file so `.json`/`.txt` serve as UTF-8 (Indic text safety).
4. Smoke-test the live URL **in incognito**: numbers render (not blank), downloads work, Telugu/Bengali glyphs display.
5. Confirm the **direct tokenizer link** works in incognito: `https://<site>.netlify.app/tokenizer/tokenizer.json` serves raw JSON, no auth. This is the exact URL that goes in the form's Tokenizer.json field and that the widget's download button links to.
6. Vercel / GH Pages / Cloudflare Pages all work with the same folder if I want a mirror/backup.

Submission = **Widget Link** (the site URL) + **Tokenizer.json** (the direct file URL above), both public, both tested in incognito (§0).

---

## 14. Tests (`uv run pytest` — run before deploy)

Implemented in `tests/`:

| test file | checks |
|---|---|
| `test_parity.py` | JS (`bpe.js`) token counts == Python `tokenizers`, exact. Self-contained (trains a tiny tokenizer). ✅ passing |
| `test_vocab_size.py` | tokenizer vocab == 10000 |
| `test_no_unk.py` | encode→decode round-trips all four corpora, zero UNK / no replacement chars |
| `test_gate_and_manifest.py` | every `X_i ≤ 1.2`, and `score.py` numbers == `manifest.results` (4 dp) |

Build-dependent tests skip cleanly (`run ./build.sh first`) until artifacts exist; the parity test runs anytime Node is present. Green suite → deploy; failures → fix first. This backs the trust claims in §11. (No hosted CI is wired up yet — run the suite locally; adding a GitHub Actions workflow is a straightforward future step.)

---

## 15. Risks (and what I did about them)

| risk | mitigation |
|---|---|
| grader uses a different article revision | pin + ship corpus & oldids; bundle is source of truth |
| ambiguous denominator (total vs unique words) | lock fertility reading; widget toggle; flag to grader |
| Δ tuned to noise, not real | held-out check; integer-replication determinism; report 4 dp |
| UNK inflates fertility | byte-level base → zero UNK, tested |
| Indic normalization mismatch | NFC everywhere; shared preprocess; edge-case fixtures |
| JS ↔ Python count drift | parity harness blocks deploy |
| vocab ≠ 10000 after dedup | build assertion + ±1 merge-target nudge |
| a language grazes the 1.2 gate | aim for X_i ≈ 1.0–1.1, keep margin |

---

## 16. Plan / milestones

| # | milestone | done when |
|---|---|---|
| M1 | fetch + pin corpus | 4 articles extracted, oldids + hashes recorded |
| M2 | shared preprocess + counters (+ JS port) | word counts stable, JS/Py parity on fixtures |
| M3 | fertility curve sweep | feasible common fertility found, `r` seeded |
| M4 | balanced 10k training | all X_i ≤ 1.2, Δ minimized by the loop |
| M5 | union fallback | only if M4's Δ is unsatisfactory; keep the better one |
| M6 | export artifacts | files valid, size==10000 asserted |
| M7 | score.py | output == manifest (4 dp) |
| M8 | widget + parity | browser counts == score.py, live compute works |
| M9 | deploy | URL live, downloads + Indic rendering verified |
| M10 | repro pass | fresh `python score.py` reproduces widget numbers |

---

## 17. Done = all of these

- [ ] vocab exactly 10,000 (asserted)
- [ ] zero UNK on all four corpora
- [ ] every `X_i ≤ 1.2` with margin
- [ ] Δ minimized *and* reproducible (held-out check optional, §8.5)
- [ ] widget computes live, shows the arithmetic
- [ ] full 10k-token list viewable + downloadable
- [ ] public URL live
- [ ] `uv run python src/score.py` reproduces widget numbers to 4 dp
- [ ] `uv run pytest` green (parity, vocab_size, no_unk, gate, score-matches-manifest)
- [ ] README covers interpretation (A1–A3), oldids, versions, re-run steps

---

## Appendix A — worked example

Say I measure `X_en=1.02, X_hi=1.06, X_te=1.09, X_bn=1.05`. Gate ✓. min=1.02 (en), max=1.09 (te), Δ=0.07 → `S≈14,286`. Tighten toward a common ~1.05 (Δ=0.02) → `S=50,000`. Same four articles, 3.5× the score, purely from evening out the spread. That's the whole point of §7.

## Appendix B — Indic edge cases the tests must cover

- **hi (Devanagari):** virama conjuncts (क्ष, त्र, ज्ञ), independent vs dependent vowels (matras), nukta, ZWJ/ZWNJ. Confirm NFC fixes matra ordering.
- **te (Telugu):** vowel signs, virama, two-part vowel signs, gemination — clusters must survive the byte-merge round-trip.
- **bn (Bengali):** ya-phala, ra-phala, khanda ta (ৎ), hasant, ZWNJ conjuncts.
- **mixed:** Latin acronyms/digits inside Indic text ("GDP", "1947") → should hit shared tokens.
- **whitespace:** NBSP, tabs, multi-newline collapse identically in JS and Python.

## Appendix C — repo layout

```
india-bpe-tokenizer/
├── readme.md  BPE_Tokenizer_Design_Spec.md  DEPLOYMENT.md
├── build.sh  pyproject.toml  uv.lock  .python-version  netlify.toml  .gitignore
├── corpus/            en.txt hi.txt te.txt bn.txt  manifest.json   (committed, pinned)
├── src/               common.py preprocess.py fetch_corpus.py curve_sweep.py
│                      train_balance.py build_union.py export_artifacts.py
│                      score.py build_widget.py
├── widget/            index.html app.js bpe.js preprocess.js styles.css
├── tests/             conftest.py _helpers.py test_parity.py test_vocab_size.py
│                      test_no_unk.py test_gate_and_manifest.py
└── dist/              generated bundle (widget + tokenizer/ + corpus/ + _headers)  [gitignored]
```

## Appendix D — glossary

- **BPE** — merge the most frequent adjacent pair, repeat.
- **byte-level BPE** — BPE over UTF-8 bytes (256 base) → zero UNK.
- **fertility** — tokens ÷ words; lower = tighter. This is `X_i`.
- **Δ (spread)** — `X_max − X_min`; the only thing the score cares about.
- **NFC** — Unicode canonical composition; makes Indic counts stable.
- **`r_i`** — integer replication factor for language `i` during joint training.

## Appendix E — assumptions & clarifications

These are ambiguities in the assignment wording that materially affect the score. I've locked a defensible default for each (see §3, A1–A3) and noted the alternative, so the design is deliberate rather than blind to the ambiguity. If I can reach the grader, these are the points I'd confirm.

1. Denominator: **total** words (my default) or **unique** words?
2. Eval text: **full article** (my default) or a fixed slice (first 5000 words)?
3. Score against **my shipped corpus** or **your own fetch**? (I ship oldids either way.)
4. **One** shared 10k tokenizer for all langs (my read), or four tokenizers summing to 10k?
5. Do special tokens count inside the 10k? (I assume yes, keep them at 0–1.)
