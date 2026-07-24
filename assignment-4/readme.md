# Session 4 — Data Cleaning Pipeline, applied to AI4Bharat Sangraha

Session 4's 8-stage cleaning pipeline (Extract → Normalize → Language ID → Quality filter →
Deduplicate → PII scrub → Decontaminate → Manifest), run **for real** against a real
10–100M row dataset: **[AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)**
(verified split, CC-BY-4.0) — the exact Indic corpus the lesson's own "V4 reality" callouts
kept citing as the pipeline that skipped every one of these stages.

- **Live widget:** `<DEPLOY_URL>` (deployed on Vercel — `vercel deploy --prod dist`, or drag `dist/` into the Vercel dashboard)
- Every number in the widget comes from an actual pipeline run over a real, live-fetched
  sample — nothing is hand-typed. Reproduce with `uv run python src/fetch_sample.py && uv run python src/pipeline.py`.

## Assignment answers, at a glance

| Question | Answer |
|---|---|
| How many strategies (stages)? | **8** — Extract, Normalize, Language ID, Quality filter, Deduplicate, PII scrub, Decontaminate, Manifest |
| What dataset was picked? | AI4Bharat Sangraha, `verified` split — **15,100,662 real rows**, 33.9 GB, CC-BY-4.0 |
| What was cleaned & why? | See widget tab 3 — real per-stage counts from a 3,600-row live sample across 12 languages |
| Any other concern cleaned up? | **2 real bugs found and fixed while building this pipeline** (see below) |
| Final statistics | 3,600 → 3,248 rows survived (90.2%), 15 mislabeled docs caught, 5 duplicates removed, 41 PII hits redacted |

## The two real bugs this run found

1. **Indic combining marks miscounted as "symbols."** `str.isalnum()` returns `False` for
   Unicode combining marks (category `M*`) — exactly how Indic vowel signs (mātrās) and
   viramas are encoded. Every native-script document (2,400/2,400) was scoring as
   symbol-heavy garbage and getting dropped. Fixed by classifying on Unicode category
   instead of `isalnum()`. Survival: **9% → 85%**.
2. **Urdu's real sentence-ender (`۔` U+06D4) was missing** from the "lines end in
   punctuation" heuristic, which only covered `. ! ? । ॥`. 257/300 Urdu docs were dropped
   even though 187 lines in that batch genuinely ended in `۔`. Fixed by adding it.
   Urdu survival: **34/300 → 202/300**.

Both are the same species of English-centric-heuristic bug Session 4 spends most of its
time warning about — found here because the pipeline ran on genuinely multilingual data
instead of assuming it away.

## Repo layout

```
session4-data-cleaning-pipeline/
├── readme.md
├── pyproject.toml / uv.lock       # uv-managed deps: requests, datasketch, langdetect
├── src/
│   ├── fetch_sample.py            # pulls a real live sample via HF datasets-server REST API
│   └── pipeline.py                # the 8-stage pipeline; writes data/stats.json
├── data/
│   ├── raw_sample.jsonl           # 3,600 real rows, 12 languages (generated)
│   ├── dataset_stats.json         # real whole-dataset size info from HF /size API
│   ├── stats.json                 # per-stage results the widget reads (generated)
│   └── manifests_sample.json      # sample of emitted provenance manifests (generated)
├── widget/                        # static SPA source (index.html / app.js / styles.css)
└── dist/                          # deployable bundle — widget + data/*.json
```

## Reproduce

```bash
uv sync
uv run python src/fetch_sample.py   # needs network — hits datasets-server.huggingface.co
uv run python src/pipeline.py       # offline — writes data/stats.json
```

Then re-copy `data/*.json` into `dist/data/` and reload the widget.

## Honesty notes

- **Sample, not the full 15.1M rows.** Processing the whole verified split needs the
  class's centrally-owned, large-memory, checkpoint-resumable dedup machine (Session 4,
  §7) — not a laptop. This run used a real, randomly-ordered 3,600-row sample (300 per
  language × 12 languages), pulled live via HuggingFace's official REST API, and ran the
  complete 8-stage pipeline against it for real.
- **Token counts are a whitespace-split proxy**, not a real BPE count — deliberately not
  repeating the words×1.3 estimate the lesson itself flags as wrong for Indic scripts.
- **The PII name-detection layer is an illustrative Latin-script heuristic only** (regex
  title-case bigrams), exactly as caveated in the lesson — it does not attempt Indic NER.

## License / attribution

Data: AI4Bharat Sangraha, CC-BY-4.0. Citation: Khan et al., *IndicLLMSuite: A Blueprint for
Creating Pre-training and Fine-Tuning Datasets for Indian Languages*, arXiv:2403.06350.
