# Building a 40B, India-First Foundation Model

Data and tokenizer strategy for a 40B-parameter model matched to Gemma-class quality, strong at coding, agentic work, and Indic languages, and "India first" in worldview.

- **Live page:** `<VERCEL_URL>`
- **Design doc (text + reasoning trail):** [`design.md`](./design.md)

The page answers all four parts of the brief — a "what the brief asked → where it's answered" map sits right under the header. Short version:

| Ask | Section |
|---|---|
| What data to collect (pretrain / post-train / RL) and why | S3, S5, S6 |
| How the data gets cleaned | S4 |
| How the model gets tested | S7 |
| Fertility per language + code/science/math/agentic → tokenizer vocab size | S1, S2 |

## Headline numbers

- **40B** params, **15T** pretraining tokens (375 tok/param, 18.75× Chinchilla-optimal — anchored to Gemma-3-27B's 14T and LLaMA-3-70B's 15T)
- **256,000**-token vocabulary — 110K of it dedicated to 22 scheduled Indic languages
- Fertility targets: **1.2** (English) → **1.4–2.1** (Indic languages, capped at what Sarvam-1/Krutrim-1 have actually measured, not overpromised) → **1.0 token/digit** (math) → **6 reserved tokens** (agentic tool-calling, Gemma 4's real scheme)
- Pretraining mix re-derived from LLaMA-3's disclosed 50/25/17/8 split, re-weighted to 38% general / 20% code / 12% math / 20% Indic / 5% agentic / 5% other languages

Every number is cited against a real technical report — see the citation strips on the page or the reference list in `design.md`.

## Repo layout

```
40b-parameter-model/
├── index.html     # the report (static, no build step)
├── styles.css      # design system: dark/light tokens, charts, tables, cards
├── design.md       # full text answers + the reasoning/process trail
├── vercel.json     # security headers for static hosting
└── readme.md       # this file
```

No build step — `index.html` and `styles.css` are plain static files.

## Deploy (Vercel)

1. [vercel.com/new](https://vercel.com/new) → import `AditeyaAItronics/llm_finetune_assignments`.
2. Set **Root Directory** to `40b-parameter-model`.
3. Framework preset: **Other** (no build command, no output directory needed — it's static).
4. Deploy. Vercel serves `index.html` directly.

Or via CLI, from inside this folder:

```bash
npm i -g vercel
vercel --prod
```

## Preview locally

```bash
cd 40b-parameter-model
python -m http.server 8000
# open http://localhost:8000
```
