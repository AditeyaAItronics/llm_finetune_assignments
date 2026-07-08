# Deployment Guide

How to build the site and publish it so the two submission links work. Read the one-line rule first: **you build `dist/`, and you deploy `dist/` — not `widget/`.**

---

## 0. Prerequisites

- [`uv`](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js (only needed if you want to run the JS↔Python parity test)
- A free [Netlify](https://netlify.com) account
- Network access (the build fetches the four Wikipedia articles once)

No GPU, no Colab — BPE training is CPU-only and finishes in seconds-to-minutes.

---

## 1. Build the bundle

From the repo root:

```bash
./build.sh
```

This runs `uv sync → fetch_corpus → train_balance → export_artifacts → score → build_widget` and produces the deployable folder **`dist/`**:

```
dist/
├── index.html  app.js  bpe.js  preprocess.js  styles.css   # the widget
├── tokenizer/  tokenizer.json  vocab.json  merges.txt  tokens.csv
├── corpus/     en.txt  hi.txt  te.txt  bn.txt
├── manifest.json
└── _headers                                                # charset + CORS
```

If the run prints `WARNING: vocab size ... != 10000`, the corpus was too small to learn 10k merges — re-check that all four articles fetched. (The real "India" articles are large enough.)

---

## 2. Verify before you deploy

```bash
# 1) the canonical score (this is what the grader runs)
uv run python src/score.py

# 2) the full test suite (vocab==10000, no UNK, gate, parity, score==manifest)
uv run pytest

# 3) preview the actual widget locally
cd dist && uv run python -m http.server 8000
#   → open http://localhost:8000  (numbers should render; downloads should work)
```

All of `score.py`, `pytest`, and the widget must agree on the numbers. If they don't, stop and fix — do not deploy numbers you can't reproduce.

---

## 3. Deploy to Netlify

### Option A — CLI (recommended)

```bash
npm install -g netlify-cli      # one-time
netlify login                   # opens browser
netlify deploy --prod           # publish dir is dist/ (from netlify.toml)
```

If it doesn't pick up `dist/` automatically, be explicit:

```bash
netlify deploy --dir=dist --prod
```

Copy the **Website URL** it prints (e.g. `https://your-site.netlify.app`).

### Option B — Drag & drop (no CLI)

1. Go to the Netlify dashboard → **Add new site → Deploy manually**.
2. Drag the **`dist/`** folder (not `widget/`, not the repo root) onto the drop zone.
3. Netlify gives you a URL.

Both options apply the correct headers (`netlify.toml` for CLI, `dist/_headers` for drag-drop).

---

## 4. Get your two submission links

After deploy, with base URL `https://<site>.netlify.app`:

| Submission field | Link |
|---|---|
| **Widget Link** | `https://<site>.netlify.app` |
| **Tokenizer.json** | `https://<site>.netlify.app/tokenizer/tokenizer.json` |

---

## 5. Verify in incognito (required by the form)

The form has a checkbox: *"I tested this link in an incognito window — it's publicly accessible."* Do it for real:

1. Open a **private / incognito** window (so you're not logged into anything).
2. Paste the **Widget Link** → the page loads and shows live numbers (not blank, no error banner).
3. Paste the **Tokenizer.json** link → raw JSON downloads or renders. It must **not** ask for login and must **not** be a Google Drive link.

Only tick the checkboxes once both pass.

**Pre-submit checklist:**

- [ ] Widget loads in incognito and shows the score + per-language table
- [ ] `tokenizer.json` link downloads raw JSON in incognito (no auth, not Drive)
- [ ] The downloaded `tokenizer.json` is the same file the widget uses (same site)
- [ ] Both captions filled in on the form
- [ ] Both "tested in incognito" boxes ticked

---

## 6. Redeploy after changes

Rebuild, then deploy again — the URL stays the same on a linked site:

```bash
./build.sh && netlify deploy --prod
```

For drag-drop, just drag the new `dist/` again onto the same site (Site → Deploys → drag to redeploy).

---

## 7. Alternative hosts (same `dist/` folder)

- **Vercel:** `npm i -g vercel && vercel deploy --prod dist`
- **Cloudflare Pages:** `npx wrangler pages deploy dist`
- **GitHub Pages:** push `dist/` contents to a `gh-pages` branch

All work because `dist/` is fully static. The tokenizer link becomes `<host>/tokenizer/tokenizer.json` in every case.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Widget loads but shows an error banner | `tokenizer/` or `corpus/` missing — you deployed `widget/` instead of `dist/`. Deploy `dist/`. |
| Numbers are blank | Files didn't load; check the browser console. Make sure you served/deployed the whole `dist/` folder. |
| Indic text shows as boxes | Font issue on the viewer's machine; content is still correct. The page requests Noto fonts and falls back to system fonts. |
| `tokenizer.json` link opens a page, not a download | You linked a viewer page, not the file. Use `.../tokenizer/tokenizer.json` exactly. |
| `WARNING: vocab size != 10000` at build | Corpus too small / a fetch failed. Re-run `./build.sh` and confirm all four `corpus/*.txt` exist and are non-trivial. |
| Netlify serves JSON as `text/plain` | Ensure `netlify.toml` (CLI) or `dist/_headers` (drag-drop) is present; both set the content type. |
