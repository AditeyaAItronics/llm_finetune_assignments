# Architecture — Neural Proofs

A single-page, zero-build educational web app. Four tiny neural networks are
implemented **from scratch in JavaScript**, trained live in the browser, and
draw their own decision surfaces on `<canvas>`. The goal is to *prove* four
deep-learning claims empirically rather than assert them.

- **No framework, no bundler, no backend.** Everything is static.
- **All computation is client-side** — nothing is sent anywhere.
- Only external dependencies: **Chart.js** (one bar chart) and **Google Fonts**,
  both from CDNs. The neural nets use no ML library.

---

## 1. File layout

```
neural-proof/
├── index.html      # markup only + <head> (meta/SEO/favicon) + links to css/js
├── styles.css      # the full design system (CSS variables + component styles)
├── app.js          # all logic: from-scratch ML, canvas viz, experiments, wiring
├── og-image.svg    # 1200×630 social-share preview image
├── netlify.toml    # publish dir (".") + a couple of safe headers
├── README.md       # run + deploy instructions
└── architecture.md # this file
```

`.claude/launch.json` (one level up, in the repo root) defines the local
preview server (`python -m http.server 8080 --directory neural-proof`); it is a
dev convenience only and is not part of the deployed site.

The app is a **light 2-file split** — markup, styles, and logic in three files,
still with **no build step**:
1. `index.html` — `<head>` (meta/SEO/OpenGraph, inline-SVG favicon, font +
   Chart.js CDN links, `<link>` to `styles.css`), the `<body>` markup, and a
   final `<script src="app.js">`.
2. `styles.css` — the full design system (CSS custom properties + components).
3. `app.js` — a plain (non-module) script holding every function described
   below. It runs after the DOM via its `DOMContentLoaded` handler.

> This is deliberately *not* split into ES modules: it keeps the file count low
> and still works when opened directly, while getting `index.html` down to ~240
> lines. If it grows, the natural next step is native ES modules
> (`<script type="module">`) split along the section boundaries in §3–§4 — no
> bundler required.

---

## 2. Design system (CSS)

- **Theme:** light. White background, dark-neutral body text for legibility.
- **Palette:** a "traffic-light" scheme driven by three CSS variables —
  `--red (#dc2626)`, `--yellow (#e0a400)`, `--green (#16a34a)`. Legacy accent
  names (`--cyan`, `--violet`, …) are remapped onto these so older rules keep
  working.
- **Per-section accent:** `#s1`→red, `#s2`→yellow, `#s3`→green, `#s4`→red,
  applied to section tags and headings.
- **Fonts:** Space Grotesk (display/headings), Inter (body), JetBrains Mono
  (code, metrics, status).
- **Key components:** `.card`, `.viz` (canvas), `.metric`, `.legend`,
  `.controls`, `.status`, `.takeaway` (the reveal-on-run explainer), `.pbar`
  (training progress bar), `.numctl` (numeric input), `pre.code` (light syntax
  highlighting). Layout is CSS grid, collapsing to one column under 840px.

---

## 3. JavaScript core (the from-scratch ML)

All logic lives in `app.js`. Shared building blocks:

### RNG & math
- `makeRng(seed)` — mulberry32 seeded PRNG for **reproducible** results.
  `Date.now`/`Math.random` are avoided so a given seed always yields the same
  run.
- `gauss(rng)` — Box–Muller normal samples (weight init, data noise).
- Activations: `relu`, `drelu`, `sigmoid`; plus `softmax` (local to S3).
- `matmul(A,B)` — used for the S2 matrix-collapse proof.

### Neural network
A minimal MLP with manual forward + backprop:
- `makeNet(shape, rng, seed)` — builds layers `[{W, b, act, inp, out}]` with
  He-style init. `act ∈ 'relu' | 'linear' | 'sigmoid'`.
- `forward(net, x)` — returns per-layer pre-activations `zs` and activations
  `acts` (needed for backprop).
- `trainStep(net, x, y, lr)` — one SGD step on a single sample. Output layer is
  sigmoid + **binary cross-entropy** (`dL/dz = a − y`); hidden layers backprop
  with the correct activation derivative. Returns the sample loss.
- `predict(net, x)` and `accuracy(net, X, Y)` — inference helpers.

### Training loops
- `trainFull(net, X, Y, epochs, lr, rng)` — synchronous; used for the small,
  fast presets.
- `trainChunked(net, X, Y, epochs, lr, rng, onProgress)` — **Promise-based**,
  spreads epochs across `requestAnimationFrame` frames (targeting ~30k
  sample-steps/frame) so large datasets never freeze the page. Drives the S4
  custom-size progress bar. This is the mechanism behind suggestion #6.

### Canvas helpers
- `fitCanvas(cv)` — HiDPI-aware sizing + 2D context.
- `drawBoundary(cv, net, X, Y, R)` — evaluates the model on a pixel grid and
  paints the decision surface (red = class 0 → green = class 1), then overlays
  the data points. Used by S1 and S2.
- `pca2(M)` (S3) — 2-component PCA via power iteration on the covariance matrix,
  with deflation for the second component. Projects the learned embedding table
  to 2D.
- `buildSurface(net)` / `drawSurface()` (S4) — samples P(class 1) over a 26×26
  grid, then renders it as a hand-rolled **isometric 3-D surface** (yaw/pitch
  rotation + orthographic projection + painter's-algorithm depth sort). No WebGL.

---

## 4. The four experiments

Each has a `runSN()` entry point that builds seeded data, trains via an
`requestAnimationFrame` chunk loop (animating the boundary as it learns),
updates the accuracy metrics, and on completion reveals a **`.takeaway`**
explainer (`setTakeaway(id, html)` — suggestion #5).

| # | Section | What it trains | What it proves |
|---|---------|----------------|----------------|
| S1 | Activations | linear+sigmoid vs 1 ReLU layer on two rings | nonlinearity is required to bend the boundary (~55% vs ~99%) |
| S2 | Depth | 1-layer, 5 linear layers, 5 layers+ReLU | stacked linear layers collapse to one matrix (shown numerically); ReLU breaks the tie |
| S3 | Embeddings | embedding→softmax next-token model on a toy grammar | category clusters emerge from next-token prediction alone (PCA plot + nearest neighbours) |
| S4 | Generalization | overparameterized MLP at n = 20/200/2000, **plus any user-entered size** | the train/held-out gap collapses as data grows |

**S4 custom size** (`trainCustomSurface`): reads the numeric input (clamped
2–8000), trains one model with `trainChunked` while showing a progress bar,
then renders that model's 3-D surface and a metric readout (dataset size, train
acc, held-out acc, colour-coded generalization gap) via `renderSurfaceInfo(n)`.
Epoch count scales with `epochsFor(n)` so small sets overfit and large sets stay
fast.

---

## 5. Wiring & lifecycle

On `DOMContentLoaded`:
- `heroAnim()` starts the hero particle-field animation.
- `bindSurfaceControls()` attaches drag-to-rotate / scroll-to-zoom to the S4 3-D
  canvas.
- Each experiment's **Run/Train button** is wired to its `runSN`. Canvases start
  **empty** — nothing renders until the user presses a button (there is no
  auto-run on scroll).
- **"▶ Run all experiments"** in the hero (`#run-all`) fires all four at once
  (suggestion #8).
- The S4 numeric input is wired to `trainCustomSurface` on click and on Enter.

Rendering is entirely pull-based: buttons → train (rAF chunks) → draw to canvas
→ update DOM metrics/takeaways. No shared reactive state beyond the small
`s4State` object (`{done, surfaces, metrics, cur}`).

---

## 6. SEO / social / favicon (suggestion #7)

- Inline-SVG **favicon** (three red/yellow/green dots) via a `data:` URI — works
  offline and on any host.
- **OpenGraph + Twitter** meta tags in `<head>`, pointing at `og-image.svg`.
- `theme-color` meta.

> **After deploying:** change `og:image` (and ideally add `og:url`) to your
> **absolute** Netlify URL, e.g. `https://your-site.netlify.app/og-image.svg`.
> Some social scrapers require absolute URLs and prefer PNG/JPG over SVG — if a
> platform won't render the SVG preview, export `og-image.svg` to a 1200×630
> PNG and point the tags at that.

---

## 7. Running & deploying

**Local preview** (any static server):
```
cd neural-proof
python -m http.server 8080     # → http://localhost:8080
```

**Deploy to Netlify:**
- Drag the `neural-proof` folder onto https://app.netlify.com/drop, **or**
- `netlify deploy --prod` from this folder (publish dir = `.`, per `netlify.toml`).

No build step runs; the files are served exactly as-is.

---

## 8. Design constraints & rationale

- **Single file, no build:** keeps the app trivially portable, auditable, and
  deployable — matching the original it was modelled on.
- **From-scratch ML:** the point is to *show the mechanism*, so a library would
  hide the very thing being proven.
- **Seeded RNG everywhere:** reproducibility — the same page always produces the
  same proof.
- **Canvas 2D over WebGL:** the visuals (including the "3-D" surface) are simple
  enough that 2D keeps the code readable and dependency-free.

## 9. Extending it

- **New experiment:** add a section following the `#sN` markup pattern
  (card + canvas + controls + `.takeaway`), write a `runSN()` that trains and
  draws, and wire its button in `DOMContentLoaded`.
- **Colour-blind-safe mode (not yet built):** red/green is the hardest pair for
  ~8% of viewers. A future toggle could swap red→orange / green→blue, or add
  marker shapes (● vs ✕) so class isn't encoded by colour alone.
- **Touch support for the 3-D surface (not yet built):** `bindSurfaceControls`
  currently handles mouse only; adding `touchstart/move` + pinch would enable
  mobile rotation/zoom.
