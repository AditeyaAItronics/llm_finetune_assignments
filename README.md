# Neural Proofs

A single-page, from-scratch clone of the "neural proof" site: four in-browser
deep-learning experiments (activations, depth, embeddings, generalization).
Pure vanilla JS + Canvas 2D, with Chart.js (CDN) for one bar chart. No build
step, no backend, no data leaves the browser.

## Run locally
Any static server works, e.g.:

```
cd neural-proof
python -m http.server 8080
# open http://localhost:8080
```

## Deploy to Netlify
- **Drag & drop:** zip or drag the `neural-proof` folder onto
  https://app.netlify.com/drop — done.
- **Git / CLI:** `netlify deploy --prod` from this folder (publish dir = `.`).

`netlify.toml` sets the publish directory and a couple of safe headers.
