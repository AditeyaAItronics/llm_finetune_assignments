// Orchestrates the widget: load tokenizer + corpora, tokenize live, render everything.
(function () {
  var LANGS = ["en", "hi", "te", "bn"];
  var NAMES = { en: "English", hi: "Hindi", te: "Telugu", bn: "Bengali" };
  var GATE = 1.2;
  var state = { tok: null, corpora: {}, mode: "total", vocab: null };

  function $(id) { return document.getElementById(id); }
  function fmt(x, d) { return (x === null || x === undefined) ? "—" : Number(x).toFixed(d === undefined ? 4 : d); }

  async function fetchText(url) {
    var r = await fetch(url);
    if (!r.ok) throw new Error("could not load " + url + " (" + r.status + ")");
    return r.text();
  }
  async function fetchJSON(url) { return JSON.parse(await fetchText(url)); }

  async function load() {
    // tokenizer model (vocab + merges)
    state.vocab = await fetchJSON("tokenizer/vocab.json");
    var merges = (await fetchText("tokenizer/merges.txt")).split("\n");
    state.tok = BPE.build(state.vocab, merges);
    // corpora
    for (var i = 0; i < LANGS.length; i++) {
      var L = LANGS[i];
      state.corpora[L] = Preprocess.preprocess(await fetchText("corpus/" + L + ".txt"));
    }
    // optional manifest (provenance)
    try { state.manifest = await fetchJSON("manifest.json"); } catch (e) { state.manifest = null; }
  }

  function uniqueWordCount(text) {
    var m = text.match(/\S+/g) || [];
    var s = new Set(m);
    return s.size;
  }

  function compute() {
    var rows = {};
    for (var i = 0; i < LANGS.length; i++) {
      var L = LANGS[i];
      var text = state.corpora[L];
      var tokens = BPE.countTokens(text, state.tok);
      var words = state.mode === "total" ? Preprocess.wordCount(text) : uniqueWordCount(text);
      rows[L] = { tokens: tokens, words: words, X: tokens / words };
    }
    var xs = LANGS.map(function (L) { return rows[L].X; });
    var xmax = Math.max.apply(null, xs), xmin = Math.min.apply(null, xs);
    var spread = xmax - xmin;
    var gate = xs.every(function (x) { return x <= GATE; });
    return {
      rows: rows, xmax: xmax, xmin: xmin, spread: spread,
      score: spread > 1e-9 ? 1000 / spread : Infinity, gate: gate
    };
  }

  function render() {
    var r = compute();
    var maxL = LANGS.reduce(function (a, b) { return r.rows[a].X >= r.rows[b].X ? a : b; });
    var minL = LANGS.reduce(function (a, b) { return r.rows[a].X <= r.rows[b].X ? a : b; });

    // score card
    $("score").textContent = r.gate ? (r.score === Infinity ? "∞" : Math.round(r.score).toLocaleString()) : "—";
    $("spread").textContent = fmt(r.spread);
    var badge = $("gate");
    badge.textContent = r.gate ? "All X ≤ 1.2 ✓" : "GATE FAILED ✗";
    badge.className = "badge " + (r.gate ? "ok" : "bad");

    // table
    var order = LANGS.slice().sort(function (a, b) { return r.rows[b].X - r.rows[a].X; });
    var html = "<tr><th>Language</th><th>Words</th><th>Tokens</th><th>X = tok/word</th><th>≤1.2</th><th>Rank</th></tr>";
    order.forEach(function (L, idx) {
      var cls = L === maxL ? "max" : (L === minL ? "min" : "");
      var ok = r.rows[L].X <= GATE;
      html += "<tr class='" + cls + "'><td>" + NAMES[L] + " <span class='tag'>(" + L + ")</span></td>" +
        "<td>" + r.rows[L].words.toLocaleString() + "</td>" +
        "<td>" + r.rows[L].tokens.toLocaleString() + "</td>" +
        "<td>" + fmt(r.rows[L].X) + "</td>" +
        "<td>" + (ok ? "✓" : "✗") + "</td>" +
        "<td>" + (idx + 1) + "</td></tr>";
    });
    $("results").innerHTML = html;

    // calc
    var c = "";
    LANGS.forEach(function (L) {
      c += NAMES[L] + ":  " + r.rows[L].tokens + " / " + r.rows[L].words + " = <b>" + fmt(r.rows[L].X) + "</b>\n";
    });
    c += "\nspread Δ = X_max − X_min = " + fmt(r.rows[maxL].X) + " − " + fmt(r.rows[minL].X) + " = <b>" + fmt(r.spread) + "</b>\n";
    c += "score  S = 1000 / Δ = <b>" + (r.gate ? (r.score === Infinity ? "∞" : fmt(r.score, 1)) : "DISQUALIFIED (gate)") + "</b>";
    $("calc").innerHTML = c;

    $("modeNote").textContent = state.mode === "total"
      ? "Denominator = total words (fertility). Primary interpretation."
      : "Denominator = unique words (type count). Secondary interpretation.";
  }

  function renderViewer(filter, tag) {
    var entries = Object.keys(state.vocab).map(function (t) { return [t, state.vocab[t]]; });
    entries.sort(function (a, b) { return a[1] - b[1]; });
    var rows = [];
    for (var i = 0; i < entries.length && rows.length < 500; i++) {
      var token = entries[i][0], id = entries[i][1];
      var dec = BPE.decodeToken(token);
      if (filter && dec.toLowerCase().indexOf(filter.toLowerCase()) === -1 &&
          token.toLowerCase().indexOf(filter.toLowerCase()) === -1) continue;
      rows.push("<tr><td>" + id + "</td><td>" + escapeHtml(token) + "</td><td>" + escapeHtml(dec) + "</td></tr>");
    }
    $("vocabCount").textContent = entries.length.toLocaleString();
    $("viewer").innerHTML = "<tr><th>ID</th><th>Token</th><th>Decoded</th></tr>" + rows.join("") +
      (rows.length >= 500 ? "<tr><td colspan=3 class='muted'>showing first 500 matches — refine search</td></tr>" : "");
  }

  function escapeHtml(s) {
    return s.replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; });
  }

  function renderProvenance() {
    if (!state.manifest) return;
    var m = state.manifest, parts = [];
    if (m.vocab_size) parts.push("vocab " + m.vocab_size.toLocaleString());
    if (m.articles) {
      LANGS.forEach(function (L) {
        if (m.articles[L]) parts.push(NAMES[L] + " oldid " + m.articles[L].oldid);
      });
    }
    if (m.library) parts.push("tokenizers " + m.library.tokenizers + " / py " + m.library.python);
    $("prov").textContent = parts.join(" · ");
  }

  function wire() {
    $("btnTotal").onclick = function () { state.mode = "total"; setActive(this); render(); };
    $("btnUnique").onclick = function () { state.mode = "unique"; setActive(this); render(); };
    $("search").oninput = function () { renderViewer(this.value.trim()); };
  }
  function setActive(btn) {
    $("btnTotal").classList.remove("active"); $("btnUnique").classList.remove("active");
    btn.classList.add("active");
  }

  async function main() {
    try {
      await load();
      wire();
      render();
      renderViewer("");
      renderProvenance();
      $("status").style.display = "none";
    } catch (e) {
      $("status").innerHTML = "<span class='err'>" + e.message +
        "</span><br><span class='muted'>Run <code>./build.sh</code> first, then serve this folder (files load from tokenizer/ and corpus/).</span>";
    }
  }
  main();
})();
