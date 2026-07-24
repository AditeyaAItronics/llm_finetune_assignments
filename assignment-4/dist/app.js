const STAGE_META = [
  { key: "1_extract", num: 1, title: "Extract", desc: "Validate the corpus is already clean prose with no HTML/entity residue leaking through extraction." },
  { key: "2_normalize", num: 2, title: "Normalize", desc: "NFC Unicode normalize, strip control/zero-width/BOM/bidi noise, unescape HTML entities, collapse whitespace — while preserving Indic ZWNJ/ZWJ joiners." },
  { key: "3_language_id", num: 3, title: "Language ID & validation", desc: "Detect each document's language at runtime and compare against its claimed folder/config label; quarantine mismatches instead of trusting the path." },
  { key: "4_quality_filter", num: 4, title: "Quality filter", desc: "Gopher/C4-style heuristic cascade (word length, symbol ratio, punctuation, duplicate lines, stop-words) — applied per-language, not assumed English." },
  { key: "5_deduplicate", num: 5, title: "Deduplicate", desc: "Shingle → MinHash signature → LSH banding to catch near-duplicates without comparing every document to every other." },
  { key: "6_pii_scrub", num: 6, title: "PII scrub", desc: "Regex layer for structured identifiers (emails/phones/IPs) plus a name-detection layer, with the precision/recall tension made explicit." },
  { key: "7_decontaminate", num: 7, title: "Decontaminate", desc: "Fingerprint held-out/benchmark material and n-gram-scan every document against it; canary strings prove a leak would be caught." },
  { key: "8_manifest", num: 8, title: "Manifest", desc: "Every surviving shard ships a provenance record (source, license, cleaning-script hash, content hash, token count) — no shard enters without one." },
];

let STATS = null;
let DATASET_STATS = null;

async function loadData() {
  const [stats, ds] = await Promise.all([
    fetch("./data/stats.json").then(r => r.json()),
    fetch("./data/dataset_stats.json").then(r => r.json()),
  ]);
  STATS = stats;
  DATASET_STATS = ds;
  render();
}

function fmt(n) {
  return Number(n).toLocaleString("en-US");
}
function pct(n) {
  return (n * 100).toFixed(1) + "%";
}

function renderStageList() {
  const ol = document.getElementById("stageList");
  ol.innerHTML = STAGE_META.map(s => {
    const st = STATS.stages[s.key];
    return `<li class="stage-item">
      <div class="stage-num">${s.num}</div>
      <div>
        <h4>${s.title} <span style="color:var(--text-dim); font-weight:400; font-size:0.85rem;">— ${fmt(st.in)} → ${fmt(st.out)}</span></h4>
        <p>${s.desc}</p>
      </div>
    </li>`;
  }).join("");
}

function renderDatasetStats() {
  const verified = DATASET_STATS.size.configs.find(c => c.config === "verified");
  document.getElementById("dsRows").textContent = fmt(verified.num_rows);
  document.getElementById("dsBytes").textContent = (verified.num_bytes_parquet_files / 1e9).toFixed(1) + " GB";
  document.getElementById("sampleSizeNote").textContent = fmt(STATS.totals.raw_sample_rows);
}

function renderFunnel() {
  const funnel = document.getElementById("funnel");
  const maxIn = STATS.stages["1_extract"].in;
  funnel.innerHTML = STAGE_META.map(s => {
    const st = STATS.stages[s.key];
    const w = (st.out / maxIn) * 100;
    return `<div class="funnel-bar-row" data-key="${s.key}">
      <div class="funnel-label">${s.num}. ${s.title}</div>
      <div class="funnel-track"><div class="funnel-fill" style="width:${w}%"></div></div>
      <div class="funnel-pct">${fmt(st.out)}</div>
    </div>`;
  }).join("");
  funnel.querySelectorAll(".funnel-bar-row").forEach(row => {
    row.addEventListener("click", () => renderStageDetail(row.dataset.key));
  });
  renderStageDetail("4_quality_filter");
}

function renderStageDetail(key) {
  const meta = STAGE_META.find(s => s.key === key);
  const st = STATS.stages[key];
  const box = document.getElementById("stageDetail");
  let extra = "";

  if (key === "3_language_id") {
    extra = `<div class="kv">
      <div><b>${st.mismatches_quarantined}</b><span>mislabeled docs quarantined</span></div>
      <div><b>${st.unsupported_by_detector_passed_through}</b><span>lang not covered by detector (passed through)</span></div>
    </div>
    <p class="note">Example real mismatches: ${st.example_mismatches.slice(0,4).map(m => `claimed <b>${m.claimed}</b> → detected <b>${m.detected}</b>`).join(" · ")}</p>`;
  } else if (key === "4_quality_filter") {
    const dl = st.dropped_by_language;
    extra = `<div class="kv">${Object.entries(dl).map(([lang,n]) => `<div><b>${n}</b><span>${lang} dropped</span></div>`).join("")}</div>
    <p class="note">${st.note}</p>`;
  } else if (key === "5_deduplicate") {
    extra = `<div class="kv">
      <div><b>${st.near_or_exact_duplicates_removed}</b><span>near/exact duplicates removed</span></div>
    </div>
    <p class="note">${st.method}</p>`;
  } else if (key === "6_pii_scrub") {
    const r = st.redactions;
    extra = `<div class="kv">
      <div><b>${r.email}</b><span>emails redacted</span></div>
      <div><b>${r.phone}</b><span>phone numbers redacted</span></div>
      <div><b>${r.ipv4}</b><span>IPv4 addresses redacted</span></div>
      <div><b>${fmt(r.name_like)}</b><span>name-like spans flagged (illustrative)</span></div>
    </div>
    <p class="note">${st.note}</p>`;
  } else if (key === "7_decontaminate") {
    extra = `<div class="kv">
      <div><b>${st.canaries_checked}</b><span>canary strings checked</span></div>
      <div><b>${st.real_contamination_found}</b><span>real contamination found</span></div>
      <div><b>${st.self_test_injected_canary_detected ? "PASS" : "FAIL"}</b><span>self-test: injected canary caught</span></div>
    </div>
    <p class="note">${st.note}</p>`;
  } else if (key === "8_manifest") {
    extra = `<div class="kv">
      <div><b>${fmt(st.manifests_emitted)}</b><span>manifests emitted</span></div>
      <div><b class="mono" style="font-size:0.95rem">${st.cleaning_script_hash}</b><span>cleaning-script hash</span></div>
    </div>
    <p class="note">${st.token_count_caveat}</p>`;
  } else if (key === "2_normalize") {
    extra = `<div class="kv">
      <div><b>${fmt(st.chars_stripped_total)}</b><span>noise characters stripped</span></div>
      <div><b>${fmt(st.indic_joiners_preserved_zwnj_zwj)}</b><span>ZWNJ/ZWJ joiners preserved</span></div>
    </div>`;
  } else if (key === "1_extract") {
    extra = `<div class="kv"><div><b>${st.docs_with_markup_residue_found}</b><span>docs with markup residue found</span></div></div>
    <p class="note">${st.note}</p>`;
  }

  box.innerHTML = `<h4>${meta.num}. ${meta.title} — ${fmt(st.in)} in → ${fmt(st.out)} out</h4>${extra}`;
}

function renderTotals() {
  const t = STATS.totals;
  const grid = document.getElementById("totalsGrid");
  grid.innerHTML = `
    <div class="stat"><b>${fmt(t.raw_sample_rows)}</b><span>raw sample rows (real, fetched live)</span></div>
    <div class="stat"><b>${fmt(t.final_rows)}</b><span>final clean rows</span></div>
    <div class="stat"><b>${pct(t.survival_rate)}</b><span>overall survival</span></div>
    <div class="stat"><b>${fmt(t.total_token_proxy)}</b><span>total tokens (whitespace proxy)</span></div>
    <div class="stat"><b>${STATS.stages["3_language_id"].mismatches_quarantined}</b><span>mislabeled docs caught</span></div>
    <div class="stat"><b>${STATS.stages["5_deduplicate"].near_or_exact_duplicates_removed}</b><span>duplicates removed</span></div>
    <div class="stat"><b>${STATS.stages["6_pii_scrub"].redactions.email + STATS.stages["6_pii_scrub"].redactions.phone}</b><span>emails+phones redacted</span></div>
    <div class="stat"><b>2</b><span>real English-centric bugs found &amp; fixed</span></div>
  `;
}

function renderLangTable() {
  const tbody = document.querySelector("#langTable tbody");
  const rows = Object.entries(STATS.per_language).sort((a,b) => b[1].raw - a[1].raw);
  tbody.innerHTML = rows.map(([lang, v]) => {
    const survival = v.raw ? v.final / v.raw : 0;
    return `<tr>
      <td>${lang}</td>
      <td>${fmt(v.raw)}</td>
      <td>${fmt(v.final)}</td>
      <td class="bar-cell"><div class="mini-bar"><div style="width:${(survival*100).toFixed(0)}%"></div></div></td>
      <td>${pct(survival)}</td>
    </tr>`;
  }).join("");
}

function renderStageTable() {
  const tbody = document.querySelector("#stageTable tbody");
  tbody.innerHTML = STAGE_META.map(s => {
    const st = STATS.stages[s.key];
    const dropped = st.in - st.out;
    return `<tr><td>${s.num}. ${s.title}</td><td>${fmt(st.in)}</td><td>${fmt(st.out)}</td><td>${fmt(dropped)}</td></tr>`;
  }).join("");
}

function render() {
  renderStageList();
  renderDatasetStats();
  renderFunnel();
  renderTotals();
  renderLangTable();
  renderStageTable();
}

// Tabs
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

loadData().catch(err => {
  document.body.innerHTML = `<div style="padding:40px;color:#f87171;font-family:monospace">Failed to load data: ${err}</div>`;
});
