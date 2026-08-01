# V5 Mixture-and-Curriculum Plan

**Session 5 — the written specification for V5's data mixture.** A share of the budget for every capability slot, the Indic slot split across its four tiers, every slot sized against real supply and tied to the benchmark it exists to win, a protected floor, an anneal reserve, difficulty and reasoning-length bands, and a proxy experiment specified at 1B/3B scale.

This continues the [40B India-first design doc](../40b-parameter-model/design.md) and the [Session 4 cleaning pipeline](../assignment-4/readme.md). The working rule is unchanged: **anchor every number to a published measurement, and argue every departure.** One rule is added, because this artifact is graded on surviving a reviewer pushing on each number:

> **No slot gets a share its real supply cannot fund** at a repetition rate the scaling-law literature says still has value. Where a share can only be reached by repeating data or generating it, that is stated in the same row as the number — not in a footnote.

Applying that rule broke this plan's own predecessor. §1 is that correction.

**Where each part of the brief is answered**

| The brief asks for | Section |
|---|---|
| A defended share of the budget for every capability slot | §2 |
| The Indic split across verified / unverified / translated / synthetic | §3 |
| Agentic, reasoning and long-context named, each pointed at inventory datasets | §2, §4 |
| Each slot tied back to the benchmark it is meant to win | §2, §5 |
| Every slot sized against real supply; where a share needs repetition or generation, said plainly | §2 (columns 4–6), §1 |
| The protected always-on floor the selector may not cross | §6 |
| The reserve held back for the cooldown | §7 |
| Difficulty and reasoning-length bands, with a concrete example at each level | §8 |
| Written as a testable hypothesis | §9, §10 |
| A concrete proxy experiment at 1B or 3B scale, and the metric that confirms or refutes | §9 |

---

## 0. The one inherited assumption

No V5-specific compute budget was issued. This plan continues the **40B parameters / 15T tokens** budget from the 40B design doc — the only budget this cohort has anchored to precedent (Gemma-3-27B at 14T, Llama-3-70B at 15T, giving 375 tokens/param at 40B). **If V5's real scale differs, the percentages and the epoch ratios transfer; the absolute token counts rescale linearly.** Stated once, here, so no later number silently depends on an unexamined premise.

**Gating.** Per the brief, this plan is reviewed only once the cohort's cleaned-token threshold is met. Session 4's pipeline is the instrument for that; its measured 90.2% survival rate on a 3,600-row Sangraha sample is the throughput assumption behind §3's tier counts.

---

## 1. The correction this plan is built on

The 40B design doc gave **20% of the budget to Indic**. That number was reasoned from ambition and from Sarvam-1's claimed 2T-token Indic corpus. It is not reachable, and demonstrating why is the most load-bearing argument in this document.

**Sangraha publishes real token counts** ([arXiv:2403.06350](https://arxiv.org/abs/2403.06350), Table 1). Read carefully, they say:

| Sangraha split | Tokens | What it actually is |
|---|---:|---|
| Verified | 64.31B | Real, human-verified sites + OCR'd PDFs + ASR'd video — **but 12.76B of this is English** |
| Unverified | 24.31B | Real Indic text, perplexity-filtered out of CulturaX / MADLAD-400 |
| Synthetic | 162.71B | English WikiMedia **machine-translated** into 14 Indic languages, plus ~72B romanized transliteration |
| **Total** | **251.32B** | The headline |

Subtract the English: **real Indic text in the largest, most systematic Indic collection effort published is ~75.9B tokens.** The 251B headline is **65% translated or transliterated**, and none of the 72B romanized tokens are real web text — they are transliterations of already-translated Wikimedia.

Four independent corpora agree on the order of magnitude, and because they all scrape the same Common Crawl plus the same Indian news and government sites, they **overlap rather than sum**: CulturaX Indic 44.6B across 15 languages, IndicCorp v2 20.9B across 24, FineWeb2 Indic ≈40B *words*, and Llama-3-Nanda aggregating every available Hindi source to **65B tokens for Hindi alone**. Sangraha's 75.9B is effectively the deduplicated union; adding post-2023 crawls and the EKA corpus plausibly reaches 80–110B — **not a different order of magnitude.**

The supply side confirms it. Hindi does not appear in Common Crawl's top 20 languages despite being the third most-spoken language on earth, and Indian languages together are **under 1%** of it; W3Techs puts English at 49.5% of websites with Hindi, Tamil and Urdu each **below 0.1%**.

**The arithmetic.** A 20% Indic slot at 15T is 3.0T tokens. Against 75.9B of real Indic text that is **39.5 epochs**. Muennighoff et al. ([arXiv:2305.16264](https://arxiv.org/abs/2305.16264), ~400 runs to 9B params / 900B tokens) measured exactly this: up to **4 epochs** of repetition is "almost as good as new data" (an 8.7B model at 4 epochs finishes **0.5%** higher in validation loss than the unique-data run), value decays with a fitted half-life of **R\*_D = 15.39** epochs, and **"at 40 epochs, repeating is worthless."**

**So a 20% Indic slot backed by real Indic data is arithmetically unreachable.** Every published Indic headline above ~100B — Sangraha's 251B, Sarvam-2T's 2T, PARAM-1's 1.52T of Hindi, BhashaKritika's 540B — is translated, romanized, synthetic, or repeated. This plan therefore does not choose the Indic share. §3 derives it.

---

## 2. The budget

Every share is either derived from supply (§3) or set by a published dose-response ablation (§5). The supply columns are in the table, not a footnote, because they are what a reviewer will push on.

| Slot | Share | Tokens | Unique supply (named dataset) | Epochs | Repeat / generate verdict | Benchmark it must win |
|---|---:|---:|---|---:|---|---|
| General web | 47.5% | 7.13T | 18.5T — FineWeb | 0.4 | fresh, no repetition | **MMLU-Pro**, GPQA-Diamond |
| Code | 25.0% | 3.75T | 3.3T — The Stack v2 | 1.1 | ~1 epoch, inside precedent | **LiveCodeBench v6**, SWE-bench Verified, BigCodeBench |
| Indic (4 tiers) | 10.0% | 1.50T | 779B — Sangraha + IndicTrans2 BT + BhashaKritika | 1.9 | free zone (≤4 ep) — **but see §3, tier-by-tier** | **MILU**, IndQA, FLORES-200 chrF++ |
| Math | 8.0% | 1.20T | 370B — MegaMath | 3.2 | **repeats 3.2×**, inside the free zone | **AIME 2025 / HMMT** via MathArena |
| Long-context | 4.0% | 600B | 639B — concatenated Stack repos + books (ProLong recipe) | 0.9 | fresh | **HELMET**, LongBench v2 |
| Other world langs | 4.0% | 600B | 2.8T — MADLAD-400 clean | 0.2 | fresh | **Global-MMLU** (culturally-sensitive subset) |
| Reasoning | 1.0% | 150B | 35B — OpenThoughts3-1.2M + Llama-Nemotron-Post-Training | 4.3 | **repeats 4.3×**, at the free-zone edge | **GPQA-Diamond**, ARC-AGI-2 |
| Agentic | 0.5% | 75B | **0.25B real** — ToolBench-class trajectories | **300** | **cannot be repeated — 99.7% must be generated** | **BFCL V4**, τ²-bench |
| **Total** | **100.0%** | **15.0T** | | | | |

Held **out** of the above: an **anneal reserve of 3% (450B tokens)**, §7. Main phase = 14.55T.

Three slots need their honesty stated in words as well as columns. **Math repeats MegaMath 3.2 times** — real repetition, but inside the zone Muennighoff measured as costing ~0.5% of loss. **Reasoning repeats 4.3 times**, which is precisely SmolLM3's published choice (4 epochs over OpenThoughts3 for its ~140B reasoning mid-training stage), sitting on the edge of the free zone rather than inside it. **Agentic cannot be sourced by repetition at all**: 75B against 0.25B of real trajectories is 300 epochs, so this slot is **a commitment to generate, not a supply we hold** — §4 explains why that is the right call rather than a gap being papered over.

---

## 3. The Indic slot, derived rather than chosen

Four published pools, three published limits, one output. Nothing here is a preference.

| Tier | Source | Real pool | Epoch cap | Why that cap | Contributes |
|---|---|---:|---:|---|---:|
| **Verified** | Sangraha Verified − English | 51.5B | 8 | Highest-trust tier we have. 8 sits between Muennighoff's free zone (4) and half-life (15.4) — we spend repetition where quality is highest | 412B |
| **Unverified** | Sangraha Unverified | 24.3B | 6 | Real text, but only perplexity-filtered out of CulturaX/MADLAD — less quality assurance, so less repetition tolerance | 146B |
| **Translated** | Sangraha "Synthetic" + IndicTrans2 back-translation | 162.7B | 3 | MT artifacts compound under repetition; held strictly inside the free zone | 488B |
| **Synthetic** | BhashaKritika (Krutrim) | 540B pool | 0.83 | **Capped by ratio, not by supply** — see below | 448B |
| **Slot total** | | | | | **1,495B = 9.97% → 10%** |

Tier shares within the slot: **verified 27.6%, unverified 9.8%, translated 32.7%, synthetic 30.0%.**

**Resolving the four-tier naming.** The brief asks for four tiers; Sangraha ships three configs. The resolution is that **Sangraha's "Synthetic" config is misnamed for our purposes** — it is machine-translated Wikimedia plus transliteration, which is translation, not generation. So verified and unverified map to Sangraha's like-named configs (minus English); **translated** = Sangraha's "Synthetic" config plus IndicTrans2's back-translation augmentation (400.9M Indic sentences, from a BPCC whose 230M bitext pairs are ~99% mined or back-translated against only ~2.2M human-verified); **synthetic** = BhashaKritika's 540B genuinely LLM-generated tokens, of which only 62.88B is grounded in real Indic documents. Every tier now has a published token count and a distinct provenance, and the two words mean different things.

**Why synthetic is capped at 30% of the slot.** Kang et al. ([arXiv:2510.01631](https://arxiv.org/abs/2510.01631) — >1,000 models, >100,000 GPU-hours) find the optimal rephrased-synthetic proportion **converges to ~30%**, delivering a 5–10× speedup at large data budgets, while *rephrased synthetic alone shows no advantage over natural web* and *textbook-style synthetic alone is notably worse downstream*. Nemotron-CC independently shipped exactly this ratio: **1.9T synthetic of 6.3T = 30%**.

**Why synthetic is defensible here and would not be for General web.** Kazdan et al. ([arXiv:2410.16713](https://arxiv.org/abs/2410.16713), ICML 2025) find that **when real data are scarce, synthetic data reduces test loss on real data — and when real data are ample, it increases test loss.** Indic is the scarce case; General web is the ample case, which is why the same tool is used in one slot and refused in the other. Gerstgrasser et al. ([arXiv:2404.01413](https://arxiv.org/abs/2404.01413)) further show model collapse is an artifact of *replacing* real data with synthetic — when data **accumulate**, test error has a finite upper bound independent of iteration count. Shumailov et al. (Nature 631, 2024) reach the same place empirically: preserving 10% of original real data reduces collapse to "only minor degradation." **Our verified + unverified tiers are 37.4% of the slot, comfortably above that threshold.**

**The tension this leaves open, stated rather than hidden.** Sarvam-1 at 2B parameters beats Llama-3.1-8B on Indic evaluations (MMLU-Indic **44.44 vs 38.30**, ARC-C-Indic **58.50 vs 45.00**, TriviaQA-Indic **90.62 vs 61.47**) on a claimed 2T Indic tokens — a figure that per §1 cannot be mostly real. Either heavy synthesis and repetition work far better for Indic than the general scaling laws predict, or Sarvam's tokenizer (fertility 1.4–2.1, versus 4–8 for generic multilingual tokenizers — the same problem [our own BPE work](../india-bpe-tokenizer/) measured) is doing most of the work. **These two explanations imply opposite Indic shares. §9 is designed to separate them.**

---

## 4. Agentic, reasoning and long-context — named, and pointed at datasets

The brief asks for these three to be named explicitly and each pointed at the dataset that fills it. They are the three thinnest slots in the inventory, so each gets its supply stated plainly.

**Reasoning — 1.0% (150B) — OpenThoughts3-1.2M + Llama-Nemotron-Post-Training-Dataset.** Sized to the only published precedent of its kind: SmolLM3's reasoning mid-training ran **~140B tokens at 4 epochs** over exactly these two sources. 150B at 4.3 epochs matches that recipe almost exactly. An earlier draft of this plan proposed 4% (600B), which would have been **17 epochs** — past the half-life and indefensible.

**Long-context — 4.0% (600B) — concatenated Stack code repositories + books, ProLong recipe.** Sits between SmolLM3's 100B and Llama 3's ~800B long-context continued-pretraining stage (5.1% of 15.6T). Composition follows ProLong's measured optimum *and* its measured failure: **60% long / 40% short**, where **100% long data degrades both long- and short-context performance**, and within the long portion **code repositories alone give recall 99.2 but weak everything else, books alone give better in-context learning and summarization, and equal books+code is best overall.** Concretely: 30% concatenated code repos, 30% books, 3% textbooks, 40% short-context refresh.

**Agentic — 0.5% (75B) — ToolBench-class trajectories as seed only, remainder generated.** Real supply is **~0.25B tokens** (ToolBench: 126,486 instances, 469,585 real API calls), so 75B is 300 epochs — worthless as repetition. This slot is **99.7% a generation commitment.** Two facts make that the honest call rather than a shortfall: ToolBench is **deprecated in practice** (RapidAPI endpoints are dead or rate-limited, making results irreproducible, and it is heavily contaminated), and **no published ablation of agentic-data share against BFCL or τ-bench exists** — because labs buy agentic capability in post-training SFT and RL, not in pretraining. So the pretraining slot's job is stated narrowly: **install tool-call syntax** (the Gemma-4 six-token lifecycle scheme adopted in the 40B design doc), not tool-use policy. **Policy is bought in RL, and this plan does not pretend otherwise.**

---

## 5. Why the other shares are what they are, and which benchmarks are dead

**Code at 25% is a measured optimum, not a preference.** Cohere's *To Code, or Not To Code?* ([arXiv:2408.10914](https://arxiv.org/abs/2408.10914)) swept code share across **0 / 25 / 50 / 75 / 90 / 100%** at fixed 200B tokens, at both 470M and 2.8B params, with consistent results: natural-language reasoning is an **inverted-U peaking at 25% (+3.4% relative)**, code generation rises roughly **linearly**, and world knowledge is **monotonically harmed — −31% relative at 75% code, −86% at 100%.** A second study ([arXiv:2409.04556](https://arxiv.org/html/2409.04556v2)) gives per-unit coefficients: compositional generalization **+0.147**, arithmetic **+0.124 to +0.135**, but English syntax **−0.416** and knowledge tasks **−0.047 to −0.097**. 25% is where reasoning peaks and the knowledge cost is still slight. Muennighoff separately found mixing **up to 50% Python code caused no natural-language deterioration and gave ~2× effective tokens** — an independent reason a large code share is safe under repetition.

**Math at 8%** is sized by DeepSeekMath's corpus dose-response on a 1.3B model: **40B math tokens → GSM8K 11.5% / MATH 8.9%; 120B → 23.8% / 13.6%; 160B tested and not preferred.** Composition follows that paper's negative result too — despite being standard practice, "training on arXiv papers brings no notable improvements on all mathematical benchmarks" — so this slot is **web-math weighted, not paper-weighted.**

**Other world languages at 4%, with natural-distribution sampling and no temperature upsampling.** [arXiv:2510.25947](https://arxiv.org/html/2510.25947v1) reframes the curse of multilinguality: performance was **stable from 25 to 400 training languages** under natural distribution, and degradation came specifically from **temperature sampling that oversamples noisy low-resource languages.** The failure mode is a sampling choice, so this plan simply declines to make it.

**General web at 47.5%** takes the remainder, close to Llama 3's disclosed 50%, with ~30% of it Nemotron-CC-style synthetic rephrasing (the same ratio justified in §3; Nemotron-CC's high-quality subset gave **+5.6 MMLU over DCLM** at 1T tokens).

**The benchmarks this plan refuses to target**, because naming the dead ones is part of tying slots to benchmarks honestly. **GSM8K and MATH are saturated** (frontier >95%); the GSM1k replication found drops of **up to ~13 points** on a freshly-authored distribution-matched set — direct evidence of memorization. **AIME 2024 is contaminated**: MathArena found models scoring **10–20% above** their difficulty-aligned expectation, and one open model ~60% above. **HumanEval and MBPP are saturated and contaminated**, and with 164 problems each problem is worth 0.6%, so differences are noise. **MMLU is saturated** (frontier 88–92%, a spread within noise) with known label errors. **BBH is saturated** (>90%). **Needle-in-a-haystack is retired**: HELMET measured its correlation with real downstream long-context tasks at **0.68** and found models score either perfectly or near-zero; even RULER's headline average correlates only **0.74–0.77**, below the 0.85 reliability threshold — so HELMET's **RAG subset (0.92 correlation with LongQA)** is our cheap development proxy. **MGSM inherits GSM8K's contamination.** **IndicGenBench and the Airavata eval suite are translate-test by construction**, the exact artifact an India-first model must not be scored on — **MILU** (~75% natively sourced from 1,500+ Indian competitive exams, 79,617 questions, 11 Indic languages) and **IndQA** (fully native, authored by 261 Indian researchers and domain experts across 12 languages) are the native-reasoning evaluations, and they are the ones in §2.

One MILU finding drives the Indic slot's *composition* more than its size: **language-specific fine-tuned Indic models "perform only slightly better than random baselines" while general multilingual models beat them**, and the largest gaps are in **Arts & Humanities and Law & Governance, not STEM.** The deficit an India-first model must close is India-specific cultural and legal knowledge — not reasoning, and not language coverage.

---

## 6. The protected always-on floor

The floor exists because of a measured extrapolation bias, not as a statement of values. Sedova et al. ([arXiv:2605.12715](https://arxiv.org/html/2605.12715), 2,000+ runs) find the **optimal weight on a scarce target corpus shrinks with model size — 9.5% at 101M params falling to 1.9% at 539M.** A mixture weight fit on a 1B proxy and transferred naively to 40B will therefore **systematically under-weight the scarce slots.** The floor corrects a known bias in the selector, which is why the selector may not negotiate it.

| Slot | Floor | = % of its target |
|---|---:|---:|
| Indic (all four tiers) | 7.0% | 70% |
| Long-context | 3.0% | 75% |
| Other world languages | 3.0% | 75% |
| Reasoning | 0.8% | 80% |
| Agentic | 0.4% | 80% |
| **Code — CEILING, not floor** | **30.0%** | **120%** |

**The code ceiling is the non-obvious half of this section.** Code is low-entropy and easy to fit, so a selector minimizing average proxy loss will over-buy it. The Cohere ablation prices that mistake exactly: **−31% relative world knowledge at 75% code, −86% at 100%.** A floor-only policy leaves the plan exposed to the one failure mode with a measured, severe cost.

**Within the Indic slot, the verified tier carries an absolute floor of 206B tokens** — 51.5B × 4 epochs. The verified tier may never be repeated *fewer* than 4 times: four epochs is the free zone, drawing less from the highest-trust Indic data we have wastes it, and the tier is small enough already that any cut makes "verified" symbolic rather than real.

---

## 7. The anneal reserve: 3% (450B tokens)

**Sizing.** Two published results pull opposite ways and the resolution matters. Hägele et al. ([arXiv:2405.18392](https://arxiv.org/html/2405.18392v3)) sweep cooldown length directly and find it **surpasses cosine between 10–20% of steps, plateauing around 20%** — but the same paper shows the required fraction **shrinks as runs lengthen**: on a 200k-step run, "just 10k cooldown steps [5%] almost perfectly match cosine." [arXiv:2512.13705](https://arxiv.org/html/2512.13705) fits that decay explicitly, with the optimal anneal ratio scaling as **T^−0.946** — roughly 1/T. Extrapolating MiniCPM's validated 10%-at-1.1T by that exponent predicts **under 1% at 15T**.

Shipped practice at comparable scale brackets it from the other side: **Nemotron-H 380B of 20T = 1.9%; Phi-4 250B = 2.5%; DeepSeek-V3's terminal low-LR tail 500B = 3.4%; Hunyuan-Large an explicit 5%; OLMo 2 ~1.2% per anneal branch.** At the far end, **Llama 3.1 405B annealed over just 40M tokens — 0.00026%** — using it as a Polyak-averaging window rather than a data phase.

**3% sits at the median of shipped practice in our token range and above the theoretical floor**, with the margin spent on souping rather than length. MiniCPM's 10%-beats-2.5% ablation is the lower bound this respects.

**Composition**, anchored to OLMo 2's published Dolmino-mix-1124 (its 50B mix is DCLM 47.2% / FLAN 16.6% / math 20.8% / Wiki 7.11% / peS2o 5.85% / StackExchange 2.45% — **~37% instruction-plus-math against essentially zero in stage 1**), with three modifications, each backed by an ablation:

**Code goes in the anneal.** Cohere measured cooldown-with-code against cooldown-without: **+3.6% NL reasoning, +10.1% world knowledge, +20% code, +4.1% win-rate**, where cooldown without code gave minimal gains. Highest-leverage anneal decision available, and free.

**SFT-grade data goes inside the decay, not after it.** MiniCPM's ablation is direct: high-quality data mixed into decay then 4B SFT scored **≈43.0**; decay on pretraining data alone then 4B SFT scored **≈35.4**; and doubling the later SFT (6B→12B) moved 33.7→33.9 — **more SFT afterwards does not recover the gap.**

**The verified Indic tier is upsampled here.** This is where a 51.5B tier can actually matter: concentrating the highest-trust Indic data at the point of maximum gradient impact rather than diluting it across 14.55T. SmolLM2 shows the size of this effect in a related slot — moving math from 10% to **14% in the decay phase tripled its math score, 7.27 → 22.07.** Position in the schedule, not just share, buys the capability.

**Souping.** Following OLMo 2, the reserve is spent as **3 independent anneal runs over different permutations of the same mix, then averaged**; they report this "consistently produces equal or better performance than any individual training run." That is why 3% rather than 1.9% — the margin buys three branches, not a longer single run.

**The caveat that must be stated.** Llama 3 found annealing on high-quality math improved an 8B model's GSM8K by **24.0%** and MATH by **6.4%**, but improvements on the **405B model were "negligible"** — strong in-context learning at scale makes the in-domain injection redundant. **At 40B we sit between those poles with no published data point at our scale.** The anneal reserve is the second-most-uncertain commitment here, which is why §9 makes it a measured arm rather than an assumption.

---

## 8. Difficulty and reasoning-length bands

Four bands, each with a worked example and each mapped to the evaluation whose difficulty it mirrors. Bands are assigned per-sample at ingest and drive both curriculum ordering and anneal composition.

**Band 0 — Recall.** Under 100 tokens, 0–1 steps. **15%** of reasoning-bearing data.
> *"भारत की राजधानी क्या है?"* → *"नई दिल्ली।"*

A single lookup with no intermediate state. Mirrors the recall floor MILU tests.

**Band 1 — Short chain.** 100–500 tokens, 2–3 steps. **35%**.
> *"A shop sells pens at ₹12 each. Ravi buys 7 pens and pays with a ₹100 note. How much change does he get?"* → 7 × 12 = 84; 100 − 84 = **₹16**.

Two arithmetic steps, both written out. Mirrors the GSM8K band — retained for curriculum shaping, not for scoring, since §5 rules GSM8K out as saturated.

**Band 2 — Multi-hop.** 500–2,000 tokens, 4–8 steps. **35%**.
> *"Under the Indian Contract Act 1872, is an agreement made by a 17-year-old to buy a bicycle enforceable, and what happens to money already paid?"* → establish the age of majority (18); classify the agreement as void ab initio; then apply the separate doctrine of restitution to the payment already made.

Two distinct legal rules chained, the second not implied by the first. Mirrors MILU's Law & Governance split — the exact domain MILU identifies as the largest Indic gap.

**Band 3 — Extended.** 2,000–8,000+ tokens. **15%**.
> A tool-use trajectory: call a rail-availability API; receive a malformed response; retry with a corrected query; call a fare API; reconcile a conflict between the two results; then answer.

**Error recovery sits inside the band, not outside it** — that is the axis BFCL V4's multi-turn and hallucination components actually score.

Two decisions inside these bands. **The distribution is deliberately not uniform**: Bands 1 and 2 carry 70% between them, because Band 0 is cheap and abundant while Band 3 is the slot with essentially no real supply (§4). **And Band 2 and 3 examples are Indic-context by default rather than translated from English templates** — the 40B design doc already identified the failure mode (machine-translated instruction data carries English framing into grammatically-correct target-language text), and MILU independently confirms the measured Indic deficit sits in Arts, Humanities, Law and Governance, precisely the content a translated template silently replaces with US or EU defaults.

---

## 9. The proxy experiment

Every number above is a hypothesis about a 40B run, and none of them has been tested at that scale. This section specifies the experiment that would confirm or refute the mixture before it is trusted at full scale, at a cost of roughly 0.5% of the full training budget.

**Why small proxies are believed to transfer.** DoReMi set domain weights on a **280M** proxy for an **8B** model (30× scale-up) and the weights held. RegMix fit its regression on **512 models at 1M parameters / 1B tokens**, transferred to a **1B model at 25B tokens** — 1,000× larger and 25× longer — where the predicted mixture beat 63 competing candidates, and validated the method up to **7B models at 100B tokens**.

**The ladder.** Two scales, three arms each, Chinchilla-matched tokens:

| | Params | Tokens |
|---|---:|---:|
| Rung 1 | 1B | 20B |
| Rung 2 | 3B | 60B |

| Arm | Mixture | What it represents |
|---|---|---|
| **A** | The §2 plan as written — Indic 10%, reasoning 1%, agentic 0.5% | The hypothesis |
| **B** | Supply-honest — every slot capped at its unrepeated real ceiling, remainder to General web and Code | The conservative null |
| **C** | The 40B design doc's original — Indic 20% reached by ~40× repetition | The claim §1 says must fail |

A fourth arm at Rung 2 only: **A-noanneal**, identical to A but spending the 3% reserve on more main-phase tokens instead, to test §7's uncertain commitment directly.

**The metrics that confirm or refute.** Per-slot held-out loss for all eight slots is the cheap continuous signal, but the decision rests on five downstream numbers, chosen because §5 rules the alternatives dead:

| Slot under test | Metric | Why this one |
|---|---|---|
| Indic | **MILU**, native subset only | ~75% natively sourced; translate-test alternatives would hide the exact gap being measured |
| Code | **LiveCodeBench v6**, problems released after the data cutoff only | Contamination-proof by construction, unlike HumanEval |
| Math | **AIME-2025-style held-out set** | GSM8K and MATH are saturated; AIME 2024 is contaminated |
| Long-context | **HELMET RAG subset** | 0.92 correlation with full LongQA at a fraction of the cost |
| Agentic | **BFCL V4 multi-turn split** | The only current standard that scores state tracking and hallucination |

**The decision rules, fixed in advance so the result cannot be rationalized afterwards.**

**On the Indic share — the plan's central bet.** If **arm C matches or beats arm A on MILU while losing no more than 1 point of MMLU-Pro**, then the repetition ceiling does not bind for Indic, §1 is wrong, and the Indic share returns to 20%. If **arm A beats C on MILU by more than the seed noise**, the derived 10% stands. If **arm B matches arm A on MILU**, then the translated and synthetic tiers are contributing nothing and the slot should shrink to its real-data core.

**On the anneal.** If **A and A-noanneal separate by less than the seed noise on MILU and AIME**, the 3% reserve is cargo-culted from smaller models and should collapse toward Llama 3's ~0%, consistent with the 405B "negligible" result in §7.

**On the code ceiling.** If a DoReMi-style selector run against these slots **never pushes code above 30%**, the ceiling in §6 is dead weight and the Cohere inverted-U does not transfer to this mixture.

**On the tokenizer confound.** Run arm A twice at Rung 1 — once with the 256K vocabulary from the 40B design doc, once with a generic multilingual tokenizer at matched size. If **MILU tracks tokenizer fertility more strongly than Indic share**, then Sarvam-1's result is a tokenizer result, the Indic *share* is close to irrelevant above the floor, and the budget should move to vocabulary allocation instead — which the 40B design doc already sized at 110K of 256K.

**Sequencing.** Rung 1 runs all four configurations first; only arms that survive Rung 1's decision rules are promoted to Rung 2. That keeps the total cost near 0.5% of the 15T budget rather than 2%.

---

## 10. What would falsify this plan

Stated as commitments so a reviewer can hold the plan to them rather than argue about them.

**The Indic ceiling is the central claim.** If arm C succeeds, §1's supply argument collapses and with it the single most consequential number in this document.

**The epoch caps are interpolations, not measurements.** Muennighoff gives 4 (free) and 15.4 (half-life) for single-source repetition; Sedova et al. find scarce corpora *mixed with abundant generic data* tolerate **15–20** repetitions, which would justify caps roughly 2× higher and an Indic share near 18%. **The gap between those two laws is the widest uncertainty band here — worth about 8 percentage points of Indic share**, and arms A versus C are precisely the test that separates them.

**The agentic position is an argument, not a measurement.** No published dose-response exists for agentic pretraining share. If a BFCL V4 gap opens between arms at Rung 2 that tracks pretraining agentic share, then agentic capability is *not* purely a post-training purchase and §4 is wrong.

**The anneal may be scale-dependent in a way that erases it.** Llama 3's 405B result says the injection stops mattering as models grow. If that threshold is below 40B, §7 should be deleted rather than shrunk.

---

## 11. Open questions, flagged rather than resolved

**The 15T / 40B budget is inherited, not issued** (§0). Every absolute token count rescales if V5's real scale differs; the percentages and epoch ratios are what survive.

**The EKA corpus (458GB, IIT-Gandhinagar / Soket, IndiaAI Mission) publishes no token count**, only "multi-billion." If it is substantially non-overlapping with Sangraha it moves §1's ceiling materially — and nobody has checked. This is the cheapest available way to improve this plan.

**Session 4's token counts are whitespace proxies, not BPE counts.** That pipeline flagged this itself. §3's tier counts come from Sangraha's published figures rather than from our pipeline precisely because of that gap — but the gap should be closed by re-running the pipeline's counting stage through the [india-bpe-tokenizer](../india-bpe-tokenizer/) before the next mixture revision.

**Sarvam-1, Krutrim and PARAM-1 publish no real-versus-synthetic split** for their Indic corpora. Every comparison in §3 against those models therefore compares our disclosed composition against their undisclosed one.

**"Other world languages" at 4% is the least-examined number in §2** — it is carried from the 40B design doc with only the sampling-strategy correction in §5 applied.

---

## References

**Scaling and repetition** — [Muennighoff et al., Scaling Data-Constrained LMs, arXiv:2305.16264](https://arxiv.org/abs/2305.16264) · [Xue et al., To Repeat or Not To Repeat, arXiv:2305.13230](https://arxiv.org/abs/2305.13230) · [Sedova et al., Mixture Pretraining Under Data Constraints, arXiv:2605.12715](https://arxiv.org/html/2605.12715) · [Prescriptive Scaling Laws for Data Constrained Training, arXiv:2605.01640](https://arxiv.org/html/2605.01640) · [Hoffmann et al., Chinchilla, arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

**Mixture selection** — [DoReMi, arXiv:2305.10429](https://arxiv.org/abs/2305.10429) · [RegMix, arXiv:2407.01492](https://arxiv.org/abs/2407.01492) · [Data Mixing Laws, arXiv:2403.16952](https://arxiv.org/abs/2403.16952) · [ATLAS transfer scaling laws, arXiv:2510.22037](https://arxiv.org/abs/2510.22037)

**Dose-response ablations** — [To Code, or Not To Code?, arXiv:2408.10914](https://arxiv.org/abs/2408.10914) · [How Does Code Pretraining Affect Task Performance?, arXiv:2409.04556](https://arxiv.org/html/2409.04556v2) · [DeepSeekMath, arXiv:2402.03300](https://arxiv.org/html/2402.03300v3) · [ProLong, ACL 2025](https://aclanthology.org/2025.acl-long.366.pdf) · [Revisiting Multilingual Data Mixtures, arXiv:2510.25947](https://arxiv.org/html/2510.25947v1)

**Synthetic data** — [Kang et al., Demystifying Synthetic Data, arXiv:2510.01631](https://arxiv.org/abs/2510.01631) · [Nemotron-CC, arXiv:2412.02595](https://arxiv.org/abs/2412.02595) · [WRAP, arXiv:2401.16380](https://arxiv.org/abs/2401.16380) · [Gerstgrasser et al., Is Model Collapse Inevitable?, arXiv:2404.01413](https://arxiv.org/abs/2404.01413) · [Kazdan et al., Collapse or Thrive?, arXiv:2410.16713](https://arxiv.org/abs/2410.16713) · [Shumailov et al., Nature 631](https://www.nature.com/articles/s41586-024-07566-y)

**Anneal and curriculum** — [OLMo 2, arXiv:2501.00656](https://arxiv.org/pdf/2501.00656) · [Dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) · [MiniCPM, arXiv:2404.06395](https://arxiv.org/pdf/2404.06395v2) · [Hägele et al., Beyond Fixed Training Durations, arXiv:2405.18392](https://arxiv.org/html/2405.18392v3) · [Annealing Scaling & Transferability, arXiv:2512.13705](https://arxiv.org/html/2512.13705) · [Llama 3 Herd, arXiv:2407.21783](https://arxiv.org/pdf/2407.21783v3) · [SmolLM2, arXiv:2502.02737](https://arxiv.org/html/2502.02737v1) · [SmolLM3](https://huggingface.co/blog/smollm3) · [Nemotron-H, arXiv:2504.03624](https://arxiv.org/html/2504.03624v2) · [Hunyuan-Large, arXiv:2411.02265](https://arxiv.org/pdf/2411.02265) · [Qwen3, arXiv:2505.09388](https://arxiv.org/pdf/2505.09388) · [DeepSeek-V3, arXiv:2412.19437](https://arxiv.org/pdf/2412.19437) · [Phi-4, arXiv:2412.08905](https://arxiv.org/html/2412.08905)

**Indic data and evaluation** — [Sangraha / IndicLLMSuite, arXiv:2403.06350](https://arxiv.org/abs/2403.06350) · [Sangraha on HF](https://huggingface.co/datasets/ai4bharat/sangraha) · [IndicTrans2, arXiv:2305.16307](https://arxiv.org/abs/2305.16307) · [BhashaKritika, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/40524/44485) · [IndicCorp v2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) · [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) · [Llama-3-Nanda-10B, arXiv:2504.06011](https://arxiv.org/abs/2504.06011) · [PARAM-1, arXiv:2507.13390](https://arxiv.org/abs/2507.13390) · [Sarvam-1](https://www.sarvam.ai/blogs/sarvam-1) · [MILU, NAACL 2025](https://aclanthology.org/2025.naacl-long.507.pdf) · [W3Techs content languages](https://w3techs.com/technologies/overview/content_language)

**Benchmarks** — [MMLU-Pro, arXiv:2406.01574](https://arxiv.org/abs/2406.01574) · [Global-MMLU, arXiv:2412.03304](https://arxiv.org/abs/2412.03304) · [MathArena, arXiv:2505.23281](https://arxiv.org/pdf/2505.23281) · [HELMET, arXiv:2410.02694](https://arxiv.org/abs/2410.02694) · [RULER, arXiv:2404.06654](https://arxiv.org/abs/2404.06654) · [LongBench v2, ACL 2025](https://aclanthology.org/2025.acl-long.183/) · [BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) · [τ²-bench, arXiv:2506.07982](https://arxiv.org/abs/2506.07982) · [LiveCodeBench](https://livecodebench.github.io/)

**Prior work in this repo** — [40b-parameter-model/design.md](../40b-parameter-model/design.md) · [assignment-4](../assignment-4/readme.md) · [india-bpe-tokenizer](../india-bpe-tokenizer/)

---

*Figures throughout are planning estimates grounded in the sources above, not measurements from a training run. The proxy experiment in §9 is specified, not yet executed.*
