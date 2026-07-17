# Design Doc — 40B, India-First Foundation Model

Data and tokenizer strategy for a 40B-parameter model matched to Gemma-class quality, strong at coding, agentic work, and Indic languages, and "India first" in worldview. This is the text version of [`index.html`](./index.html) plus the reasoning trail behind it — what was tried, what got revised, and why.

---

## 0. The brief, and how it maps to this doc

The assignment asks for four things. Here's where each is answered, both in this file and on the live page:

| Ask | Section here | Section on the page |
|---|---|---|
| What data to collect, for pretraining / post-training / RL-alignment, and why | §3, §5, §6 | S3, S5, S6 |
| How to clean the data for these objectives | §4 | S4 |
| How to test the model against the objective | §7 | S7 |
| Fertility targets per language, plus coding/science/math/agentic, and the resulting tokenizer vocab size | §2 | S1, S2 |

---

## 1. Approach

The instinct on a task like this is to pick round numbers that "feel" competitive — a vocab size that sounds big, a data split that sounds balanced. That produces a plan that can't be defended if someone asks "why 20% and not 15%?"

So the working rule for every number in this doc was: **find a real, shipped model that already made this decision, and anchor to it.** Where our situation differs from that precedent (mainly: this model is explicitly India-first, the precedents mostly aren't), the number moves — but the move itself has to be explained relative to the anchor, not invented from nothing.

Four anchors did most of the work:

- **Gemma 2 / Gemma 3 technical reports** — vocab size (256K / 262K), training token counts (13T–14T at 27B), tokenizer design (SentencePiece, digit-splitting).
- **LLaMA 3 Herd of Models** — the only frontier paper that discloses an actual pretraining mix (50% general / 25% math / 17% code / 8% multilingual).
- **Sarvam-1 and Krutrim-1** — real, shipped Indic-focused tokenizers, with published vocab sizes and *measured* fertility numbers. These matter most: they're the evidence that our fertility targets are achievable, not aspirational.
- **Chinchilla (Hoffmann et al.)** — the compute-optimal ratio every real model deliberately overshoots, and by how much.

---

## 2. Tokenizer & fertility (answers "what fertility, what vocab size")

### 2.1 Why the tokenizer is decided before the data mix

Fertility (tokens per word) determines how much effective context and how much inference cost each language gets. A model that spends 4 tokens on a Hindi word it spends 1.3 tokens on in English is worse for Indic users no matter how good the pretraining data is. So fertility targets are a design decision made up front, not something read off after training.

### 2.2 Fertility targets, by language group

| Group | Target (tokens/word) | Basis |
|---|---|---|
| English | 1.2 | Standard baseline — GPT-4o, LLaMA-3, Gemma all land 1.2–1.3 |
| Devanagari cluster (Hindi, Marathi, Nepali, Sanskrit, Konkani, Bodo, Maithili, Dogri) | 1.4–1.6 | Matches Sarvam-1's measured low end |
| Bengali, Gujarati, Odia, Punjabi | 1.5–1.8 | Other Brahmic scripts, less morphologically complex than Dravidian |
| Tamil, Telugu, Kannada, Malayalam | 1.8–2.1 | Agglutinative morphology; this is Sarvam-1's *measured ceiling*, not a number we chose to undershoot — promising better without a new tokenization technique would be unearned |
| Urdu (Perso-Arabic, RTL) | 1.7–1.9 | Separate script family, separate merge budget |
| Santhali, Manipuri, Kashmiri, Sindhi, and other low-resource scheduled languages | 2.2–2.8 | Honest target given small available corpora — not dressed up to look better |

### 2.3 Fertility for coding, science/math, and agentic tasks

These aren't natural language, so "tokens per word" doesn't apply — each needed its own metric:

- **Code — 3.0–3.3 bytes/token.** Compression ratio, not fertility, is the right measure here. Target anchored to StarCoder2-class code tokenizers, dense enough that long files still fit in context.
- **Math & science — 1 token per digit.** Gemma and Qwen tokenize numbers digit-by-digit (`"482"` → 3 tokens); LLaMA-3 instead merges numbers into 3-digit chunks. We follow Gemma/Qwen: per-digit splitting gives predictable digit-position boundaries, which measurably helps arithmetic (the alternative causes a token-count discontinuity right at the 10s/100s boundary that models have to learn around). LaTeX operators and Greek letters also get single dedicated tokens rather than fragmenting into multiple BPE pieces.
- **Agentic / tool-calling — 6 reserved lifecycle tokens.** Rather than leaving tool-call JSON to generic BPE merges, Gemma 4 reserves 6 special tokens for the tool-use lifecycle (open/close pairs for tool-definition, tool-call, and tool-result, plus one string-delimiter token that keeps `{`, `}`, and quotes inside string values from being misread as structure). We adopt the same scheme rather than inventing our own — it's a solved problem.

### 2.4 Vocab size: 256,000

Derivation, top-down:

1. Gemma-2 shipped a 256K vocab at 27B params — one full pretraining run's worth of evidence that this scale works at a size close to ours.
2. Gemma-3 went to 262,144 to cover Gemini's ~140-language scope. We don't need that scope, so we don't pay for it.
3. Internal allocation:

| Allocation | Tokens | Share | Rationale |
|---|---:|---:|---|
| English / Latin | 42,000 | 16.4% | Matches the Llama-3/Gemma baseline allocation |
| Code | 32,000 | 12.5% | StarCoder2-class code vocab size |
| 22 Indic languages (~11 scripts) | 110,000 | 43.0% | Exceeds Sarvam-1's *entire* 68K vocab, because Sarvam covers 10 "major" languages and we cover all 22 scheduled ones, including low-resource scripts Sarvam doesn't attempt |
| Other world languages | 55,000 | 21.5% | Keeps the model globally usable, not India-only |
| Math/science notation, agentic reserved tokens, growth headroom | 17,000 | 6.6% | LaTeX/operator symbols (~6K), the Gemma-4-style tool-call tokens (~2K), remaining reserved for control tokens and future growth |
| **Total** | **256,000** | **100%** | |

**Why not push the Indic allocation even higher, e.g. to match Krutrim's per-language 100K training cap?** Krutrim trains per-language tokenizers up to 100K each but *merges and prunes* into a shared 70,400-token final vocab — the 100K figure is a training-time ceiling, not the deployed size. Our 110K for 22 languages is already proportionally larger than what either shipped Indic tokenizer needed for far fewer languages.

---

## 3. Pretraining data mix (~15 trillion tokens)

### 3.1 Token budget: 15T

Chinchilla-optimal is 20 tokens/parameter — a rule that minimizes training FLOPs and ignores inference cost, which is why no competitive shipped model uses it. Real precedents, plotted on the same log-log axis (params vs. tokens/param):

| Model | Tokens | Params | Tokens/param | × Chinchilla-optimal |
|---|---:|---:|---:|---:|
| LLaMA-2-7B | 2T | 7B | 286 | 14.3× |
| Gemma-2-27B | 13T | 27B | 482 | 24.1× |
| Gemma-3-27B | 14T | 27B | 519 | 25.9× |
| LLaMA-3-8B | 15T | 8B | 1,875 | 93.8× |
| LLaMA-3-70B | 15T | 70B | 214 | 10.7× |
| **Ours — 40B** | **15T** | **40B** | **375** | **18.75×** |

Two real models land almost exactly on our token count: Gemma-3-27B (14T) and LLaMA-3-70B (15T, the same corpus reused for its 8B sibling — evidence that 15T tokens is itself achievable at scale, independent of model size). At 40B params, 15T tokens puts us at 375 tokens/param — deliberately between Gemma-3-27B's ratio and LLaMA-3-70B's, because a 40B model's deployment lifetime looks more like a 70B's (long-lived production model) than an 8B's (squeezed for minimum footprint).

This is a ceiling assumption, not a free lunch: it only pays off if the extra tokens are genuinely new information. If Indic-corpus scarcity (see §3.3) forces repeated epochs to hit 15T, the effective ratio is worse than the headline number — flagged as an open question in §8.

### 3.2 The mix

LLaMA-3 is the one frontier model that discloses its real mix: 50% general knowledge, 25% math/reasoning, 17% code, 8% multilingual. That split is correct for an English-first model and wrong for an India-first one — the 8% multilingual figure is the one to argue with.

| Slice | Share | Tokens | Reasoning |
|---|---:|---:|---|
| General web | 38% | 5.70T | Cut from LLaMA-3's 50% to fund Indic + code |
| Code | 20% | 3.00T | Up from LLaMA-3's 17%, short of Qwen2.5-Coder's 70% specialist mix — coding is a stated objective, but this stays a general model |
| Math/reasoning | 12% | 1.80T | Below LLaMA-3's aggressive 25% — math still transfers to coding/agentic reasoning, just not tuned as hard for an English benchmark suite we're not optimizing for alone |
| Indic languages (native) | 20% | 3.00T | 2.5× LLaMA-3's *entire* multilingual share, deliberately. At 3.00T tokens this alone is 1.5× the size of Sarvam-1's whole dedicated Indic corpus (2T) — ambitious, but anchored to a real shipped number |
| Agentic / synthetic trajectories | 5% | 0.75T | Natural agentic trace data barely exists at scale; mostly bootstrapped tool-use trajectories |
| Other world languages | 5% | 0.75T | Keeps the model globally usable, roughly preserving LLaMA-3's residual multilingual share |

### 3.3 Why 20% Indic and not more

Native-quality Indic web text is genuinely scarcer than English or code. Past roughly 20%, filling the quota starts forcing in low-quality scrapes or heavy back-translation, which damages fluency more than the added volume helps. This is also the reason the 15T token-budget assumption in §3.1 carries risk: if 3T tokens of genuinely fresh, native-quality Indic text isn't reachable, the plan leans on synthetic augmentation (backtranslation + native-fluency filtering) to close the gap — unproven at this scale, called out explicitly rather than papered over.

---

## 4. Data cleaning

The core problem: almost all off-the-shelf quality, toxicity, and language-ID tooling is trained on English-majority web data. Reusing it as-is doesn't just underperform on Indic languages — it can *actively strip* legitimate content (religious, caste, or regional-political discussion) that gets misread as toxic by classifiers trained on a different culture's notion of what toxic looks like. That directly undermines "India first," so cleaning isn't a generic step — it's redone per language family.

| Step | What | Why it can't just reuse English tooling |
|---|---|---|
| 1. Dedup | Per-language MinHash/LSH, document- and near-dup level | English-tuned similarity thresholds over- or under-merge on Indic scripts |
| 2. Quality filtering | Fresh classifiers trained per language family | A translated English quality classifier imports English notions of "good prose" into languages with different registers |
| 3. Language ID | Custom LID model trained on code-mixed text | Standard LID (fastText, CLD3) misclassifies Hinglish/Tanglish-style code-mixing, which is extremely common in real Indian user input (social media, chat) |
| 4. Code-specific | License-aware filtering, repo-level (not file-level) dedup, AST validity checks, lint/execution-based filtering | Standard for code pipelines, unrelated to the Indic-specific issues above |
| 5. PII scrubbing | Standard patterns + India-specific identifier formats (Aadhaar-like numbers, PAN) | Generic PII regexes miss India-specific ID formats |
| 6. Toxicity/safety | Thresholds calibrated per language with native-speaker annotators | Ported English safety classifiers over-censor legitimate Indic-language discourse |

---

## 5. Post-training (SFT)

The failure mode here is subtler than in pretraining: **machine-translated instruction data carries English framing into grammatically-correct target-language text.** A Hindi instruction translated from an English template can read perfectly fluent while still assuming US/Euro-centric context (currency, units, legal defaults, cultural references) — which quietly defeats "India first" at the semantic level while looking fine on the page. So the rule is that Indic SFT data is majority natively authored or native-reviewed, not machine-translated.

| Slice | Share | Rationale |
|---|---:|---|
| Coding / agentic tool-use | 35% | Largest share — matches the stated coding/agentic priority |
| General instruction-following | 30% | Baseline instruction-tuning coverage |
| Indic-native (authored, not MT'd) | 25% | Sized to be a first-class citizen, not an afterthought |
| India-context reasoning (civics, law, regional knowledge) | 10% | Smallest share, but the one most responsible for the model's *default* frame of reference |

---

## 6. RL / alignment

Two different levers, solving two different problems — worth keeping separate rather than folding into one "RLHF" bucket:

- **RLVR (verifiable rewards) — code & math.** Execution/unit-test pass rate as the reward signal. Language-agnostic, cheap to scale, the same recipe behind the recent DeepSeek/Qwen-class jumps in coding and reasoning benchmarks. This lever moves *capability*.
- **Agentic RL.** Multi-step tool-use environments with outcome-based (task-completion) reward rather than step-level supervision, mirroring the training recipe behind current frontier agentic gains. Also a capability lever.
- **Preference tuning (RLHF/RLAIF) — India-based annotators.** This is the lever that actually encodes "India first" into the model's behavior. A preference model or "constitution" built on US/Euro-centric annotator judgment will encode that framing regardless of how much Indic-language pretraining data came before it — worldview is decided here, not in §3.

---

## 7. Evaluation

Three tiers, because each one catches a failure the others miss:

**Tier 1 — Standard.** MMLU, GSM8K/MATH, HumanEval/MBPP, SWE-bench, BFCL/AgentBench. Table-stakes parity with Gemma-4-class general, coding, and agentic ability — the "is this actually competitive" check.

**Tier 2 — Indic-specific.** IndicXTREME, IndicGLUE, MILU, FLORES, and native-language (not translate-test) reasoning sets. This matters because a model can look strong on translated benchmarks while reasoning poorly *in* the target language — translate-test artifacts hide exactly the gap this whole project is meant to close.

**Tier 3 — Custom / worldview.** Two checks that don't exist as off-the-shelf benchmarks:
- A **fertility audit**: measure realized tokens/word per language on held-out corpora against the §2.2 targets. This is a concrete, checkable number, not a vibe.
- A **default-framing rubric**: score the model's *unprompted* assumptions on ambiguous prompts (currency, dates, holidays, legal system, geography) — whether "India first" is the default, not just something the model can produce when explicitly asked.

Safety and red-teaming run separately across religious, caste, regional, and linguistic axes with Indian annotators, for the same reason cleaning (§4) can't reuse Western toxicity classifiers as-is.

---

## 8. Open questions — not settled, flagged on purpose

- **3T-token native Indic web availability is unverified.** The plan assumes synthetic augmentation (backtranslation + native-fluency filtering) can close the gap between what's readily scraped and the §3.2 target. That assumption itself hasn't been tested at this volume.
- **The 375 tokens/param budget (§3.1) assumes genuinely new tokens, not repeated epochs.** If Indic-corpus scarcity forces multi-epoch training to hit 15T tokens, the real overtraining ratio is worse than the headline number, and repeated-epoch effects on quality are a known open problem in the scaling-laws literature.

---

## 9. Process notes — how this plan changed across drafts

First pass used round, unanchored numbers (e.g., "~12T tokens," vocab size picked by rough intuition). Revised after checking primary sources:

- **Token budget moved from a guessed 11–13T to a cited 15T** once Gemma-3-27B (14T) and LLaMA-3-70B (15T) turned up as near-exact precedents, and the Chinchilla-multiple framing (§3.1 table) gave a way to justify where 40B should sit relative to both.
- **Fertility targets were capped at what Sarvam-1 and Krutrim-1 have actually measured** (1.4–2.1 across their languages) rather than left as an unsupported guess — the Dravidian-language target specifically stops at 2.1 because that's Sarvam-1's real ceiling, and claiming better without a new technique would be a number chosen to look good rather than one backed by evidence.
- **Math and agentic tokenization were originally missing entirely** — the first draft only defined a fertility target for code. Adding them required finding how shipped models actually handle non-code structured tokens: Gemma/Qwen's per-digit number splitting (vs. LLaMA-3's merged digit tokens) for math, and Gemma 4's real 6-token tool-call lifecycle scheme for agentic use — both adopted rather than invented, since these are solved problems.
- **The pretraining mix was re-derived from LLaMA-3's disclosed 50/25/17/8 split** instead of built from scratch, specifically so every departure from it (Indic 8%→20%, code 17%→20%, math 25%→12%) has a stated reason rather than being an arbitrary-looking table.

---

## References

- Gemma Team, "Gemma 3 Technical Report," [arXiv:2503.19786](https://arxiv.org/pdf/2503.19786)
- Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size," [arXiv:2408.00118](https://arxiv.org/pdf/2408.00118)
- Meta AI, "The Llama 3 Herd of Models," [arXiv:2407.21783](https://arxiv.org/html/2407.21783)
- Qwen Team, "Qwen2.5 Technical Report," [arXiv:2412.15115](https://arxiv.org/pdf/2412.15115)
- Qwen Team, "Qwen2.5-Coder Technical Report," [arXiv:2409.12186](https://arxiv.org/pdf/2409.12186)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Sarvam AI, ["Sarvam-1: The First Indian Language LLM"](https://www.sarvam.ai/blogs/sarvam-1)
- Ola Krutrim, "Krutrim LLM: A Novel Tokenization Strategy for Multilingual Indic Languages," [arXiv:2407.12481](https://arxiv.org/html/2407.12481v2)
- "IndicSuperTokenizer: An Optimized Tokenizer for Indic Multilingual LLMs," [arXiv:2511.03237](https://arxiv.org/html/2511.03237v1)
- "IndicGenBench: A Multilingual Benchmark for Indic Languages," [arXiv:2404.16816](https://arxiv.org/pdf/2404.16816)
- Google AI, ["Function calling with Gemma 4"](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)

Figures throughout are planning estimates grounded in the sources above, not measurements from an actual training run.
