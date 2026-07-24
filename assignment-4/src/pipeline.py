"""
Session 4 cleaning pipeline, run for real against a real sample pulled from
AI4Bharat Sangraha (verified split, CC-BY-4.0).

Eight stages, in the order the lesson specifies: Extract -> Normalize ->
Language ID -> Quality filter -> Deduplicate -> PII scrub -> Decontaminate ->
Manifest. Every count in stats.json below comes from actually running these
stages over data/raw_sample.jsonl - nothing is hand-typed.
"""
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 42  # deterministic langdetect

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_SAMPLE = DATA / "raw_sample.jsonl"

# --- claimed folder code -> ISO 639-1 code langdetect uses -----------------
CLAIM_TO_ISO = {
    "eng": "en", "hin": "hi", "tel": "te", "ben": "bn", "mal": "ml",
    "tam": "ta", "guj": "gu", "mar": "mr", "urd": "ur", "pan": "pa",
    "kan": "kn", "asm": "as",
}
# langdetect's supported set (ISO 639-1); langs outside this can never "match"
LANGDETECT_SUPPORTED = {
    "en", "hi", "te", "bn", "ml", "ta", "gu", "mr", "ur", "pa", "kn",
}

# Small, real per-language stop-word samples (enough to test presence, not
# exhaustive). Absence of a list means we DO NOT apply the stop-word rule for
# that language, rather than silently scoring it against English's list --
# this is the exact script-aware fix Session 4 called out.
STOPWORDS = {
    "en": {"the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "to", "in", "on", "for", "with", "that", "this"},
    "hi": {"है", "और", "के", "की", "का", "में", "से", "को", "यह", "था", "थी", "हैं", "पर"},
    "bn": {"এবং", "এর", "কি", "না", "হয়", "সাথে", "থেকে", "একটি", "ছিল"},
}

ZERO_WIDTH_NOISE = "​﻿‪‫‬‭‮⁠"
INDIC_JOINERS = "‌‍"  # ZWNJ, ZWJ -- must survive normalization
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = html.unescape(s)
    s = CONTROL_RE.sub("", s)
    for ch in ZERO_WIDTH_NOISE:
        s = s.replace(ch, "")
    s = s.replace("�", "")  # replacement char (corrupt bytes)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def looks_like_markup(s: str) -> bool:
    """Stage 1 (Extract) validation: Sangraha ships pre-extracted prose, so
    we just confirm no HTML/markup residue leaked through extraction."""
    tag_hits = len(re.findall(r"</?[a-zA-Z][^>]{0,30}>", s))
    return tag_hits > 0 or ("&amp;" in s or "&lt;" in s or "&gt;" in s)


def word_tokenize(s: str):
    return re.findall(r"\S+", s)


def is_script_char(c: str) -> bool:
    """True for letters, digits, AND combining marks (Unicode category M*).
    Indic scripts encode vowel signs (matras) and viramas as combining marks,
    e.g. U+094B DEVANAGARI VOWEL SIGN O -- str.isalnum() returns False for
    these, which would silently misclassify most Indic text as symbol-heavy.
    This is the exact English-centric-heuristic bug Session 4 called out."""
    if c.isspace():
        return False
    cat = unicodedata.category(c)
    return cat[0] in ("L", "M", "N")


def quality_signals(text: str, lang_iso: str):
    words = word_tokenize(text)
    n_words = len(words)
    if n_words == 0:
        return None
    mean_word_len = sum(len(w) for w in words) / n_words
    symbol_chars = sum(1 for c in text if not is_script_char(c) and not c.isspace())
    symbol_ratio = symbol_chars / max(len(text), 1)
    lines = [l for l in text.split("\n") if l.strip()]
    end_punct = sum(1 for l in lines if l.strip()[-1:] in ".!?।॥۔؟\"'”’")
    end_punct_ratio = (end_punct / len(lines)) if lines else 0.0
    dup_lines = len(lines) - len(set(lines))
    dup_line_frac = (dup_lines / len(lines)) if lines else 0.0
    bigrams = Counter(zip(words, words[1:])) if n_words > 1 else Counter()
    top2 = bigrams.most_common(1)
    top2_frac = (top2[0][1] * 2 / n_words) if top2 else 0.0

    stop_list = STOPWORDS.get(lang_iso)
    if stop_list is not None:
        stop_hits = sum(1 for w in words if w.strip(".,!?;:\"'()").lower() in stop_list)
        stopword_rule_applied = True
        stopword_pass = stop_hits >= 2
    else:
        stopword_rule_applied = False
        stopword_pass = True  # no list for this language -> don't penalize it

    checks = {
        "mean_word_length_3_10": 3 <= mean_word_len <= 10,
        "symbol_ratio_lt_0.10": symbol_ratio < 0.10,
        "lines_end_punct_ge_0.30": end_punct_ratio >= 0.30 if lines else True,
        "dup_line_frac_lt_0.30": dup_line_frac < 0.30,
        "top2gram_frac_lt_0.20": top2_frac < 0.20,
        "stopwords_present": stopword_pass,
        "word_count_50_100000": 50 <= n_words <= 100000,
    }
    passed = sum(checks.values())
    return {
        "n_words": n_words,
        "mean_word_len": round(mean_word_len, 2),
        "symbol_ratio": round(symbol_ratio, 3),
        "end_punct_ratio": round(end_punct_ratio, 3),
        "dup_line_frac": round(dup_line_frac, 3),
        "top2gram_frac": round(top2_frac, 3),
        "stopword_rule_applied": stopword_rule_applied,
        "checks": checks,
        "n_pass": passed,
        "n_total": len(checks),
        "kept": passed == len(checks),
    }


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3,4}\)?[-.\s]?){2,3}\d{3,4}")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Illustrative name heuristic (Latin-script only, mirrors the lesson's
# explicit "precision-vs-recall, illustrative" framing for the NER layer):
# two-or-more consecutive Capitalized words not at the start of a sentence.
NAME_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b")


def pii_scrub(text: str):
    counts = {"email": 0, "phone": 0, "ipv4": 0, "name_like": 0}
    redacted = text

    def repl(tag):
        def _r(m):
            counts[tag] += 1
            return f"[{tag.upper()}]"
        return _r

    redacted = EMAIL_RE.sub(repl("email"), redacted)
    redacted = IPV4_RE.sub(repl("ipv4"), redacted)
    redacted = PHONE_RE.sub(lambda m: repl("phone")(m) if len(re.sub(r"\D", "", m.group())) >= 7 else m.group(), redacted)
    redacted = NAME_RE.sub(repl("name_like"), redacted)
    return redacted, counts


def shingles(text: str, k=5):
    words = word_tokenize(text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 0))} or {text.lower()}


CANARIES = [
    "benchmark canary guid 8f31c02e this exact sentence must never appear in any training corpus",
    "benchmark canary guid 51ac77db this exact sentence must never appear in any training corpus",
]


def ngram_overlap(text: str, canary: str, n=8):
    t_words = word_tokenize(text.lower())
    c_words = word_tokenize(canary.lower())
    if len(c_words) < n:
        return False
    c_ngrams = {tuple(c_words[i:i + n]) for i in range(len(c_words) - n + 1)}
    t_ngrams = {tuple(t_words[i:i + n]) for i in range(len(t_words) - n + 1)}
    return bool(c_ngrams & t_ngrams)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    rows = [json.loads(l) for l in RAW_SAMPLE.read_text(encoding="utf-8").splitlines()]
    n_raw = len(rows)
    stats = {"pipeline_version": "session4-v1", "run_ts": datetime.now(timezone.utc).isoformat(),
              "stages": {}, "per_language": defaultdict(lambda: {"raw": 0, "final": 0})}

    for r in rows:
        stats["per_language"][r["_lang_claimed"]]["raw"] += 1

    # ---------------- Stage 1: Extract (validate, no HTML residue) --------
    stage1_survivors = []
    markup_found = 0
    for r in rows:
        if looks_like_markup(r["text"]):
            markup_found += 1
        stage1_survivors.append(r)  # Sangraha text is pre-extracted; nothing dropped here
    stats["stages"]["1_extract"] = {
        "in": n_raw, "out": len(stage1_survivors),
        "docs_with_markup_residue_found": markup_found,
        "note": "Sangraha ships already-extracted prose; this stage validates no HTML/entity residue leaked through, it does not drop documents.",
    }

    # ---------------- Stage 2: Normalize -----------------------------------
    stage2_survivors = []
    chars_stripped_total = 0
    joiners_preserved = 0
    for r in stage1_survivors:
        before = r["text"]
        after = clean_text(before)
        chars_stripped_total += max(len(before) - len(after), 0)
        joiners_preserved += sum(after.count(c) for c in INDIC_JOINERS)
        if after:
            r2 = dict(r)
            r2["text"] = after
            stage2_survivors.append(r2)
    stats["stages"]["2_normalize"] = {
        "in": len(stage1_survivors), "out": len(stage2_survivors),
        "chars_stripped_total": chars_stripped_total,
        "indic_joiners_preserved_zwnj_zwj": joiners_preserved,
    }

    # ---------------- Stage 3: Language ID validation ----------------------
    stage3_survivors = []
    mismatches = []
    undetectable = 0
    for r in stage2_survivors:
        claimed_iso = CLAIM_TO_ISO.get(r["_lang_claimed"])
        detected = None
        try:
            detected = detect(r["text"][:2000])
        except LangDetectException:
            detected = None
        r["_lang_detected"] = detected
        if claimed_iso not in LANGDETECT_SUPPORTED:
            r["_lang_status"] = "unsupported_by_detector"
            stage3_survivors.append(r)
            undetectable += 1
        elif detected == claimed_iso:
            r["_lang_status"] = "match"
            stage3_survivors.append(r)
        else:
            r["_lang_status"] = "mismatch"
            mismatches.append({"doc_id": r.get("doc_id"), "claimed": claimed_iso, "detected": detected})
            # quarantined, not silently kept
    stats["stages"]["3_language_id"] = {
        "in": len(stage2_survivors), "out": len(stage3_survivors),
        "mismatches_quarantined": len(mismatches),
        "unsupported_by_detector_passed_through": undetectable,
        "example_mismatches": mismatches[:8],
    }

    # ---------------- Stage 4: Quality filter -------------------------------
    stage4_survivors = []
    dropped_by_lang = defaultdict(int)
    quality_examples_dropped = []
    for r in stage3_survivors:
        iso = CLAIM_TO_ISO.get(r["_lang_claimed"])
        q = quality_signals(r["text"], iso)
        if q is None:
            dropped_by_lang[r["_lang_claimed"]] += 1
            continue
        r["_quality"] = q
        if q["kept"]:
            stage4_survivors.append(r)
        else:
            dropped_by_lang[r["_lang_claimed"]] += 1
            if len(quality_examples_dropped) < 8:
                failed = [k for k, v in q["checks"].items() if not v]
                quality_examples_dropped.append({"doc_id": r.get("doc_id"), "lang": r["_lang_claimed"], "failed_checks": failed})
    stats["stages"]["4_quality_filter"] = {
        "in": len(stage3_survivors), "out": len(stage4_survivors),
        "dropped_by_language": dict(dropped_by_lang),
        "example_dropped": quality_examples_dropped,
        "note": "Stop-word rule only applied for languages with a real stop-word list (en/hi/bn here); other languages skip that rule rather than being judged against English's list.",
    }

    # ---------------- Stage 5: Deduplicate (MinHash + LSH) -----------------
    lsh = MinHashLSH(threshold=0.75, num_perm=112)
    minhashes = {}
    for r in stage4_survivors:
        m = MinHash(num_perm=112)
        for sh in shingles(r["text"], k=5):
            m.update(sh.encode("utf-8"))
        minhashes[r["doc_id"]] = m

    stage5_survivors = []
    seen_dupe_of = {}
    removed_dupes = []
    for r in stage4_survivors:
        did = r["doc_id"]
        if did in seen_dupe_of:
            continue
        result = lsh.query(minhashes[did])
        lsh.insert(did, minhashes[did])
        for other in result:
            if other != did:
                seen_dupe_of[other] = did
        stage5_survivors.append(r)
    # second pass: drop anything that ended up flagged as a dupe of something already kept
    final5 = [r for r in stage5_survivors if r["doc_id"] not in seen_dupe_of]
    removed_dupes = list(seen_dupe_of.items())[:8]
    stats["stages"]["5_deduplicate"] = {
        "in": len(stage4_survivors), "out": len(final5),
        "near_or_exact_duplicates_removed": len(seen_dupe_of),
        "method": "MinHash (num_perm=112, k=5 word shingles) + LSH banding, threshold 0.75",
        "example_duplicate_pairs": removed_dupes,
    }
    stage5_survivors = final5

    # ---------------- Stage 6: PII scrub ------------------------------------
    stage6_survivors = []
    pii_totals = Counter()
    for r in stage5_survivors:
        redacted, counts = pii_scrub(r["text"])
        r["text"] = redacted
        for k, v in counts.items():
            pii_totals[k] += v
        stage6_survivors.append(r)
    stats["stages"]["6_pii_scrub"] = {
        "in": len(stage5_survivors), "out": len(stage6_survivors),
        "redactions": dict(pii_totals),
        "note": "Regex layer (email/phone/ipv4) is exact-pattern, near-zero false positives. Name layer is an illustrative Latin-script heuristic only -- it does not run on Indic scripts, which is the precision/recall gap Session 4 flagged for Indic NER.",
    }

    # ---------------- Stage 7: Decontaminate --------------------------------
    flagged = []
    for r in stage6_survivors:
        for c in CANARIES:
            if ngram_overlap(r["text"], c, n=8):
                flagged.append(r.get("doc_id"))
    # self-test: prove the scanner actually works by injecting one canary into a scratch copy
    test_doc = stage6_survivors[0]["text"] + " " + CANARIES[0] if stage6_survivors else CANARIES[0]
    self_test_caught = ngram_overlap(test_doc, CANARIES[0], n=8)
    stats["stages"]["7_decontaminate"] = {
        "in": len(stage6_survivors), "out": len(stage6_survivors) - len(flagged),
        "canaries_checked": len(CANARIES),
        "real_contamination_found": len(flagged),
        "self_test_injected_canary_detected": self_test_caught,
        "note": "Held-out canary strings never appear in Sangraha; the self-test proves the n-gram scanner would catch them if they did.",
    }
    stage7_survivors = [r for r in stage6_survivors if r.get("doc_id") not in flagged]

    # ---------------- Stage 8: Manifest -------------------------------------
    pipeline_src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    cleaning_script_hash = sha256(pipeline_src)
    manifests = []
    lang_token_totals = Counter()
    for r in stage7_survivors:
        text = r["text"]
        token_count = len(word_tokenize(text))  # whitespace-token proxy, NOT a real BPE count -- flagged as such
        lang_token_totals[r["_lang_claimed"]] += token_count
        manifests.append({
            "shard_id": f"sangraha_{r['_lang_claimed']}_{r['doc_id'][:10]}",
            "source_url": f"https://huggingface.co/datasets/ai4bharat/sangraha/viewer/verified/{r['_lang_claimed']}",
            "license_class": "CC-BY-4.0",
            "contributor_id": "ai4bharat",
            "cleaning_script": "pipeline.py",
            "cleaning_script_hash": cleaning_script_hash[:16],
            "ingest_ts": stats["run_ts"],
            "sha256": sha256(text)[:16],
            "token_count_proxy": token_count,
            "lang_claimed": r["_lang_claimed"],
            "lang_detected": r.get("_lang_detected"),
        })
    stats["stages"]["8_manifest"] = {
        "in": len(stage7_survivors), "out": len(stage7_survivors),
        "manifests_emitted": len(manifests),
        "cleaning_script_hash": cleaning_script_hash[:16],
        "token_count_caveat": "Token counts here are whitespace-split proxies, not a real BPE tokenizer count -- deliberately NOT repeating V4's flawed words*1.3 estimate.",
    }

    for r in stage7_survivors:
        stats["per_language"][r["_lang_claimed"]]["final"] += 1

    stats["totals"] = {
        "raw_sample_rows": n_raw,
        "final_rows": len(stage7_survivors),
        "survival_rate": round(len(stage7_survivors) / n_raw, 4),
        "total_token_proxy": sum(lang_token_totals.values()),
    }
    stats["per_language"] = dict(stats["per_language"])

    (DATA / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA / "manifests_sample.json").write_text(json.dumps(manifests[:20], indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stats["totals"], indent=2))
    print("\nPer-stage in/out:")
    for name, s in stats["stages"].items():
        print(f"  {name}: {s['in']} -> {s['out']}")


if __name__ == "__main__":
    main()
