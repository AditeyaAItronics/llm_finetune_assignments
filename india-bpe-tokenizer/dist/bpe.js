// Character-level BPE encoder in JS — replicates the HuggingFace `tokenizers`
// tokenizer built by src/train_balance.py:
//   * model  = BPE(byte_fallback=True)     -> unseen chars fall back to <0xXX> byte tokens
//   * pre_tok= Metaspace()                  -> each word becomes one "▁word" pre-token
//   * fertility is measured on whitespace-normalized text (one ▁word per word),
//     so this matches Python `tok.encode(normalize_spaces(text))` exactly.
// Used by the widget to compute token counts live, and by tests/parity in Node.
(function (root) {
  var enc = new TextEncoder();

  // Build a tokenizer from vocab {token:id} and merges (array of "A B" strings).
  // Tokens never contain a literal space (word starts are marked with ▁ = U+2581,
  // byte tokens look like "<0x20>"), so "A B" splits unambiguously and we can key
  // the rank map on the whole line.
  function build(vocab, mergesLines) {
    var ranks = new Map();
    var rank = 0;
    for (var i = 0; i < mergesLines.length; i++) {
      var ln = mergesLines[i];
      if (!ln || ln.charAt(0) === "#") continue;
      ranks.set(ln, rank++);
    }
    return { vocab: vocab, ranks: ranks, cache: new Map() };
  }

  // A single character -> its UTF-8 byte tokens, e.g. "☃" -> ["<0xE2>","<0x98>","<0x83>"].
  function byteFallback(ch) {
    var bytes = enc.encode(ch);
    var out = [];
    for (var i = 0; i < bytes.length; i++) {
      out.push("<0x" + bytes[i].toString(16).toUpperCase().padStart(2, "0") + ">");
    }
    return out;
  }

  // Initial symbols for a word: prepend the Metaspace marker, split into characters,
  // and byte-fallback any character that isn't in the vocab (zero UNK).
  function initialSymbols(word, t) {
    var chars = Array.from("▁" + word); // ▁word
    var syms = [];
    for (var i = 0; i < chars.length; i++) {
      var c = chars[i];
      if (Object.prototype.hasOwnProperty.call(t.vocab, c)) syms.push(c);
      else {
        var bf = byteFallback(c);
        for (var j = 0; j < bf.length; j++) syms.push(bf[j]);
      }
    }
    return syms;
  }

  // Greedy BPE over an array of symbol strings: repeatedly merge the lowest-rank
  // adjacent pair (all its occurrences), exactly as HF applies merges.
  function mergeSymbols(syms, t) {
    if (syms.length <= 1) return syms;
    while (true) {
      var minRank = Infinity, mi = -1;
      for (var i = 0; i < syms.length - 1; i++) {
        var r = t.ranks.get(syms[i] + " " + syms[i + 1]);
        if (r !== undefined && r < minRank) { minRank = r; mi = i; }
      }
      if (mi === -1) break;
      var a = syms[mi], b = syms[mi + 1];
      var out = [];
      var j = 0;
      while (j < syms.length) {
        if (j < syms.length - 1 && syms[j] === a && syms[j + 1] === b) { out.push(a + b); j += 2; }
        else { out.push(syms[j]); j += 1; }
      }
      syms = out;
      if (syms.length === 1) break;
    }
    return syms;
  }

  function encodeWord(word, t) {
    if (t.cache.has(word)) return t.cache.get(word);
    var syms = mergeSymbols(initialSymbols(word, t), t);
    t.cache.set(word, syms);
    return syms;
  }

  // Encode text -> array of token strings. Whitespace is normalized to words first
  // (one ▁word pre-token per word), matching preprocess.normalize_spaces in Python.
  function encode(text, t) {
    var words = text.match(/\S+/g) || [];
    var tokens = [];
    for (var k = 0; k < words.length; k++) {
      var syms = encodeWord(words[k], t);
      for (var m = 0; m < syms.length; m++) tokens.push(syms[m]);
    }
    return tokens;
  }

  function countTokens(text, t) {
    return encode(text, t).length;
  }

  // token -> readable text for the viewer: ▁ becomes a leading space; byte-fallback
  // fragments (not printable on their own) are shown verbatim.
  function decodeToken(token) {
    if (token.indexOf("<0x") === 0 && token.charAt(token.length - 1) === ">") return token;
    return token.replace(/▁/g, " ");
  }

  var api = { build: build, encode: encode, countTokens: countTokens, decodeToken: decodeToken };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.BPE = api;
})(typeof window !== "undefined" ? window : globalThis);
