// Byte-level BPE encoder in JS — replicates HuggingFace ByteLevel BPE
// (add_prefix_space=True) so token counts match the Python `tokenizers` library.
// Used by the widget to compute token counts live, and by tests/parity in Node.
(function (root) {
  var enc = new TextEncoder();

  // GPT-2 / ByteLevel pre-tokenization regex (same split HF uses by default).
  var PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  // byte (0..255) -> printable "visible byte" char, and its inverse.
  function bytesToUnicode() {
    var bs = [];
    var i;
    for (i = 33; i <= 126; i++) bs.push(i);
    for (i = 161; i <= 172; i++) bs.push(i);
    for (i = 174; i <= 255; i++) bs.push(i);
    var cs = bs.slice();
    var n = 0;
    for (var b = 0; b < 256; b++) {
      if (bs.indexOf(b) === -1) { bs.push(b); cs.push(256 + n); n++; }
    }
    var b2u = {}, u2b = {};
    for (i = 0; i < bs.length; i++) {
      var ch = String.fromCharCode(cs[i]);
      b2u[bs[i]] = ch;
      u2b[ch] = bs[i];
    }
    return { b2u: b2u, u2b: u2b };
  }

  var MAP = bytesToUnicode();

  function toVisible(piece) {
    var bytes = enc.encode(piece);
    var s = "";
    for (var i = 0; i < bytes.length; i++) s += MAP.b2u[bytes[i]];
    return s;
  }

  // Build a tokenizer object from vocab {token:id} and merges (array of "A B" strings).
  function build(vocab, mergesLines) {
    var ranks = new Map();
    var rank = 0;
    for (var i = 0; i < mergesLines.length; i++) {
      var ln = mergesLines[i];
      if (!ln || ln.charAt(0) === "#") continue;
      ranks.set(ln, rank++);       // key exactly as "A B"
    }
    return { vocab: vocab, ranks: ranks, cache: new Map() };
  }

  // classic BPE over one pre-token's visible-byte string
  function bpe(token, t) {
    if (t.cache.has(token)) return t.cache.get(token);
    var word = Array.from(token);
    if (word.length <= 1) { t.cache.set(token, word); return word; }
    while (true) {
      var minRank = Infinity, mi = -1;
      for (var i = 0; i < word.length - 1; i++) {
        var r = t.ranks.get(word[i] + " " + word[i + 1]);
        if (r !== undefined && r < minRank) { minRank = r; mi = i; }
      }
      if (mi === -1) break;
      var a = word[mi], b = word[mi + 1];
      var out = [];
      var j = 0;
      while (j < word.length) {
        if (j < word.length - 1 && word[j] === a && word[j + 1] === b) {
          out.push(a + b); j += 2;
        } else { out.push(word[j]); j += 1; }
      }
      word = out;
      if (word.length === 1) break;
    }
    t.cache.set(token, word);
    return word;
  }

  // Encode text -> array of token strings (add_prefix_space=True behavior).
  function encode(text, t) {
    if (text.length && !/^\s/.test(text)) text = " " + text; // prefix space
    var tokens = [];
    var m;
    PAT.lastIndex = 0;
    while ((m = PAT.exec(text)) !== null) {
      var visible = toVisible(m[0]);
      var pieces = bpe(visible, t);
      for (var k = 0; k < pieces.length; k++) tokens.push(pieces[k]);
    }
    return tokens;
  }

  function countTokens(text, t) {
    return encode(text, t).length;
  }

  // visible-byte token -> readable UTF-8 (for the token viewer)
  function decodeToken(token) {
    var bytes = [];
    for (var i = 0; i < token.length; i++) {
      var b = MAP.u2b[token[i]];
      if (b === undefined) return token;
      bytes.push(b);
    }
    try {
      return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
    } catch (e) { return token; }
  }

  var api = { build: build, encode: encode, countTokens: countTokens, decodeToken: decodeToken };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.BPE = api;
})(typeof window !== "undefined" ? window : globalThis);
