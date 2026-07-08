// Shared preprocessing — MUST mirror src/preprocess.py exactly.
// Same regexes, same NFC, so the widget's word counts match score.py.
(function (root) {
  function preprocess(text) {
    text = text.normalize("NFC");
    text = text.replace(/\[\s*\d+\s*\]/g, "");   // drop [1]-style ref markers
    text = text.replace(/[^\S\n]+/g, " ");        // collapse horizontal whitespace
    text = text.split("\n").map(function (l) { return l.trim(); }).join("\n");
    text = text.replace(/\n{3,}/g, "\n\n");       // tidy blank lines
    return text.trim();
  }

  function wordCount(text) {
    var m = text.match(/\S+/g);
    return m ? m.length : 0;
  }

  var api = { preprocess: preprocess, wordCount: wordCount };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.Preprocess = api;
})(typeof window !== "undefined" ? window : globalThis);
