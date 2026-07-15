"""JS (widget/bpe.js) must produce identical token counts to Python `tokenizers`.

Self-contained: trains a small tokenizer, then runs the widget's JS encoder in Node
over the same strings and asserts the counts match exactly. This is what guarantees
the widget's live numbers equal score.py's.
"""
import json
import shutil
import subprocess
import tempfile

import pytest

import _helpers as H

SAMPLES = [
    "भारत एक विशाल देश है और यहाँ की संस्कृति प्राचीन है।",
    "India is a country. The capital is New Delhi in 1947.",
    "భారత దేశం దక్షిణ ఆసియాలో ఒక పెద్ద దేశం మరియు ఇక్కడ చాలా భాషలు ఉన్నాయి.",
    "ভারত দক্ষিণ এশিয়ার একটি বৃহৎ দেশ এবং এর সংস্কৃতি প্রাচীন।",
    "Mixed: हिन्दी తెలుగు বাংলা English 12345 !!!",
]

TRAIN = [
    "भारत एक विशाल देश है।", "India is a country in South Asia.",
    "భారత దేశం దక్షిణ ఆసియాలో ఒక దేశం.", "ভারত দক্ষিণ এশিয়ার একটি দেশ।",
    "GDP 1947 New Delhi — capital city.", "हिन्दी, తెలుగు, বাংলা",
] * 40


def test_js_python_parity():
    if shutil.which("node") is None:
        pytest.skip("node not available")

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.trainers import BpeTrainer

    # Same recipe as src/train_balance.py: char-level BPE + byte-fallback + Metaspace.
    tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = Metaspace()
    trainer = BpeTrainer(vocab_size=800,
                         special_tokens=[f"<0x{i:02X}>" for i in range(256)],
                         show_progress=False)
    tok.train_from_iterator(iter(TRAIN), trainer=trainer)

    d = tempfile.mkdtemp()
    tok.model.save(d)
    # Fertility is measured on whitespace-normalized text (one ▁word per word),
    # which is exactly what bpe.js's countTokens does internally.
    norm = [" ".join(s.split()) for s in SAMPLES]
    (open(f"{d}/samples.json", "w", encoding="utf-8")
     .write(json.dumps(SAMPLES, ensure_ascii=False)))
    py_counts = [len(tok.encode(s).ids) for s in norm]

    bpe_js = H.ROOT / "widget" / "bpe.js"
    node = f"""
      const fs=require("fs");
      const BPE=require({json.dumps(str(bpe_js))});
      const vocab=JSON.parse(fs.readFileSync({json.dumps(d + "/vocab.json")},"utf8"));
      const merges=fs.readFileSync({json.dumps(d + "/merges.txt")},"utf8").split("\\n");
      const t=BPE.build(vocab,merges);
      const s=JSON.parse(fs.readFileSync({json.dumps(d + "/samples.json")},"utf8"));
      console.log(JSON.stringify(s.map(x=>BPE.countTokens(x,t))));
    """
    js_counts = json.loads(subprocess.check_output(["node", "-e", node]).decode())
    assert js_counts == py_counts, (js_counts, py_counts)
