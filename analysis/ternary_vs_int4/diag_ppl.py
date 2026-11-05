"""POST-HOC diagnostic (EXP-0011, not part of the preregistered decision): does the BitNet I2_S GGUF under bitnet.cpp reproduce the numerics of the
official checkpoint? Computes perplexity over identical text with (a) the HF checkpoint (GPU, bf16) replicating llama-perplexity's protocol
(per-chunk BOS, chunk 512, score the second half of each chunk) and (b) llama-perplexity on the GGUF. A large gap means the GGUF path is degraded;
agreement means low task scores in the GGUF eval come from the scoring format, not from the numerics. Output: results/ternary_vs_int4/diagnostic/.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import config as C

OUT = Path(__file__).resolve().parents[2] / "results" / "ternary_vs_int4" / "diagnostic"
CHUNK, NCHUNKS = 512, 20
WIKI_URL = "https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/wiki.test.raw.gz"


def wiki_text() -> Path:
    p = C.DATA_DIR / "wiki.test.raw"
    if not p.exists():
        raw = urllib.request.urlopen(WIKI_URL, timeout=120).read()
        p.write_bytes(gzip.decompress(raw))
    return p


def gguf_ppl(model_file: str, text: Path) -> dict:
    cmd = [str(C.RUNTIME_BIN["bitnet"] / "llama-perplexity"), "-m", str(C.MODELS_DIR / model_file), "-f", str(text), "-c", str(CHUNK), "--chunks", str(NCHUNKS),
           "-t", "8", "-ngl", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"Final estimate: PPL = ([\d.]+)", out)
    return {"cmd": cmd, "ppl": float(m.group(1)) if m else None, "tail": out[-600:]}


def hf_ppl(text: Path) -> dict:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    name = "microsoft/bitnet-b1.58-2B-4T"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    ids = tok(text.read_text(encoding="utf8"), add_special_tokens=False)["input_ids"]
    bos = tok.bos_token_id
    nll, cnt = 0.0, 0
    for i in range(NCHUNKS):
        chunk = ids[i * CHUNK:(i + 1) * CHUNK]
        if len(chunk) < CHUNK:
            break
        chunk = [bos] + chunk[1:]                      # llama-perplexity overwrites the first token of each chunk with BOS
        x = torch.tensor([chunk]).cuda()
        with torch.no_grad():
            logits = model(x).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        for j in range(CHUNK // 2 - 1, CHUNK - 1):     # second half, as llama-perplexity
            nll -= lp[j, chunk[j + 1]].item()
            cnt += 1
    return {"ppl": math.exp(nll / cnt), "tokens_scored": cnt, "bos_token_id": bos}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = wiki_text()
    res = {"chunk": CHUNK, "chunks": NCHUNKS, "text_sha_note": "ikawrakow wiki.test.raw"}
    res["gguf_bitnet_i2s_bitnetcpp"] = gguf_ppl(C.MODELS[C.TERNARY]["file"], text)
    res["gguf_qwen25_1p5b_q4km_bitnetcpp"] = gguf_ppl(C.MODELS["QWEN25_1P5B_Q4KM"]["file"], text)
    res["hf_bitnet_bf16"] = hf_ppl(text)
    (OUT / "ppl_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("ppl") if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
