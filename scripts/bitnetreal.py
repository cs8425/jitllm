#!/usr/bin/env python3
"""Record transformers' logits on a trained BitNet b1.58 checkpoint.

    $JITLLM_HF_PY scripts/bitnetreal.py <checkpoint dir> [name]

-> testdata/golden/bitnet/real-<name>.json (name defaults to the directory's)

The reference is transformers' BitNetForCausalLM through its own "bitnet"
quantizer, run in float32 on the checkpoint as it ships: for
microsoft/bitnet-b1.58-2B-4T the uint8 codes and the bf16 weight_scale each
projection is multiplied by (autobitlinear, offline), every projection's input
quantized to int8 per token. Each prompt is run greedy for REAL_GEN tokens;
then the prompt and its continuation once more, teacher-forced, and every
position's logits recorded as the vocabulary's first REAL_HEAD and the
reference's REAL_TOP highest -- k3gold.py's real-checkpoint schema, which
engine/model's TestBitNetRealMatchesReference reads.

The prompts carry the tokenizer's BOS (<|begin_of_text|>), as the release's
own examples do.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "testdata", "golden", "bitnet")
REAL_GEN, REAL_HEAD, REAL_TOP = 32, 256, 64
TEXTS = ["The capital of France is", "def fibonacci(n):\n", "In 1969, the first person to walk on the Moon"]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.normpath(d))
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
    m.config._attn_implementation = "eager"
    m.eval()
    tk = AutoTokenizer.from_pretrained(d)
    seqs = []
    with torch.no_grad():
        for text in TEXTS:
            prompt = tk(text).input_ids
            ids = list(prompt)
            for _ in range(REAL_GEN):
                ids.append(int(m(torch.tensor([ids])).logits[0, -1].argmax()))
            lg = m(torch.tensor([ids[:-1]])).logits[0]
            rows = []
            for r in lg:
                v, i = r.topk(REAL_TOP)
                rows.append({"argmax": int(r.argmax()), "max": float(r.max()),
                             "head": [round(float(x), 6) for x in r[:REAL_HEAD]],
                             "top": [int(x) for x in i], "topv": [round(float(x), 6) for x in v]})
            cont = ids[len(prompt):]
            print(f"{text!r} -> {tk.decode(cont)!r}")
            seqs.append({"text": text, "prompt": prompt, "cont": cont, "reply": tk.decode(cont),
                         "rows": rows})
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "real-" + name + ".json")
    json.dump({"dtype": "float32", "vocab": m.config.vocab_size, "seqs": seqs}, open(dst, "w"))
    print("wrote", dst)


if __name__ == "__main__":
    main()
