#!/usr/bin/env python3
"""LFM2-VL's goldens (d1-3B's tower), from the reference itself.

    $JITLLM_HF_PY scripts/lfm2visiongold.py --model-free
        -> testdata/golden/lfm2-layout.json: Lfm2VlImageProcessor's plan for
           many picture sizes (tiles, rows, columns and the smart-resized
           size), and the processor's expanded image tokens for each.

    $JITLLM_HF_PY scripts/lfm2visiongold.py DIR OUT IMAGE...
        -> OUT/: per picture, the processor's pixels (every piece,
           [y][x][channel], back to back), the tower's last_hidden_state
           (post-LayerNorm, each piece unpadded) and the projector's rows,
           from the checkpoint in DIR (d1-3B, Lfm2VlForConditionalGeneration)
           in float32; and OUT/decisions.json, d1's own SystemOne
           probabilities over the pictures (its runner.py, float32).

The reference is transformers' Lfm2VlProcessor (torchvision's uint8 resize,
the class's tiling) and Lfm2VlModel.get_image_features; the decisions are the
checkpoint's own code (runner.py: cap_pixels, prefix_text, the readout).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [(640, 488), (896, 896), (384, 384), (100, 100), (3000, 2000), (4000, 300), (300, 4000),
         (1920, 1080), (2358, 844), (1272, 1398), (257, 255), (16, 3000), (961, 959), (1024, 1024),
         (500, 333), (512, 512), (725, 724), (1100, 700), (640, 512), (2048, 64), (33, 33)]


def processor(d=None):
    from transformers import Lfm2VlImageProcessor
    if d:
        cfg = json.load(open(os.path.join(d, "processor_config.json")))["image_processor"]
        cfg.pop("image_processor_type", None)
        return Lfm2VlImageProcessor(**cfg)
    return Lfm2VlImageProcessor(resample=3)


def plan(ip, img):
    out = ip(images=[[img]], return_tensors="pt", return_row_col_info=True)
    return out


def model_free():
    from PIL import Image
    ip = processor()
    rows = []
    for w, h in SIZES:
        o = plan(ip, Image.new("RGB", (w, h), (90, 140, 200)))
        shapes = o["spatial_shapes"].tolist()
        rows.append({"w": w, "h": h, "rows": int(o["image_rows"][0]), "cols": int(o["image_cols"][0]),
                     "size": [int(x) for x in o["image_sizes"][0]], "pieces": shapes})
    dst = os.path.join(HERE, "..", "testdata", "golden", "lfm2-layout.json")
    json.dump({"sizes": rows}, open(dst, "w"), indent=0)
    print("wrote", dst, len(rows), "sizes")


def pieces_hwc(o, patch=16):
    """The processor's pixel_values, [n][patches][ky*kx*c] padded, as each
    piece's [y][x][c], unpadded."""
    out = []
    for i, (gh, gw) in enumerate(o["spatial_shapes"].tolist()):
        n = gh * gw
        p = o["pixel_values"][i, :n].reshape(gh, gw, patch, patch, 3).permute(0, 2, 1, 3, 4)
        out.append(p.reshape(gh * patch, gw * patch, 3).numpy().astype(np.float32))
    return out


def checkpoint(d, out, images):
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    ip = processor(d)
    lf = d1ref(d, "lfm2_vl")
    # The checkpoint's class is HF's Lfm2VlForConditionalGeneration with its
    # language model swapped for the hybrid stack; the tower and projector
    # are HF's own.
    m = lf.Lfm2VlForConditionalGeneration.from_pretrained(d, dtype=torch.float32, attn_implementation="sdpa")
    m.eval()
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        o = ip(images=[[img]], return_tensors="pt", return_row_col_info=True)
        px = pieces_hwc(o)
        with torch.no_grad():
            vo = m.model.vision_tower(pixel_values=o["pixel_values"], spatial_shapes=o["spatial_shapes"],
                                      pixel_attention_mask=o["pixel_attention_mask"])
            lens = o["pixel_attention_mask"].sum(1).tolist()
            vit = torch.cat([vo.last_hidden_state[i, :n] for i, n in enumerate(lens)])
            emb = torch.cat(m.model.get_image_features(pixel_values=o["pixel_values"],
                                                       spatial_shapes=o["spatial_shapes"],
                                                       pixel_attention_mask=o["pixel_attention_mask"]).pooler_output)
        base = os.path.join(out, name)
        np.concatenate([p.reshape(-1) for p in px]).tofile(base + ".px")
        vit.numpy().astype(np.float32).tofile(base + ".vit")
        emb.numpy().astype(np.float32).tofile(base + ".emb")
        meta = {"w": img.size[0], "h": img.size[1], "rows": int(o["image_rows"][0]),
                "cols": int(o["image_cols"][0]), "pieces": o["spatial_shapes"].tolist()}
        json.dump(meta, open(base + ".json", "w"))
        print(f"{name}: {img.size} -> {meta['cols']}x{meta['rows']} tiles, pieces {meta['pieces']}, "
              f"emb {tuple(emb.shape)}")
    decisions(d, out, images, m)


def d1ref(d, mod):
    """The checkpoint's own module mod, imported as a package from DIR."""
    import importlib
    import importlib.util
    if "d1ref" not in sys.modules:
        spec = importlib.util.spec_from_file_location("d1ref", os.path.join(d, "__init__.py"),
                                                      submodule_search_locations=[d])
        sys.modules["d1ref"] = importlib.util.module_from_spec(spec)
    return importlib.import_module("d1ref." + mod)


def decisions(d, out, images, m):
    """d1's own runner over the pictures: a choice, a noul and a score, with
    a text state and with none."""
    from PIL import Image
    from transformers import AutoTokenizer
    runner = d1ref(d, "runner")
    s1 = runner.SystemOne(model=m, tokenizer=AutoTokenizer.from_pretrained(d), device="cpu")
    qs = {
        "kind": {"type": "choice", "instructions": "What does the picture show?",
                 "criteria": {"photo": "a photograph", "chart": "a chart or a diagram", "shapes": "simple shapes"}},
        "people": {"type": "noul", "instructions": "Is there a person in the picture?"},
        "busy": {"type": "score", "instructions": "How busy is the picture?",
                 "criteria": ["empty", "sparse", "moderate", "busy"]},
    }
    cases = []
    for path in images:
        img = Image.open(path).convert("RGB")
        for state in (None, "A user uploaded this picture to a support ticket."):
            r = s1.system_one(state, qs, [img])
            cases.append({"image": os.path.basename(path), "state": state, "questions": qs, "response": r})
            print(os.path.basename(path), repr(state)[:20], json.dumps(r["answers"])[:160])
    json.dump(cases, open(os.path.join(out, "decisions.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--model-free"]:
        model_free()
    elif len(sys.argv) > 3:
        checkpoint(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
