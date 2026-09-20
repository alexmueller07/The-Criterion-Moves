#!/usr/bin/env python3
"""Casing-robust answer-position logit dump.

WHY (2026-09-09). `eval_gen.py --dump_logits` records z at ONE hard-coded pair of
vocabulary rows, resolved as " Yes"/" No" (LLaVA: 3869/1939). The model's emitted
answer token drifts with the task: at the TextVQA and VizWiz stages it emits
'yes'/'no' (4874/694), which that pair does not contain, so `argmax_is` is
"other" on 100% of rows there and the recorded statistic has decoupled from the
decision the model actually makes (it agrees with the realized argmax only 90.7%
of the time). Verified on the real tokenizer: the old pair captures NEITHER
emitted id.

This script writes the same kind of dump using the POOLED statistic from
`answer_tokens.py`:

    g = logsumexp(z over yes-casing variants) - logsumexp(z over no-casing variants)

and additionally records, per row, which pooled set the realized argmax fell into,
so this failure mode is visible directly in any future dump.

It is a SEPARATE script, not a patch to eval_gen.py, because ~85 arms are queued
and a pending job runs whatever is on disk when it starts; changing the shared
instrument mid-flight would make queued arms incomparable with completed ones.

Usage:
  python3 dump_pooled_logits.py --backbone llava15 --prompts P.jsonl \
      --data_root D [--adapter A] --out OUT.jsonl [--batch 12] [--limit N]
"""
import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_tokens import (pooled_decision_stat,  # noqa: E402
                           realized_side, resolve_answer_token_sets)
from model_zoo import get_backbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="llava15")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    backbone = get_backbone(args.backbone)
    model, processor = backbone.load(backbone.model_id, device_map="cuda:0")
    # The answer position is attention_mask.sum(1) - 1, which is the last REAL
    # token only under right padding. Same requirement as eval_gen's dump path.
    processor.tokenizer.padding_side = "right"
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[dump] merged adapter {args.adapter}", flush=True)
    model.eval()

    yes_ids, no_ids = resolve_answer_token_sets(processor.tokenizer, backbone)
    tok = processor.tokenizer
    print("[dump] pooled yes ids: %s" % [(i, tok.decode([i])) for i in yes_ids],
          flush=True)
    print("[dump] pooled no  ids: %s" % [(i, tok.decode([i])) for i in no_ids],
          flush=True)

    rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for l in f:
                try:
                    done.add(json.loads(l)["id"])
                except Exception:
                    pass
        print(f"[dump] resuming, {len(done)} done", flush=True)
    rows = [r for r in rows if r["id"] not in done]

    param = next(model.parameters())
    fout = open(args.out, "a")
    n_other = 0
    t0 = time.time()
    i = 0
    bs = args.batch
    while i < len(rows):
        chunk = rows[i:i + bs]
        try:
            images = [backbone.anchor_image(
                Image.open(os.path.join(args.data_root, r["image"])).convert("RGB"))
                for r in chunk]
            texts = [backbone.build_prompt(r["prompt"]) for r in chunk]
            enc = processor(text=texts, images=images, return_tensors="pt",
                            padding=True, truncation=True, max_length=2048)
            mi = {}
            for k, v in enc.items():
                if hasattr(v, "to"):
                    v = v.to("cuda:0", param.dtype) if v.dtype.is_floating_point \
                        else v.to("cuda:0")
                mi[k] = v
            with torch.no_grad():
                out = model(use_cache=False, **mi)
            logits = out.logits
            attn = mi["attention_mask"]
            assert logits.shape[1] == attn.shape[1], (
                f"logit length {logits.shape[1]} != input length {attn.shape[1]}")
            last = attn.sum(dim=1) - 1
            idx = last.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
            step = logits.gather(1, idx).squeeze(1).float()
            g = pooled_decision_stat(step, yes_ids, no_ids)
            amax = step.argmax(dim=-1)
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            print(f"[dump] OOM -> batch {bs}", flush=True)
            continue
        for j, r in enumerate(chunk):
            aid = int(amax[j])
            side = realized_side(aid, yes_ids, no_ids)
            n_other += (side == "other")
            rec = {k: v for k, v in r.items() if k not in ("image", "prompt")}
            rec.update({
                "g_pooled": round(float(g[j]), 4),
                "argmax_id": aid,
                "argmax_token": tok.decode([aid]),
                # which POOLED set the emitted token fell in. "other" here is the
                # real alarm: it means neither casing set caught what the model
                # actually said, which is the failure the single-id instrument
                # hit silently.
                "argmax_side": side,
                "adapter": args.adapter or "none",
            })
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        i += len(chunk)
        if i % (bs * 20) < bs:
            print(f"[dump] {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    fout.close()
    frac = n_other / max(1, len(rows))
    print(f"[dump] done. argmax outside BOTH pooled sets: {n_other}/{len(rows)} "
          f"({frac:.4f})", flush=True)
    # A high rate means even pooled casing variants miss the emitted token, i.e.
    # the model is answering with something else entirely. Fail loudly: a silent
    # pass here is exactly how the single-id bug survived.
    if frac > 0.02:
        print("DUMP_POOLED_WARN: >2% of rows fall outside both pooled sets", flush=True)
        return 2
    print("DUMP_POOLED_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
