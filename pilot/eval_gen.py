"""Generation runner: base model + optional adapter over a prompts JSONL.

Input rows need {id, image, prompt}; all other fields are passed through to the
output row. Output rows add {output, n_new_tokens, truncated, gen_s}.
Greedy decoding. Audit fields are mandatory (generation-budget confound).

--dump_logits (2026-09-08): logit-level instrument for the criterion-drift
mechanism. ONE teacher-forced forward per prompt (no generation); records the
' Yes' / ' No' next-token logits at the answer position, their gap
g = z_yes - z_no (the decision variable of design_notes/
method_calibration_theory.md; greedy answers "yes" iff g > 0), and the argmax
token. Right padding, answer position = attention_mask.sum(1) - 1, full
processor call (backbone-agnostic), OOM-halving. The default generation path
is untouched.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.gate_checks import run_all_gates
from model_zoo import available_backbones, get_backbone

import torch
from PIL import Image


def run_dump(rows, model, processor, backbone, data_root, batch, yes_id, no_id,
             device="cuda:0", adapter_label="none", log=print, on_rec=None, blind=False,
             yes_ids=None, no_ids=None):
    """Teacher-forced answer-position logits for every row (no generation).

    Returns the list of output rows (input row minus image/prompt, plus
    {z_yes, z_no, gap, argmax_id, argmax_token, argmax_is, n_input_tokens,
    adapter, fwd_s}). Requires processor.tokenizer.padding_side == "right":
    the answer position is attention_mask.sum(1) - 1, which is only the last
    REAL token under right padding. Chunks of `batch`; on CUDA OOM a chunk is
    halved and retried (down to 1), mirroring gen_chunk. `on_rec` (if given)
    receives each record as soon as it exists (streaming write for resume).
    Mirrors FaithfulnessAnchor._answer_logits: floats cast to the model dtype,
    every processor output forwarded, use_cache=False, logit length asserted
    against the input length, margins read in fp32.

    2026-09-14: z_yes/z_no are POOLED over casing variants when `yes_ids`/`no_ids`
    are supplied -- logsumexp over each side rather than a single vocabulary row.
    This is not cosmetic. The model's argmax answer token FLIPS casing between
    training stages (capitalised at k1/k3, lowercase at k2/k4, ~100% of 9000 rows
    each), so a fixed-id readout measures rows the model is not emitting at roughly
    half the stages. Every dump written before this change is single-casing and
    unusable for criterion analysis; see the 2026-09-12 prereg entry. The scalar
    yes_id/no_id are retained only for the argmax_is label and the log line."""
    tok = processor.tokenizer
    assert getattr(tok, "padding_side", "right") == "right", (
        f"dump_logits needs right padding, got {tok.padding_side!r}")
    param = next(model.parameters())
    yes_str = tok.decode([yes_id])
    no_str = tok.decode([no_id])
    if yes_ids is None or no_ids is None:
        from method.answer_tokens import resolve_answer_token_sets
        yes_ids, no_ids = resolve_answer_token_sets(tok, backbone)
    yes_ids = list(yes_ids); no_ids = list(no_ids)
    assert yes_ids and no_ids, "empty answer-token set; refusing to dump"
    log(f"[dump] POOLED answer tokens: {len(yes_ids)} yes variants "
        f"{[tok.decode([i]) for i in yes_ids]}, {len(no_ids)} no variants "
        f"{[tok.decode([i]) for i in no_ids]}")

    def dump_chunk(chunk):
        texts = [backbone.build_prompt(r["prompt"]) for r in chunk]
        images = [_load_img(data_root, r, blind)
                  for r in chunk]
        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True)
        model_inputs = {}
        for name, v in enc.items():
            if hasattr(v, "to"):
                v = v.to(device, param.dtype) if v.dtype.is_floating_point \
                    else v.to(device)
            model_inputs[name] = v
        t0 = time.time()
        try:
            with torch.inference_mode():
                out = model(use_cache=False, **model_inputs)
        except torch.cuda.OutOfMemoryError:
            del model_inputs, enc
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise
            mid = (len(chunk) + 1) // 2
            log(f"[dump] OOM at batch {len(chunk)} -> splitting to {mid}")
            recs = dump_chunk(chunk[:mid])
            recs += dump_chunk(chunk[mid:])
            return recs
        dt = time.time() - t0
        logits = out.logits
        attn = model_inputs["attention_mask"]
        assert logits.shape[1] == attn.shape[1], (
            f"logit length {logits.shape[1]} != input length {attn.shape[1]}"
            " - image-token expansion mismatch; last-position index would be wrong")
        assert logits.shape[-1] > max(max(yes_ids), max(no_ids))
        last = attn.sum(dim=1) - 1                       # (B,) last REAL token
        step = logits[torch.arange(logits.shape[0], device=logits.device), last]
        step = step.float()                              # (B, V) in fp32
        # pooled over casing variants: logsumexp per side, so a stage that emits
        # "yes" and one that emits "Yes" are measured on the same statistic
        yi = torch.tensor(yes_ids, device=step.device)
        ni = torch.tensor(no_ids, device=step.device)
        z_yes = torch.logsumexp(step.index_select(1, yi), dim=1)
        z_no = torch.logsumexp(step.index_select(1, ni), dim=1)
        arg = step.argmax(dim=-1)
        recs = []
        for i, r in enumerate(chunk):
            rec = dict(r)
            rec.pop("image", None)
            rec.pop("prompt", None)
            aid = int(arg[i])
            zy, zn = float(z_yes[i]), float(z_no[i])
            rec.update({
                "z_yes": zy, "z_no": zn, "gap": zy - zn,
                "argmax_id": aid,
                "argmax_token": tok.decode([aid]),
                "argmax_is": ("yes" if aid in yes_ids else
                              "no" if aid in no_ids else "other"),
                "n_input_tokens": int(attn[i].sum()),
                "adapter": adapter_label,
                "fwd_s": round(dt / len(chunk), 4),
            })
            if on_rec is not None:
                on_rec(rec)
            recs.append(rec)
        return recs

    log(f"[dump] yes_id={yes_id} ({yes_str!r}) no_id={no_id} ({no_str!r}) "
        f"padding_side={tok.padding_side} dtype={param.dtype}")
    all_recs = []
    t_all = time.time()
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        all_recs += dump_chunk(chunk)
        if (start // batch) % 20 == 0:
            done = start + len(chunk)
            rate = done / max(1e-9, time.time() - t_all)
            log(f"[dump] {done}/{len(rows)} ({rate:.2f} ex/s)")
    return all_recs



def _load_img(data_root, r, blind=False):
    """Load a prompt's image; with blind=True return a constant mid-gray image of
    the same size (the prior-vs-evidence probe: how much of the yes/no decision
    statistic survives with NO visual evidence)."""
    img = Image.open(os.path.join(data_root, r["image"])).convert("RGB")
    if blind:
        return Image.new("RGB", img.size, (128, 128, 128))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--adapter", default=None, help="omit for S0 base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--backbone", default="llava15",
                    choices=available_backbones(),
                    help="model backbone adapter from model_zoo.py")
    ap.add_argument("--blind", action="store_true",
                    help="replace every image with constant gray (language-prior probe)")
    ap.add_argument("--dump_logits", action="store_true",
                    help="no generation: one teacher-forced forward per prompt, "
                         "record ' Yes'/' No' logits + gap at the answer position")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    run_all_gates(out_dir, min_free_gb=20.0)

    backbone = get_backbone(args.backbone)
    model, processor = backbone.load(backbone.model_id, device_map="cuda:0")
    processor.tokenizer.padding_side = "left"
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[gen] merged adapter {args.adapter}", flush=True)
    model.eval()

    rows = [json.loads(l) for l in open(args.prompts)]
    if args.limit > 0:
        rows = rows[:args.limit]

    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for l in f:
                try:
                    done_ids.add(json.loads(l)["id"])
                except Exception:
                    pass
        print(f"[gen] resuming: {len(done_ids)} already done", flush=True)
    rows = [r for r in rows if r["id"] not in done_ids]

    if args.dump_logits:
        # Logit instrument: right padding (answer position = last real token),
        # in-context ' Yes'/' No' ids shared with the anchor loss.
        from method.faith_loss import resolve_answer_token_ids
        processor.tokenizer.padding_side = "right"
        yes_id, no_id = resolve_answer_token_ids(processor.tokenizer, backbone)
        from method.answer_tokens import resolve_answer_token_sets
        _yes_ids, _no_ids = resolve_answer_token_sets(processor.tokenizer, backbone)
        fout = open(args.out, "a")

        def _write(rec):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

        run_dump(rows, model, processor, backbone, args.data_root, args.batch,
                 yes_id, no_id, device="cuda:0",
                 yes_ids=_yes_ids, no_ids=_no_ids,
                 adapter_label=args.adapter or "none",
                 log=lambda m: print(m, flush=True), on_rec=_write, blind=args.blind)
        fout.close()
        print(f"[dump] DONE -> {args.out}", flush=True)
        return

    def gen_chunk(chunk):
        """Generate for a chunk; on CUDA OOM, halve and recurse (down to 1).
        Bigger backbones (Qwen2.5-VL 8.45B) OOM on 24GB at LLaVA batch sizes
        for long-caption generation; this makes any --batch self-recover so
        already-queued eval jobs (which pick up this code at start) survive."""
        texts = [backbone.build_prompt(r["prompt"]) for r in chunk]
        images = [_load_img(args.data_root, r, args.blind)
                  for r in chunk]
        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True).to("cuda:0")
        t0 = time.time()
        try:
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     **backbone.generation_kwargs(processor))
        except torch.cuda.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise
            mid = (len(chunk) + 1) // 2
            print(f"[gen] OOM at batch {len(chunk)} -> splitting to {mid}", flush=True)
            recs = gen_chunk(chunk[:mid])
            recs += gen_chunk(chunk[mid:])
            return recs
        dt = time.time() - t0
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        decoded = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        recs = []
        for r, d, nt in zip(chunk, decoded, new_tokens):
            n_new = int((nt != processor.tokenizer.pad_token_id).sum())
            ended_with_eos = bool((nt == processor.tokenizer.eos_token_id).any())
            rec = dict(r)
            rec.pop("image", None)
            rec.update({
                "output": d.strip(),
                "n_new_tokens": n_new,
                "truncated": bool(n_new >= args.max_new_tokens and not ended_with_eos),
                "adapter": args.adapter or "none",
                "gen_s": round(dt / len(chunk), 3),
            })
            recs.append(rec)
        return recs

    fout = open(args.out, "a")
    t_all = time.time()
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        for rec in gen_chunk(chunk):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if (start // args.batch) % 20 == 0:
            done = start + len(chunk)
            rate = done / max(1e-9, time.time() - t_all)
            print(f"[gen] {done}/{len(rows)} ({rate:.2f} ex/s)", flush=True)
    fout.close()
    print(f"[gen] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
