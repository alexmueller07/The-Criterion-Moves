# Integrating the Faithfulness Anchor into `pilot/train_lora.py`

Minimal recipe. Five paste points, no other changes. With `--faith_pairs` unset
the trainer's behavior is bit-identical to today's (the anchor is `None`; only a
micro-step counter increments), so the running SEQ/JOINT pilot arms are
unaffected by merging this.

Prerequisite (cluster, once):

```
python pilot/method/build_grounding_pairs.py --root $ROOT
```

(`$ROOT` = the same directory passed as `--data_root` to training. Needs the
archives `$ROOT/downloads/annotations_trainval2014.zip` and `val2014.zip` that
`data_prep.py` already downloaded, plus `$ROOT/chair/images.jsonl` and
`$ROOT/pope/prompts.jsonl` for the disjointness filter.)

## 1. Import (top of file, next to the existing local import)

```python
from common.gate_checks import run_all_gates
from method.faith_loss import FaithfulnessAnchor
```

The existing `sys.path.insert` of the `pilot/` directory resolves `method.*`
already (namespace package; no `__init__.py` needed).

## 2. Args (after the `--lora_r` line, before `args = ap.parse_args()`)

```python
    ap.add_argument("--faith_pairs", default=None,
                    help="grounding_pairs.jsonl (relative paths inside resolve "
                         "against --data_root); enables the faithfulness anchor")
    ap.add_argument("--faith_weight", type=float, default=0.1)
    ap.add_argument("--faith_k", type=int, default=4)
    ap.add_argument("--faith_margin", type=float, default=2.0)
```

## 3. Instantiate (after `collator = Collator(...)` and `accum = ...`)

```python
    anchor = None
    if args.faith_pairs:
        anchor = FaithfulnessAnchor(processor, args.faith_pairs, args.data_root,
                                    k_pairs=args.faith_k,
                                    margin=args.faith_margin, device="cuda")
        print(f"[train] faith anchor: {len(anchor.pairs)} pairs k={anchor.k} "
              f"margin={anchor.margin} weight={args.faith_weight}", flush=True)
```

## 4. Trainer subclass (after the `callbacks.append(LossGuard())` line, before `targs = TrainingArguments(...)`)

A `TrainerCallback` is NOT sufficient here: callbacks only receive control-flow
hooks (`on_step_end`, `on_log`, ...) after `training_step` has already run
`backward()` — they cannot contribute a term to the loss that gradients flow
through. Loss composition has to happen in `compute_loss`, so subclass `Trainer`
(closing over `anchor`, `accum`, and `args`, like `MarkCkpt` closes over `marks`):

```python
    class FaithTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._micro = 0
            self.faith_history = []

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss, outputs = super().compute_loss(model, inputs,
                                                 return_outputs=True, **kwargs)
            if anchor is not None and self._micro % accum == 0:
                f = anchor.loss(model)
                self.faith_history.append(float(f.detach()))
                loss = loss + args.faith_weight * accum * f
            self._micro += 1
            return (loss, outputs) if return_outputs else loss
```

Then change the construction line (nothing else about it):

```python
    trainer = FaithTrainer(model=model, args=targs, train_dataset=ds,
                           data_collator=collator, callbacks=callbacks)
```

Why once per accumulation cycle, and why `* accum`: the anchor forward (3·k
sequences) is too expensive to run on all `accum` (=32 at pilot settings)
micro-steps, so it runs on the first micro-step of each cycle only. The Trainer
divides every micro-step loss by `gradient_accumulation_steps` before
`backward()` on the `num_items_in_batch is None` path — which is the active path
here, per the existing NOTE in `train_lora.py` about the PEFT wrapper hiding
`num_items_in_batch`. Multiplying the single per-cycle term by `accum` cancels
that division exactly, so the contribution per optimizer step is
`faith_weight * ∇L_faith` — i.e. `--faith_weight` means what METHOD.md's
`\lambda_{\rm faith}` means. (`**kwargs` keeps the override compatible with
Trainer versions that do and don't pass `num_items_in_batch`.) The `LossGuard`
threshold (`25 * accum`) has ample headroom for the added term (`~0.1 * 32 *
sp(2) ≈ 4` on one micro-step in `accum`).

Cost: one extra forward of `3k` (=12) ~600-token sequences per optimizer step,
against 32 task micro-batches — measure the wall-clock delta in the smoke run
rather than assuming it is negligible.

## 5. Sanity + logging (two small blocks)

After the existing `with torch.no_grad():` pre-train check block (the model is
on cuda by then):

```python
    if anchor is not None:
        with torch.no_grad():
            f0 = float(anchor.loss(model))
        assert 0.0 < f0 < 50.0, f"pre-train faith loss {f0} implausible"
        print(f"[train] pre-train faith loss {f0:.4f}", flush=True)
```

And one extra key in the `log = {...}` dict at the end:

```python
        "faith_history": trainer.faith_history,
```

## Smoke recipe (before any real faith run)

10 optimizer steps, tiny accumulation so the anchor fires 10 times, against a
no-faith twin:

```
python pilot/method/build_grounding_pairs.py --root $ROOT --smoke

python pilot/train_lora.py \
  --data $ROOT/tasks/scienceqa/train.jsonl --data_root $ROOT \
  --out $ROOT/runs/faith_smoke0 --max_steps 10 --micro_batch 2 --eff_batch 8

python pilot/train_lora.py \
  --data $ROOT/tasks/scienceqa/train.jsonl --data_root $ROOT \
  --out $ROOT/runs/faith_smoke1 --max_steps 10 --micro_batch 2 --eff_batch 8 \
  --faith_pairs $ROOT/grounding/grounding_pairs.jsonl \
  --faith_weight 0.1 --faith_k 4 --faith_margin 2.0

python - "$ROOT/runs/faith_smoke0/training_log.json" \
         "$ROOT/runs/faith_smoke1/training_log.json" <<'EOF'
import json, math, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
fh = b["faith_history"]
assert fh and len(fh) == 10, f"expected 10 faith evals, got {len(fh) if fh else 0}"
assert all(math.isfinite(v) for v in fh), f"non-finite faith loss: {fh}"
first, last = sum(fh[:3]) / 3, sum(fh[-3:]) / 3
assert last <= first * 1.25 + 0.05, f"faith loss rising: {first:.3f} -> {last:.3f}"
r = b["train_loss"] / a["train_loss"]
assert 1 / 3 < r < 3, f"task loss changed order of magnitude: {a['train_loss']:.3f} -> {b['train_loss']:.3f}"
print(f"SMOKE OK  faith {first:.3f}->{last:.3f}  task {a['train_loss']:.3f} vs {b['train_loss']:.3f}")
EOF
```

What the three asserts establish: (1) faith loss is finite at every evaluation;
(2) it is decreasing-or-stable over 10 steps (1.25x band absorbs
k=4-sample + dropout noise — it fails only on a real rise); (3) the task loss is
order-of-magnitude unaffected (the faith run's `train_loss` includes the
λ-weighted faith term, so the 3x band is deliberately loose; both runs' logged
values carry the same accumulation-factor inflation, so the ratio is clean).
Per the standing smoke rule: never trust a faith run whose smoke wasn't green.
