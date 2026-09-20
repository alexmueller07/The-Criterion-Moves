"""Assert the faith-anchor smoke run is green (see INTEGRATION.md).

Usage: check_faith_smoke.py <no_faith_training_log.json> <faith_training_log.json>
"""
import json
import math
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
fh = b.get("faith_history") or []
assert fh and len(fh) == 10, f"expected 10 faith evals, got {len(fh)}"
assert all(math.isfinite(v) for v in fh), f"non-finite faith loss: {fh}"
first, last = sum(fh[:3]) / 3, sum(fh[-3:]) / 3
assert last <= first * 1.25 + 0.05, f"faith loss rising: {first:.3f} -> {last:.3f}"
r = b["train_loss"] / a["train_loss"]
assert 1 / 3 < r < 3, (
    f"task loss changed order of magnitude: {a['train_loss']:.3f} -> {b['train_loss']:.3f}")
print(f"SMOKE OK  faith {first:.3f}->{last:.3f}  "
      f"task {a['train_loss']:.3f} vs {b['train_loss']:.3f}")
