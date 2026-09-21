"""E4 on UCIT: is the criterion set by the most recently trained task?

Table 5 records E4 as absent on the full study, because the single-task control
arms were never run there. They are running now, one per UCIT task, each
training the untuned base on that task alone for the matched step budget.

The test. For task T at ordering position k, compare the criterion of
base -> T alone against the criterion of the sequential arm at stage k, which
reached T after training k-1 other tasks first. If the criterion is memoryless,
the two agree.

The design also supplies its own error bar, which the paper otherwise lacks.
The task at position 1 is a REPLICATION: stage 1 of the sequential arm IS
base -> T alone, trained twice with different run-level nondeterminism. Its gap
is therefore a training-run noise floor, and every deeper comparison is read
against it rather than against zero. The paper's bootstrap intervals cover
evaluation-set noise only, which is the wrong quantity for a claim about
training.

SCOPE OF THAT FLOOR -- verified by reading run_arm.py, not assumed. For the
position-1 task the single-task control and sequential stage 1 are the SAME
computation: same seed, same train.jsonl, same step budget, extra=None in both
branches. So the gap measures run-level nondeterminism alone (GPU assignment,
bf16 kernel nondeterminism), NOT seed variance, which is strictly wider and is
what the paper's three seeds already bound. Inside E4 this is exactly the right
null, because the E4 contrast holds task and seed fixed and varies only the
history. It must never be quoted as the paper's general error bar.

Usage (on the cluster):
    python3 analysis/e4_memorylessness.py
"""
import glob
import json
import math
import os
import re

RESULTS = os.path.expanduser("~/cl-halluc/results_fs")
READOUT = os.path.expanduser("~/cl-halluc/readout/fs_aggregate.json")
ORDER_O1 = ["ArxivQA", "CLEVR-Math", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"]
# Each task's own mean post-settling criterion pull on the sequential arm
# (paper Table 2). A task near zero here is criterion-INACTIVE: it barely moves
# the threshold, so reaching it in a sequence leaves the previous task's
# criterion standing. The pilot's claim is memorylessness with respect to the
# most recent criterion-ACTIVE supervision, so a large gap on an inactive task
# is what that claim predicts, not evidence against it. Separating the two is
# the whole point of running all six.
TASK_PULL = {"ArxivQA": -0.109, "CLEVR-Math": +0.005, "Flickr30k": +0.163,
             "IconQA": -0.251, "ImageNet-R": -0.184, "VizWiz": +0.139}
ACTIVE_CUT = 0.05   # |pull| below this counts as criterion-inactive
PARSE_FLOOR = 0.95


def z(p, n=4500):
    p = min(max(p, 1.0 / (2 * n)), 1 - 1.0 / (2 * n))
    lo, hi = -10.0, 10.0
    for _ in range(100):
        m = (lo + hi) / 2
        if 0.5 * (1 + math.erf(m / math.sqrt(2))) < p:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def parse_yn(out):
    t = re.sub(r"^[^a-z]+", "", (out or "").strip().lower())
    return 1 if t.startswith("yes") else 0 if t.startswith("no") else None


def score(cell_dir):
    f = os.path.join(cell_dir, "pope_gen.jsonl")
    if not (os.path.exists(os.path.join(cell_dir, "EVAL_DONE")) and os.path.exists(f)):
        return None
    hy = hn = fy = fn = bad = tot = 0
    for line in open(f):
        r = json.loads(line)
        tot += 1
        a = parse_yn(r.get("output", ""))
        if a is None:
            bad += 1
            continue
        if r["gt"] == "yes":
            hy += a == 1
            hn += a == 0
        else:
            fy += a == 1
            fn += a == 0
    if not (hy + hn) or not (fy + fn):
        return None
    H, FA = hy / (hy + hn), fy / (fy + fn)
    rate = (tot - bad) / tot
    return {"c": -0.5 * (z(H) + z(FA)), "dprime": z(H) - z(FA),
            "parse_rate": rate, "usable": rate >= PARSE_FLOOR, "n": tot}


def main():
    rd = json.load(open(READOUT))["backbones"]["llava15"]
    seq = rd["arms"].get("seq|o1|s17")
    if seq is None:
        print("sequential reference cell absent from the readout")
        return
    print("untuned base c = %+.4f\n" % rd["base"]["pope"]["c"])

    singles, floor = {}, None
    for k, task in enumerate(ORDER_O1, start=1):
        s = score(os.path.join(RESULTS, "fsL_single_%s_o1_s17_k1" % task))
        ref = seq.get(str(k), {}).get("pope")
        if s is None or ref is None:
            print("  %-12s position %d: control not scored yet" % (task, k))
            continue
        if not s["usable"]:
            print("  %-12s position %d: FORMAT COLLAPSE (%.1f%% parsed), no criterion"
                  % (task, k, 100 * s["parse_rate"]))
            continue
        singles[k] = (task, s["c"], ref["c"])
        if k == 1:
            floor = abs(s["c"] - ref["c"])
            print("  %-12s position 1:  single %+.4f   sequential %+.4f   gap %+.4f"
                  "   <-- same computation twice: run-level noise floor"
                  % (task, s["c"], ref["c"], s["c"] - ref["c"]))

    if not singles:
        print("\nnothing scored yet.")
        return
    if floor is None:
        print("\nThe position-1 replication is not in yet, so there is no measured")
        print("floor and no gap below can be called small or large.")
        return

    # Two competing models, scored on the same observations.
    #   NAIVE  : c is whatever the most recent task produces on its own.
    #   CARRY  : c is whatever the most recent criterion-ACTIVE task left, so an
    #            inactive task carries its predecessor's criterion forward.
    # They make the SAME prediction on an active task -- those positions test
    # memorylessness itself. They differ only on inactive tasks, which is where
    # the two models can be told apart.
    # A third model, and the only one that uses the controls for what they
    # independently measure. INCREMENT: a task moves the criterion by a fixed
    # amount, so the sequential arm should advance from wherever it already is
    # by the shift that task produces on the untuned base,
    #   pred = c_seq(k-1) + (c_single(T_k) - c_base).
    # This is deliberately NOT built from Table 2's per-task pulls: those are the
    # means of the very Delta c_k this would be predicting, so scoring against
    # them is circular and would fit well by construction. The control arm and
    # the base are independent of the sequential trajectory, so this one can fail.
    base_c = rd["base"]["pope"]["c"]
    print("\n%-12s %-8s %8s %8s %8s %9s   %s"
          % ("task", "kind", "observed", "naive", "carry", "increment", "residual / floor"))
    res_n, res_c, res_i, tested = [], [], [], 0
    ACTIVE_RESID = []
    for k in sorted(singles):
        if k == 1:
            continue
        task, c_single, c_seq = singles[k]
        active = abs(TASK_PULL.get(task, 0.0)) >= ACTIVE_CUT
        prev = seq.get(str(k - 1), {}).get("pope")
        c_carry = c_single if active else (prev["c"] if prev else None)
        c_inc = (prev["c"] + (c_single - base_c)) if prev else None
        rn = abs(c_seq - c_single)
        rc = abs(c_seq - c_carry) if c_carry is not None else None
        ri = abs(c_seq - c_inc) if c_inc is not None else None
        res_n.append(rn)
        if active:
            ACTIVE_RESID.append(rn)
        if rc is not None:
            res_c.append(rc)
        if ri is not None:
            res_i.append(ri)
        tested += 1
        print("%-12s %-8s %+8.4f %+8.4f %8s %9s   naive %5.1fx  carry %s  incr %s"
              % (task, "active" if active else "INACTIVE", c_seq, c_single,
                 ("%+.4f" % c_carry) if c_carry is not None else "  --",
                 ("%+.4f" % c_inc) if c_inc is not None else "   --",
                 rn / floor, ("%5.1fx" % (rc / floor)) if rc is not None else " --",
                 ("%5.1fx" % (ri / floor)) if ri is not None else " --"))

    print("\nrun-level noise floor (position-1 replication, same computation twice): %.4f" % floor)
    if res_n:
        print("NAIVE  memorylessness: mean residual %.4f (%.1fx floor), worst %.1fx"
              % (sum(res_n) / len(res_n), (sum(res_n) / len(res_n)) / floor, max(res_n) / floor))
    if res_c:
        print("CARRY  (last criterion-active task): mean residual %.4f (%.1fx floor), worst %.1fx"
              % (sum(res_c) / len(res_c), (sum(res_c) / len(res_c)) / floor, max(res_c) / floor))
    if res_i:
        print("INCR   (previous level + the task's shift on the base): mean residual %.4f (%.1fx floor), worst %.1fx"
              % (sum(res_i) / len(res_i), (sum(res_i) / len(res_i)) / floor, max(res_i) / floor))
    print("\nEndpoint (prereg 2026-09-20): mean NAIVE residual over criterion-ACTIVE")
    print("positions only, in floor units. <=3x holds, 3-8x partial memory, >8x fails.")
    act = [abs(r) for r in ACTIVE_RESID]
    if act:
        m = (sum(act) / len(act)) / floor
        print("  active positions scored: %d of 4   mean %.1fx   -> %s"
              % (len(act), m, "HOLDS" if m <= 3 else ("PARTIAL MEMORY" if m <= 8 else "FAILS")))

    if len(singles) < len(ORDER_O1):
        print("\nINCOMPLETE: %d control(s) outstanding; no E4 verdict until all are in."
              % (len(ORDER_O1) - len(singles)))
    else:
        print("\nAll six controls in. The active-task rows are the memorylessness test;")
        print("the inactive rows are where the two models separate.")


if __name__ == "__main__":
    main()
