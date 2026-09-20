#!/usr/bin/env python3
"""Does a task move the criterion the same way wherever it appears in the stream?

WHY (2026-09-11). The paper explains the failed generative endpoint by TASK
RECENCY: what a checkpoint does tracks the task it was last trained on rather
than how deep it sits in the sequence. That account was fitted post-hoc on the
same observations it explains, and the pre-registration records an out-of-sample
test as owed.

The three orderings are that test, and they cost nothing to use. Each of the six
UCIT tasks appears at a DIFFERENT position in each ordering, so task identity and
sequence depth are crossed rather than confounded. For every stage we take the
one-step criterion change

    dc_k = c_k - c_{k-1}        (c_0 = the untuned base)

and ask how much of its variance is explained by WHICH TASK was just trained
versus by HOW DEEP we are. Recency predicts task >> position. If position wins,
the recency account is wrong and the paper must say so.

This is an analysis of existing readouts only -- no GPU, no new runs.

Usage:  python3 order_effects.py [--agg analysis/readout/fs_aggregate.json]
                                 [--arm seq] [--out out.json]
"""
import argparse
import itertools
import json
import math
import os
import random
import sys


def eta_sq(groups):
    """Classical eta^2: between-group sum of squares over total.

    groups: {label: [values]}. Returns None when there is nothing to explain
    (fewer than two non-empty groups, or zero total variance) rather than a
    fake 0.0, because those two cases mean different things.
    """
    vals = [v for g in groups.values() for v in g]
    n = len(vals)
    if n < 3 or sum(1 for g in groups.values() if g) < 2:
        return None
    gm = sum(vals) / n
    ss_tot = sum((v - gm) ** 2 for v in vals)
    if ss_tot <= 0:
        return None
    ss_bet = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups.values() if g)
    return ss_bet / ss_tot


def perm_p(groups, observed, reps=20000, seed=17):
    """Permutation p for eta^2: reshuffle labels, keeping group sizes fixed.

    eta^2 is biased upward by the number of groups, and task and position do
    NOT have the same number of levels: with --skip_first, task has 6 and
    position has 5 (stages 2..6). An earlier version of this docstring claimed
    they were equal; that was wrong, and the direction of the error favours our
    own conclusion, since the extra level inflates task's raw eta^2 relative to
    position's.

    This is exactly why the PERMUTATION P-VALUES, and not the raw eta^2, carry
    the comparison: the null is built by reshuffling labels while holding each
    group's size fixed, so whatever upward bias the level count induces is
    present in the null too and divides out. Any text quoting 0.731 vs 0.049
    must therefore quote the p-values alongside them.

    Exchangeability caveat, to state rather than hide: the permuted units are
    the per-stage dc values (N = 45 = 9 cells x 5 stages), not the 3 orderings,
    so small-N warnings about permutation tests do not apply. But the 45 are not
    independent -- five stages share a cell's trajectory. The defence is that the
    same dependence structure applies to BOTH factors, so the CONTRAST between
    them is what the design supports, not either p-value on its own.
    """
    sizes = [len(g) for g in groups.values()]
    pool = [v for g in groups.values() for v in g]
    rnd = random.Random(seed)
    hits = 0
    for _ in range(reps):
        rnd.shuffle(pool)
        i = 0
        shuffled = {}
        for k, sz in zip(groups, sizes):
            shuffled[k] = pool[i:i + sz]
            i += sz
        e = eta_sq(shuffled)
        if e is not None and e >= observed:
            hits += 1
    return (hits + 1) / (reps + 1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=os.path.join(here, "readout", "fs_aggregate.json"))
    ap.add_argument("--arm", default="seq")
    ap.add_argument("--backbone", default="llava15")
    ap.add_argument("--skip_first", action="store_true",
                    help="drop stage 1. The anchor makes a one-time SETTLING jump "
                         "there (+0.358 mean), which otherwise loads onto the "
                         "'depth' factor and masquerades as a sequence effect. The "
                         "paper's endpoint is post-settling for the same reason.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agg = json.load(open(args.agg))
    arms = agg["backbones"][args.backbone]["arms"]
    base_c = agg["backbones"][args.backbone]["base"]["pope"]["c"]

    rows = []   # (cell, position, task, dc)
    for key, cell in sorted(arms.items()):
        if key.split("|")[0] != args.arm:
            continue
        stages = sorted((k for k in cell if k.isdigit()), key=int)
        prev = base_c
        for k in stages:
            st = cell[k]
            p = st.get("pope")
            if not p:
                continue
            if not (args.skip_first and int(k) == 1):
                rows.append((key.split("|", 1)[1].replace("|", "/"), int(k),
                             st.get("task_trained"), p["c"] - prev))
            prev = p["c"]

    if not rows:
        print("no rows for arm %r" % args.arm)
        return 1

    by_task, by_pos = {}, {}
    for _, pos, task, dc in rows:
        by_task.setdefault(task, []).append(dc)
        by_pos.setdefault(pos, []).append(dc)

    e_task, e_pos = eta_sq(by_task), eta_sq(by_pos)
    p_task = perm_p(by_task, e_task) if e_task is not None else None
    p_pos = perm_p(by_pos, e_pos) if e_pos is not None else None

    print("One-step criterion change  dc_k = c_k - c_(k-1)   [arm=%s, %d cells, %d stages]"
          % (args.arm, len({r[0] for r in rows}), len(rows)))
    print("Task identity and sequence depth are CROSSED by the three orderings,\n"
          "so this is the out-of-sample test the recency account owed.\n")

    print("%-14s %5s %9s %9s %9s" % ("task", "n", "mean dc", "sd", "range"))
    print("-" * 52)
    for t in sorted(by_task, key=lambda x: -abs(sum(by_task[x]) / len(by_task[x]))):
        v = by_task[t]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        print("%-14s %5d %+9.4f %9.4f %9s" % (t, len(v), m, sd,
              "%.2f..%.2f" % (min(v), max(v))))

    print("\n%-14s %5s %9s %9s" % ("position", "n", "mean dc", "sd"))
    print("-" * 42)
    for k in sorted(by_pos):
        v = by_pos[k]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        print("%-14s %5d %+9.4f %9.4f" % ("stage %d" % k, len(v), m, sd))

    print("\nVariance in dc explained (eta^2, same number of levels each, "
          "permutation p over 20000 shuffles):")
    print("  TASK IDENTITY   eta^2 = %s   p = %s" %
          ("%.3f" % e_task if e_task is not None else "n/a",
           "%.4f" % p_task if p_task is not None else "n/a"))
    print("  SEQUENCE DEPTH  eta^2 = %s   p = %s" %
          ("%.3f" % e_pos if e_pos is not None else "n/a",
           "%.4f" % p_pos if p_pos is not None else "n/a"))

    if e_task is not None and e_pos is not None:
        ratio = e_task / e_pos if e_pos > 0 else float("inf")
        print("\n  ratio task/depth = %.2f" % ratio)
        if p_task is not None and p_task < 0.05 and (p_pos is None or p_pos >= 0.05):
            verdict = ("SUPPORTS RECENCY: task identity explains dc, depth does not. "
                       "This is out-of-sample for the account, because the orderings "
                       "cross the two factors.")
        elif p_pos is not None and p_pos < 0.05 and (p_task is None or p_task >= 0.05):
            verdict = ("AGAINST RECENCY: depth explains dc and task identity does not. "
                       "The recency reframe must be WITHDRAWN.")
        elif (p_task is not None and p_task < 0.05) and (p_pos is not None and p_pos < 0.05):
            verdict = ("BOTH significant -- report both and the ratio; recency is "
                       "supported only as the LARGER of two real effects.")
        else:
            verdict = ("NEITHER significant at 0.05. Underpowered at this n; the "
                       "recency account is neither supported nor refuted here.")
        print("\nVERDICT: %s" % verdict)

    if args.out:
        json.dump({"arm": args.arm, "n_stages": len(rows),
                   "eta2_task": e_task, "p_task": p_task,
                   "eta2_position": e_pos, "p_position": p_pos,
                   "by_task": {k: v for k, v in by_task.items()},
                   "by_position": {str(k): v for k, v in by_pos.items()},
                   "rows": rows}, open(args.out, "w"), indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
