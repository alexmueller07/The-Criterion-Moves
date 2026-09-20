#!/usr/bin/env python3
"""Are task identity and sequence depth actually CROSSED by our three orderings?

WHY THIS EXISTS (2026-09-12). The headline of the order-effects analysis is that
task identity explains the one-step criterion movement (eta^2 = 0.731, p < 0.0001)
while sequence depth does not (0.049, p = 0.73). A sweep of the task-ordering
literature surfaced the confound that would destroy that result:

    if the orderings do not balance task DIFFICULTY against POSITION, then
    eta^2(task) silently absorbs difficulty and eta^2(depth) is suppressed --
    which is exactly the pattern we report.

Concretely (arXiv 2608.18066): "The default order exhibits an implicit easy-to-hard
curriculum... In general, tasks of different difficulties are not distributed evenly
in the default order." A single canonical ordering has this problem by construction,
because each task sits at exactly one depth and the two factors are then the same
variable wearing two labels.

THE ANSWER IS A PROPERTY OF THE DESIGN, NOT OF THE DATA, so it can be checked
without any results: o1 is canonical, o2 is its reverse, o3 is a fixed shuffle
(seed 41). This script prints the task-by-position incidence and asserts the
condition that makes the two factors separable -- every task must be SCORED at two
or more distinct depths. (Scored means position >= 2: `order_effects.py --skip_first`
drops the first stage, whose dc has no predecessor.)

This is a partial crossing, not a complete one: 3 of the 6! = 720 possible
orderings. It is sufficient to separate the factors, and the paper should claim
exactly that and no more.
"""
import collections
import random
import sys

CANON = ["ArxivQA", "CLEVR-Math", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"]


def orderings():
    """Exactly as fullstudy/ucit_prep.py freezes them into the manifest."""
    o3 = list(CANON)
    random.Random(41).shuffle(o3)
    return {"o1": list(CANON), "o2": list(reversed(CANON)), "o3": o3}


def main():
    orders = orderings()
    for k, v in orders.items():
        print("%s: %s" % (k, " -> ".join(v)))
    print()

    at = collections.defaultdict(set)
    cnt = collections.defaultdict(collections.Counter)
    for v in orders.values():
        for pos, t in enumerate(v, 1):
            at[t].add(pos)
            cnt[t][pos] += 1

    print("%-13s %s   all positions      scored (p>=2)"
          % ("task", " ".join("p%d" % p for p in range(1, 7))))
    print("-" * 74)
    for t in CANON:
        row = " ".join(" %d" % cnt[t][p] if cnt[t][p] else " ." for p in range(1, 7))
        print("%-13s %s   %-18s %s"
              % (t, row, sorted(at[t]), sorted(p for p in at[t] if p >= 2)))

    scored = {t: [p for p in at[t] if p >= 2] for t in CANON}
    worst = min(len(v) for v in scored.values())
    print("\ndistinct SCORED depths per task: min %d, max %d"
          % (worst, max(len(v) for v in scored.values())))

    bad = [t for t, v in scored.items() if len(v) < 2]
    assert not bad, (
        "task identity and depth are CONFOUNDED for %s: each is scored at only one "
        "depth, so eta^2(task) and eta^2(depth) are not separable and the order-"
        "effects headline may not be quoted. Fix the orderings, not the analysis."
        % ", ".join(bad))

    print("\nCROSSED: every task is scored at >= 2 distinct depths, so eta^2(task)")
    print("cannot be a relabelling of depth and the difficulty-rides-position")
    print("confound is controlled by design. Note this is a PARTIAL crossing --")
    print("3 of 720 possible orderings -- which is enough to separate the two")
    print("factors and is all the paper may claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
