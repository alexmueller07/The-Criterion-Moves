"""Do forgetting and criterion drift answer to the same factor? They do not.

Written 2026-09-16 after a blind reviewer observed, correctly, that the paper
asserted "a forgetting curve plotted against depth is plotting the wrong axis"
while never plotting a forgetting curve. The full study scores all six UCIT tasks
at every stage of every cell, so the curve was computable from the released
aggregate all along.

POST-HOC. This was not pre-registered. It is reported as a reframing of an
over-reaching claim, not as a confirmatory result.

Two quantities, measured on the same nine sequential cells and the same streams:

  forgetting        acc(task, at the stage that trained it) - acc(task, later stage)
  criterion step    c_k - c_{k-1}, post-settling (stage 1 dropped)

and two candidate factors: which task, and how long ago / how deep.

Open-ended tasks (Flickr30k, VizWiz) are excluded from the forgetting side: they
are scored by exact-match containment and sit at ~0, so a difference of two near-
zero numbers is not forgetting. That leaves four tasks and 90 (task, later-stage)
observations against the criterion side's 45 transitions.

eta^2 is an in-sample variance share and is biased upward by level count, so the
permutation p-values carry the comparison, each computed under its own factor's
level structure.
"""
import argparse, collections, json, os, random


def eta2(groups, vals):
    n = len(vals)
    gm = sum(vals) / n
    sst = sum((x - gm) ** 2 for x in vals)
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for g in set(groups):
        sub = [vals[i] for i in range(n) if groups[i] == g]
        ssb += len(sub) * ((sum(sub) / len(sub)) - gm) ** 2
    return ssb / sst


def perm_p(groups, vals, observed, B, rng):
    g = list(groups)
    hits = 0
    for _ in range(B):
        rng.shuffle(g)
        if eta2(g, vals) >= observed:
            hits += 1
    return (hits + 1) / (B + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "readout", "fs_aggregate.json"))
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    arms = json.load(open(args.agg))["backbones"]["llava15"]["arms"]

    forget, crit, excluded = [], [], set()
    for k, v in sorted(arms.items()):
        if not k.startswith("seq|"):
            continue
        if not all(str(s) in v and "tasks" in v[str(s)] and "pope" in v[str(s)] for s in range(1, 7)):
            continue
        trained = {v[str(s)]["task_trained"]: s for s in range(1, 7)}
        for t, kt in trained.items():
            if v[str(kt)]["tasks"][t]["open_ended"]:
                excluded.add(t)
                continue
            a_train = v[str(kt)]["tasks"][t]["acc"]
            for s in range(kt + 1, 7):
                forget.append({"cell": k, "task": t, "elapsed": s - kt, "depth": s,
                               "forgetting": a_train - v[str(s)]["tasks"][t]["acc"]})
        for s in range(2, 7):
            crit.append({"cell": k, "task": v[str(s)]["task_trained"], "depth": s,
                         "dc": v[str(s)]["pope"]["c"] - v[str(s - 1)]["pope"]["c"]})

    fv = [r["forgetting"] for r in forget]
    cv = [r["dc"] for r in crit]
    rep = {"_note": "POST-HOC. Not pre-registered. eta^2 is descriptive; the permutation p carries it.",
           "open_ended_excluded_from_forgetting": sorted(excluded),
           "n_forgetting_obs": len(forget), "n_criterion_transitions": len(crit),
           "forgetting": {}, "criterion_step": {}}
    for name, key in (("task", "task"), ("elapsed", "elapsed"), ("depth", "depth")):
        g = [r[key] for r in forget]
        e = eta2(g, fv)
        rep["forgetting"][name] = {"eta2": round(e, 4), "p": round(perm_p(g, fv, e, args.B, rng), 5),
                                   "levels": len(set(g))}
    for name, key in (("task", "task"), ("depth", "depth")):
        g = [r[key] for r in crit]
        e = eta2(g, cv)
        rep["criterion_step"][name] = {"eta2": round(e, 4), "p": round(perm_p(g, cv, e, args.B, rng), 5),
                                       "levels": len(set(g))}

    # within-task monotone check: does forgetting grow with elapse holding task fixed?
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in forget:
        by[r["task"]][r["elapsed"]].append(r["forgetting"])
    rises = {}
    for t, d in by.items():
        ms = [sum(v) / len(v) for _, v in sorted(d.items())]
        rises[t] = bool(ms[-1] > ms[0])
    rep["within_task_forgetting_rises_with_elapse"] = rises
    rep["within_task_rises_count"] = "%d/%d" % (sum(rises.values()), len(rises))
    rep["mean_forgetting_by_elapsed"] = {
        str(e): round(sum(v) / len(v), 4)
        for e, v in sorted(collections.defaultdict(
            list, {e: [r["forgetting"] for r in forget if r["elapsed"] == e]
                   for e in sorted({r["elapsed"] for r in forget})}).items())}

    print(json.dumps(rep, indent=1))
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
