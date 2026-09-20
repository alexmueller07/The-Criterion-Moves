"""Per-backbone criterion-trajectory readout, written 2026-09-20.

Runs the same quantities the paper reports for LLaVA-1.5-7B over ANY backbone
present in the readout, so the second-backbone arms can be read the moment they
land without hand-editing anything. Stage counts are read from the data, not
assumed: the LLaVA/UCIT matrix is six stages, the Qwen sequence is five, and the
post-settling convention (drop the k=1 settling step from the untuned base) is
applied to whatever length a cell actually has.

Validated by running it on llava15, where every printed value must match the
numbers in the paper. Usage:
    python3 analysis/backbone_trajectory_readout.py [backbone ...]
"""
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
READOUT = os.path.join(HERE, "readout", "fs_aggregate.json")


def z(p, n=4500):
    """Inverse normal CDF by bisection, with a 1/(2n) correction at the edges."""
    p = min(max(p, 1.0 / (2 * n)), 1 - 1.0 / (2 * n))
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def stages_of(cell):
    """Consecutive stage indices present from 1 upward, with a scored POPE entry."""
    out = []
    k = 1
    while str(k) in cell and "pope" in cell[str(k)]:
        out.append(cell[str(k)]["pope"])
        k += 1
    return out


def zroc(ps):
    """OLS of z(H) on z(FA) across a cell's stages -> (b, R^2, d_a)."""
    xs = [z(p["FA"]) for p in ps]
    ys = [z(p["H"]) for p in ps]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ssr = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    return b, r2, math.sqrt(2 / (1 + b * b)) * a


def report(name, bb):
    arms = bb.get("arms", {})
    complete = {k: stages_of(v) for k, v in sorted(arms.items())}
    complete = {k: v for k, v in complete.items() if len(v) >= 3}
    print("\n=== %s ===" % name)
    if not complete:
        print("  no cell has >=3 scored stages yet (arms present: %d)" % len(arms))
        return
    base = bb.get("base", {}).get("pope")
    if base:
        print("  untuned base: c = %+.4f  F1 = %.4f" % (base["c"], base.get("f1", float("nan"))))
    groups = {}
    for k, ps in complete.items():
        groups.setdefault(k.split("|")[0], []).append((k, ps))
    for arm, cells in sorted(groups.items()):
        drift, endc, meanc, dr, r2s, das = [], [], [], [], [], []
        for _, ps in cells:
            cs = [p["c"] for p in ps]
            # Delta c_k = c_k - c_{k-1} for k = 2..K. The excluded settling step is
            # S_0 -> S_1, from the untuned base, which is not in this list at all --
            # so every consecutive pair HERE is post-settling. Indexing from 2
            # silently drops a real transition (0.640 instead of the paper's 0.781).
            drift.append(sum(abs(cs[i] - cs[i - 1]) for i in range(1, len(cs))))
            endc.append(cs[-1])
            meanc.append(st.mean(abs(c) for c in cs))
            ds = [p["dprime"] for p in ps if "dprime" in p]
            if len(ds) == len(ps):
                dr.append(max(ds) - min(ds))
            f = zroc(ps)
            if f:
                r2s.append(f[1])
                das.append(f[2])
        n = len(cells)
        stage_lens = sorted({len(ps) for _, ps in cells})
        print("  %-10s n=%d  stages=%s" % (arm, n, stage_lens))
        print("      post-settling sum|dc| (k>=2) : %.3f   [%.3f, %.3f]"
              % (st.mean(drift), min(drift), max(drift)))
        print("      endpoint c                   : %+.3f  [%+.3f, %+.3f]"
              % (st.mean(endc), min(endc), max(endc)))
        print("      mean |c|                     : %.3f" % st.mean(meanc))
        if dr:
            print("      per-cell max d' range        : %.3f" % st.mean(dr))
        if r2s:
            print("      z-ROC R^2 mean/min           : %.3f / %.3f" % (st.mean(r2s), min(r2s)))
            print("      d_a mean (sd)                : %.3f (%.3f)"
                  % (st.mean(das), st.stdev(das) if len(das) > 1 else 0.0))


def main():
    bbs = json.load(open(READOUT))["backbones"]
    want = sys.argv[1:] or sorted(bbs)
    for name in want:
        if name not in bbs:
            print("\n=== %s ===\n  not present in the readout" % name)
            continue
        report(name, bbs[name])


if __name__ == "__main__":
    main()
