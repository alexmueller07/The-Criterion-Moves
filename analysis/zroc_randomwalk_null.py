"""Is the z-ROC linearity an artifact of fitting six points?

A natural objection to Result 1: with only six checkpoints per cell, a high
R^2 in z-space may be easy to obtain for any smooth-ish trajectory, so the
linearity may carry no information about the evidence distributions.

This tests the objection directly rather than arguing about it. It builds the
null the objection describes -- a random walk in (z(H), z(FA)) with step sizes
matched to the ones we actually observe, optionally with a per-cell systematic
drift -- and asks how often that null reaches the R^2 we report.

The step sizes are estimated from the data itself, so the null is not a straw
man: it wanders exactly as far per stage as our checkpoints do. What it lacks
is the constraint that the wandering stay on one line.

Run:  python3 analysis/zroc_randomwalk_null.py [--B 20000] [--out readout.json]
"""
import argparse
import json
import math
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def z(p, n=4500):
    p = min(max(p, 1.0 / (2 * n)), 1 - 1.0 / (2 * n))
    lo, hi = -10.0, 10.0
    for _ in range(80):
        m = (lo + hi) / 2
        if 0.5 * (1 + math.erf(m / math.sqrt(2))) < p:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def r2(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ssr = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - my) ** 2 for y in ys)
    return (1 - ssr / sst) if sst > 0 else float("nan")


def observed(readout):
    arms = json.load(open(readout))["backbones"]["llava15"]["arms"]
    r2s, dh, df = [], [], []
    for k, v in arms.items():
        if not k.startswith("seq|"):
            continue
        zh = [z(v[str(s)]["pope"]["H"]) for s in range(1, 7)]
        zf = [z(v[str(s)]["pope"]["FA"]) for s in range(1, 7)]
        r2s.append(r2(zf, zh))
        dh += [zh[i] - zh[i - 1] for i in range(1, 6)]
        df += [zf[i] - zf[i - 1] for i in range(1, 6)]
    return r2s, st.stdev(dh), st.stdev(df)


def simulate_cell(rng, sd_h, sd_f, drift):
    zh, zf = [0.0], [0.0]
    bh = rng.gauss(0, sd_h * 0.5) if drift else 0.0
    bf = rng.gauss(0, sd_f * 0.5) if drift else 0.0
    for _ in range(5):
        zh.append(zh[-1] + bh + rng.gauss(0, sd_h))
        zf.append(zf[-1] + bf + rng.gauss(0, sd_f))
    return r2(zf, zh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", default=os.path.join(HERE, "readout", "fs_aggregate.json"))
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    r2s, sd_h, sd_f = observed(a.readout)
    obs_mean, obs_min, n_cells = st.mean(r2s), min(r2s), len(r2s)
    print("observed: %d sequential cells, mean R^2 %.4f (min %.4f)" % (n_cells, obs_mean, obs_min))
    print("per-step sd estimated from the data: z(H) %.4f, z(FA) %.4f\n" % (sd_h, sd_f))

    report = {"n_cells": n_cells, "observed_mean_r2": obs_mean, "observed_min_r2": obs_min,
              "step_sd_zH": sd_h, "step_sd_zFA": sd_f, "B": a.B, "seed": a.seed, "nulls": {}}

    for label, drift in (("isotropic_random_walk", False), ("random_walk_with_drift", True)):
        rng = random.Random(a.seed)
        singles, means = [], []
        for _ in range(a.B):
            cell = [simulate_cell(rng, sd_h, sd_f, drift) for _ in range(n_cells)]
            cell = [c for c in cell if c == c]
            if not cell:
                continue
            singles.extend(cell)
            means.append(st.mean(cell))
        singles.sort(); means.sort()
        p_single = (sum(1 for v in singles if v >= obs_mean) + 1) / (len(singles) + 1)
        p_mean = (sum(1 for m in means if m >= obs_mean) + 1) / (len(means) + 1)
        report["nulls"][label] = {
            "single_cell_median_r2": singles[len(singles) // 2],
            "single_cell_p99_r2": singles[int(0.99 * len(singles))],
            "p_single_cell_ge_observed_mean": p_single,
            "mean_over_cells_median": means[len(means) // 2],
            "mean_over_cells_max": means[-1],
            "p_mean_ge_observed": p_mean,
        }
        print("%s (B=%d):" % (label.replace("_", " "), a.B))
        print("   one cell:      median R^2 %.3f, 99th pct %.3f, P(>= %.3f) = %.4f"
              % (singles[len(singles) // 2], singles[int(0.99 * len(singles))], obs_mean, p_single))
        print("   mean of %d:     median %.3f, best of %d studies %.3f, P(>= %.3f) = %.5f\n"
              % (n_cells, means[len(means) // 2], len(means), means[-1], obs_mean, p_mean))

    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
