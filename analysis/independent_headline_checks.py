"""Independent re-derivation of the paper's headline quantities.

Written 2026-09-16 so that the LLM-use statement's claim -- that the headline
quantities were re-derived by an implementation independent of the scripts that
first produced them -- is backed by a file a reader can run.

It reads only the released readouts (analysis/readout/fs_aggregate.json and
analysis/readout/matched_cells_critbal_ewc.json) and shares no code with the
analysis scripts: its own z-transform, its own OLS, its own sums. Every check
asserts against the value printed in the paper and exits non-zero on a mismatch.
The z-ROC model comparison (maximum likelihood) lives in the sibling
independent_zroc_modelcomp_check.py and is run from here.

Rounding convention: half-up, as in the paper.
"""
import json
import math
import os
import statistics as st
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = json.load(open(os.path.join(HERE, "readout", "fs_aggregate.json")))["backbones"]["llava15"]
ARMS, BASE = AGG["arms"], AGG["base"]["pope"]
CSTAR = 0.0878
FAILS = []


def rnd(x, nd):
    return float(Decimal(repr(x)).quantize(Decimal("1." + "0" * nd), rounding=ROUND_HALF_UP))


def check(label, got, want, nd):
    ok = rnd(got, nd) == want
    print("  %-52s paper %-9s got %-9s %s" % (label, want, rnd(got, nd), "OK" if ok else "MISMATCH"))
    if not ok:
        FAILS.append(label)


def cells(prefix):
    out = {}
    for k, v in sorted(ARMS.items()):
        if k.startswith(prefix + "|") and all(str(s) in v and "pope" in v[str(s)] for s in range(1, 7)):
            out[k] = [v[str(s)]["pope"] for s in range(1, 7)]
    return out


def z(p, n=4500):
    p = min(max(p, 1.0 / (2 * n)), 1 - 1.0 / (2 * n))
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


SEQ, ANC, JNT = cells("seq"), cells("anchor"), cells("joint")

print("z-ROC fits (OLS of z(H) on z(FA) over six stages)")
for name, cs, r2w, minw, daw, sdw in (("sequential", SEQ, 0.957, 0.844, 2.159, 0.040),
                                      ("joint", JNT, 0.945, 0.919, 2.181, 0.042)):
    r2s, das, rms = [], [], []
    for stages in cs.values():
        xs = [z(p["FA"]) for p in stages]
        ys = [z(p["H"]) for p in stages]
        mx, my = st.mean(xs), st.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        ssr = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
        r2s.append(1 - ssr / sum((y - my) ** 2 for y in ys))
        rms.append(math.sqrt(ssr / len(xs)))
        das.append(math.sqrt(2 / (1 + b * b)) * a)
    check(name + " mean R^2", st.mean(r2s), r2w, 3)
    check(name + " min R^2", min(r2s), minw, 3)
    check(name + " mean d_a", st.mean(das), daw, 3)
    check(name + " sd d_a", st.stdev(das), sdw, 3)
    if name == "sequential":
        check("sequential mean residual RMSE", st.mean(rms), 0.018, 3)

print("\nTable 2 (per-cell values, then mean over cells)")


def post_settling(stages):
    c = [p["c"] for p in stages]
    return sum(abs(c[i] - c[i - 1]) for i in range(1, 6))


for name, cs, ps, ec, mc, mcs, f1 in (("SEQ", SEQ, 0.781, 0.090, 0.196, 0.146, 0.8625),
                                      ("anchor", ANC, 0.364, 0.697, 0.731, 0.643, 0.7963),
                                      ("JOINT", JNT, 0.228, 0.088, 0.099, 0.048, 0.8667)):
    check(name + " post-settling sum|dc|", st.mean(post_settling(s) for s in cs.values()), ps, 3)
    check(name + " endpoint c", st.mean(s[-1]["c"] for s in cs.values()), ec, 3)
    check(name + " mean |c|", st.mean(st.mean(abs(p["c"]) for p in s) for s in cs.values()), mc, 3)
    # Table 3 prints placement against c* (the prereg forbids ranking placement
    # from c = 0), so the printed row is checked too, not only the c = 0 form.
    check(name + " mean |c - c*|", st.mean(st.mean(abs(p["c"] - CSTAR) for p in s) for s in cs.values()), mcs, 3)
    check(name + " endpoint F1", st.mean(s[-1]["f1"] for s in cs.values()), f1, 4)

seq_ps = st.mean(post_settling(s) for s in SEQ.values())
anc_ps = st.mean(post_settling(s) for s in ANC.values())
jnt_ps = st.mean(post_settling(s) for s in JNT.values())
check("anchor closes this share of the SEQ-to-joint gap (%)", 100 * (seq_ps - anc_ps) / (seq_ps - jnt_ps), 75, 0)
check("anchor worse placed than SEQ at every stage (cells)",
      sum(1 for k in SEQ if max(abs(p["c"]) for p in SEQ[k]) < min(abs(p["c"]) for p in ANC[k.replace("seq|", "anchor|")])), 9, 0)

print("\nTable 3 (matched single cells, from matched_cells_critbal_ewc.json)")
MC = json.load(open(os.path.join(HERE, "readout", "matched_cells_critbal_ewc.json")))
for key, ps, ec, dist, f1 in (("fsL_ewc_o1_s17", 0.4963, 0.134, 0.046, 0.8622),
                              ("fsL_lwf_o1_s17", 0.3819, 0.092, 0.004, 0.8618),
                              ("fsL_critbal_o1_s17", 0.2836, 0.246, 0.158, 0.8380),
                              ("fsL_ewc_o1_s31", 0.9166, 0.198, 0.110, 0.8573)):
    r = MC[key]
    check(key + " post-settling", r["post_settling"], ps, 4)
    check(key + " endpoint c", r["endpoint_c"], ec, 3)
    check(key + " |c - c*|", abs(r["endpoint_c"] - CSTAR), dist, 3)
    check(key + " endpoint F1", r["endpoint_f1"], f1, 4)

print("\nTask versus depth, sequential criterion step (post-settling)")
groups_t, groups_d, vals = [], [], []
for stages_key, v in ARMS.items():
    if not stages_key.startswith("seq|"):
        continue
    for s in range(2, 7):
        vals.append(v[str(s)]["pope"]["c"] - v[str(s - 1)]["pope"]["c"])
        groups_t.append(v[str(s)]["task_trained"])
        groups_d.append(s)


def eta2(groups):
    gm = st.mean(vals)
    sst = sum((x - gm) ** 2 for x in vals)
    ssb = sum(len([x for x, g in zip(vals, groups) if g == q]) *
              (st.mean([x for x, g in zip(vals, groups) if g == q]) - gm) ** 2 for q in set(groups))
    return ssb / sst


check("eta^2, task identity", eta2(groups_t), 0.731, 3)
check("eta^2, depth", eta2(groups_d), 0.049, 3)

print("\nParser audit (scorer exclusions)")
tot = excl = 0
worst = (-1, ("", 0))
for k, v in ARMS.items():
    for s in range(1, 7):
        p = v.get(str(s), {}).get("pope")
        if not p:
            continue
        tot += p["n_total"]
        e = p["n_total"] - p["n_parsed"]
        excl += e
        worst = max(worst, (e, (k, s)))
check("POPE answers scored", tot, 1080000, 0)
check("rows excluded", excl, 602, 0)
check("exclusion rate (%)", 100 * excl / tot, 0.056, 3)
check("worst stage-cell, rows", worst[0], 101, 0)
wc = ARMS[worst[1][0]]
# Density at the cell's own operating point, and the 4500 items of one class --
# not phi(0) over all 9000, which understates the bound by ~1.8x.
zfa = z(wc[str(worst[1][1])]["pope"]["FA"])
bound = 0.5 * (worst[0] / 4500) / (math.exp(-zfa * zfa / 2) / math.sqrt(2 * math.pi))
check("worst-case criterion movement", bound, 0.052, 3)
rng = max(wc[str(s)]["pope"]["c"] for s in range(1, 7)) - min(wc[str(s)]["pope"]["c"] for s in range(1, 7))
check("that cell's criterion range", rng, 0.395, 3)
check("margin (x)", rng / bound, 7.6, 1)

print("\nRandom-walk null for the z-ROC linearity")
try:
    RW = json.load(open(os.path.join(HERE, "readout", "zroc_randomwalk_null.json")))
except FileNotFoundError:
    print("  zroc_randomwalk_null.json absent -- SKIPPED")
    RW = None
if RW:
    check("observed mean R^2 (9 sequential cells)", RW["observed_mean_r2"], 0.957, 3)
    for lab, want_p in (("isotropic_random_walk", 0.004), ("random_walk_with_drift", 0.012)):
        check(lab + ": P(single cell >= observed)", RW["nulls"][lab]["p_single_cell_ge_observed_mean"], want_p, 3)
        if RW["nulls"][lab]["mean_over_cells_max"] >= RW["observed_mean_r2"]:
            FAILS.append(lab + ": a simulated nine-cell study reached the observed mean")
    print("  %-52s %s" % ("no simulated 9-cell study reached the observed mean",
                          "OK" if not any("nine-cell" in f for f in FAILS) else "MISMATCH"))

print("\nSecond backbone (Qwen2.5-VL-7B, five-task UCIT ordering, three seeds)")
QB_C, QB_D = 0.5479, 2.6658          # untuned Qwen base, same 9000 POPE items
try:
    Q = json.load(open(os.path.join(HERE, "readout", "qwen_q5_trajectory.json")))
except FileNotFoundError:
    print("  qwen_q5_trajectory.json absent -- second-backbone checks SKIPPED")
    Q = None
if Q:
    usable = {s: [k for k in sorted(Q[s], key=int) if Q[s][k]["usable"]] for s in Q}
    crs = [max(Q[s][k]["c"] for k in usable[s]) - min(Q[s][k]["c"] for k in usable[s]) for s in Q]
    drs = [max(Q[s][k]["dprime"] for k in usable[s]) - min(Q[s][k]["dprime"] for k in usable[s]) for s in Q]
    check("Qwen criterion range (mean over seeds)", st.mean(crs), 0.162, 3)
    check("Qwen d' range (mean over seeds)", st.mean(drs), 0.112, 3)
    check("Qwen criterion-to-d' range ratio", st.mean(crs) / st.mean(drs), 1.45, 2)
    _lc = [max(v[str(k)]["pope"]["c"] for k in range(1, 7)) - min(v[str(k)]["pope"]["c"] for k in range(1, 7))
           for kk, v in ARMS.items() if kk.startswith("seq|")]
    _ld = [max(v[str(k)]["pope"]["dprime"] for k in range(1, 7)) - min(v[str(k)]["pope"]["dprime"] for k in range(1, 7))
           for kk, v in ARMS.items() if kk.startswith("seq|")]
    # LLaVA on the SAME definition as the Qwen ratio (mean within-cell range); the
    # paper once quoted 2.36x here, which was the absolute-range ratio mislabelled.
    check("LLaVA criterion-to-d' range ratio (same definition)", st.mean(_lc) / st.mean(_ld), 2.66, 2)
    check("LLaVA / Qwen absolute criterion range", st.mean(_lc) / st.mean(crs), 2.35, 2)
    check("Qwen endpoint dc from base", st.mean(Q[s]["5"]["c"] - QB_C for s in Q), -0.079, 3)
    check("Qwen endpoint dd' from base", st.mean(Q[s]["5"]["dprime"] - QB_D for s in Q), -0.048, 3)
    check("Qwen worst |dd'| from base (margin 0.30)",
          max(max(abs(Q[s][k]["dprime"] - QB_D) for k in usable[s]) for s in Q), 0.084, 3)
    for seed, want in (("s17", 87.7), ("s23", 11.8), ("s31", 79.0)):
        check("Qwen stage-4 parse rate, seed " + seed[1:] + " (%)",
              100 * Q[seed]["4"]["parse_rate"], want, 1)
        if Q[seed]["4"]["usable"]:
            FAILS.append("stage 4 of " + seed + " should be flagged unusable")

print("\nLate experiments (fs_aggregate_v2.json, dcl_readout.json; added 2026-09-21)")
V2 = json.load(open(os.path.join(HERE, "readout", "fs_aggregate_v2.json")))["backbones"]["llava15"]
V2A, V2B = V2["arms"], V2["base"]["pope"]
# v2 must reproduce the canonical readout before any new cell is trusted
check("v2 reproduces all canonical cells (mismatches)",
      sum(1 for k, v in ARMS.items() for s_ in v if s_.isdigit() and "pope" in v[s_]
          and abs(v[s_]["pope"]["c"] - V2A[k][s_]["pope"]["c"]) > 1e-9), 0, 0)
O1 = ["ArxivQA", "CLEVR-Math", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"]
SQ = V2A["seq|o1|s17"]
single = {t: V2A["single_%s|o1|s17" % t]["1"]["pope"]["c"] for t in O1}
seqc = {t: SQ[str(k)]["pope"]["c"] for k, t in enumerate(O1, 1)}
for t, (ws, wq) in zip(O1, ((0.201, 0.212), (0.346, 0.195), (0.443, 0.318), (0.068, 0.029), (0.227, -0.009), (0.308, 0.068))):
    check("E4 single-task c, " + t, single[t], ws, 3)
    check("E4 sequential c at its position, " + t, seqc[t], wq, 3)
floor = abs(single["ArxivQA"] - seqc["ArxivQA"])
check("E4 run-level floor (position-1 replication)", floor, 0.011, 3)
active = ["Flickr30k", "IconQA", "ImageNet-R", "VizWiz"]
check("E4 mean active gap / floor", st.mean(abs(single[t] - seqc[t]) for t in active) / floor, 14.4, 1)
check("E4 positions where single > sequential (of 5)", sum(single[t] > seqc[t] for t in O1[1:]), 5, 0)
check("E4 smallest deep gap", min(single[t] - seqc[t] for t in O1[1:]), 0.04, 2)
check("E4 largest deep gap", max(single[t] - seqc[t] for t in O1[1:]), 0.24, 2)
carry, incr = [], []
for k, t in enumerate(O1[1:], 2):
    prev = SQ[str(k - 1)]["pope"]["c"]
    act = t != "CLEVR-Math"
    carry.append(abs(seqc[t] - (single[t] if act else prev)))
    incr.append(abs(seqc[t] - (prev + single[t] - V2B["c"])))
check("E4 carry model mean residual / floor", st.mean(carry) / floor, 11.8, 1)
check("E4 increment model mean residual / floor", st.mean(incr) / floor, 11.2, 1)
check("E4 Flickr30k shift mid-stream", seqc["Flickr30k"] - seqc["CLEVR-Math"], 0.124, 3)
check("E4 Flickr30k shift from base", single["Flickr30k"] - V2B["c"], 0.012, 3)
# E3 recency test on the same controls
ch = {t: V2A["single_%s|o1|s17" % t]["1"]["chair"]["chair_i60"] for t in O1}
for t, w in (("CLEVR-Math", 0.1109), ("IconQA", 0.1066), ("ArxivQA", 0.1041), ("ImageNet-R", 0.1006), ("VizWiz", 0.0635), ("Flickr30k", 0.0617)):
    check("E3 test: single-task CHAIR@60, " + t, ch[t], w, 4)
pred = ["CLEVR-Math", "ArxivQA", "VizWiz", "IconQA", "Flickr30k", "ImageNet-R"]
got = sorted(O1, key=lambda t: -ch[t])
check("E3 test: Spearman rho vs predicted order",
      1 - 6 * sum((pred.index(t) - got.index(t)) ** 2 for t in O1) / (6 * 35), 0.60, 2)
for o, (a, b), wp, (wlo, whi) in (("o1", ("ArxivQA", "ImageNet-R"), -0.0035, (-0.0588, -0.0288)),
                                   ("o2", ("VizWiz", "CLEVR-Math"), 0.0474, (0.0161, 0.0265))):
    d = [V2A[k]["5"]["chair"]["chair_i60"] - V2A[k]["1"]["chair"]["chair_i60"] for k in V2A if k.startswith("seq|%s|" % o)]
    check("E3 test: %s control-predicted gap" % o, ch[b] - ch[a], wp, 4)
    check("E3 test: %s seed band low" % o, min(d), wlo, 4)
    check("E3 test: %s seed band high" % o, max(d), whi, 4)
    if min(d) <= ch[b] - ch[a] <= max(d):
        FAILS.append("E3 test %s gap should fall OUTSIDE its seed band" % o)
# ER and the third JOINT cell (single-cell table rows)
def row(key):
    v = V2A[key]
    cs = [v[str(k)]["pope"]["c"] for k in range(1, 7)]
    return (sum(abs(cs[i] - cs[i - 1]) for i in range(1, 6)), cs[-1], abs(cs[-1] - CSTAR),
            v["6"]["pope"]["f1"], max(abs(v[str(k)]["pope"]["dprime"] - V2B["dprime"]) for k in range(1, 7)))
for key, want in (("er|o1|s17", (0.7663, 0.090, 0.003, 0.8609, 0.202)),
                  ("joint|o1|s31", (0.1254, 0.166, 0.078, 0.8615, 0.109))):
    for lab, g, w, nd in zip(("drift", "endpoint c", "|c-c*|", "endpoint F1", "max |dd'|"), row(key), want, (4, 3, 3, 4, 3)):
        check("%s %s" % (key.split("|")[0].upper(), lab), g, w, nd)
check("ER drift rise over matched SEQ (%)", 100 * (row("er|o1|s17")[0] / 0.5451 - 1), 41, 0)
# DCL
D = json.load(open(os.path.join(HERE, "readout", "dcl_readout.json")))
for arm, w_ps, w_dd, w_cr, w_dr in (("SEQ", 0.132, 0.095, 0.528, 0.151), ("JOINT", 0.161, 0.046, 0.377, 0.110)):
    c_, d_ = D[arm]["c"], D[arm]["dprime"]
    check("DCL %s per-step |dc|" % arm, st.mean(abs(c_[i] - c_[i - 1]) for i in range(1, 5)), w_ps, 3)
    check("DCL %s per-step |dd'|" % arm, st.mean(abs(d_[i] - d_[i - 1]) for i in range(1, 5)), w_dd, 3)
    check("DCL %s c range" % arm, max(c_) - min(c_), w_cr, 3)
    check("DCL %s d' range" % arm, max(d_) - min(d_), w_dr, 3)
    check("DCL %s range ratio" % arm, (max(c_) - min(c_)) / (max(d_) - min(d_)), 3.5 if arm == "SEQ" else 3.4, 1)
    check("DCL %s all stages parsed" % arm, min(D[arm]["parse_rate"]), 1.0, 3)
check("DCL verdict string", 1 if D["verdict"] == "DOES NOT REPRODUCE" else 0, 1, 0)
# Figure 1, panel (c): the median-drift run
MR = V2A["seq|o3|s31"]
accs = [0.5 * (V2B["H"] + 1 - V2B["FA"])] + [0.5 * (MR[str(k)]["pope"]["H"] + 1 - MR[str(k)]["pope"]["FA"]) for k in range(1, 7)]
dcs = [0] + [MR[str(k)]["pope"]["c"] - V2B["c"] for k in range(1, 7)]
dds = [0] + [MR[str(k)]["pope"]["dprime"] - V2B["dprime"] for k in range(1, 7)]
check("Fig 1c accuracy min", min(accs), 0.86, 2)
check("Fig 1c accuracy max", max(accs), 0.87, 2)
check("Fig 1c c range", max(dcs) - min(dcs), 0.40, 2)
check("Fig 1c d' range", max(dds) - min(dds), 0.12, 2)

print("\nExternal-review numbers (2026-09-21)")
import random as _rnd
from collections import defaultdict as _dd
BASE = AGG["base"]["pope"]
def _cells(arm):
    return {k: [v[str(s_)]["pope"] for s_ in range(1, 7)] for k, v in ARMS.items() if k.startswith(arm + "|")}
def _fit(pts):
    xs = [z(p_["FA"]) for p_ in pts]; ys = [z(p_["H"]) for p_ in pts]; mx, my = st.mean(xs), st.mean(ys)
    b_ = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a_ = my - b_ * mx
    return a_, b_, math.sqrt(st.mean([(y - a_ - b_ * x) ** 2 for x, y in zip(xs, ys)])), xs
_Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
_phi = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
# Table 3 rates and the abstract's percentages
for arm, wH, wHr, wF, wFr in (("seq", 0.846, (0.81, 0.87), 0.116, (0.08, 0.14)),
                              ("anchor", 0.683, (0.66, 0.72), 0.032, (0.02, 0.05)),
                              ("joint", 0.851, (0.84, 0.86), 0.112, (0.11, 0.12))):
    E = [c_[-1] for c_ in _cells(arm).values()]
    check(arm + " endpoint H", st.mean(e["H"] for e in E), wH, 3)
    check(arm + " endpoint H min", min(e["H"] for e in E), wHr[0], 2)
    check(arm + " endpoint H max", max(e["H"] for e in E), wHr[1], 2)
    check(arm + " endpoint FA", st.mean(e["FA"] for e in E), wF, 3)
    check(arm + " endpoint FA min", min(e["FA"] for e in E), wFr[0], 2)
    check(arm + " endpoint FA max", max(e["FA"] for e in E), wFr[1], 2)
check("base H", BASE["H"], 0.771, 3)
check("base FA", BASE["FA"], 0.054, 3)
acc = lambda e: 0.5 * (e["H"] + 1 - e["FA"])
check("anchor accuracy deficit vs SEQ (points)",
      100 * (st.mean(acc(c_[-1]) for c_ in _cells("seq").values()) - st.mean(acc(c_[-1]) for c_ in _cells("anchor").values())), 4.0, 1)
# fixed-ROC prediction of the d' decline, and the base on the sequential curves
zH0, zF0 = z(BASE["H"]), z(BASE["FA"])
pred, dab, das, res = [], [], [], []
for pts in _cells("seq").values():
    a_, b_, r_, xs = _fit(pts)
    pred.append((zH0 + b_ * (xs[-1] - zF0) - xs[-1]) - (zH0 - zF0))
    dab.append(math.sqrt(2 / (1 + b_ * b_)) * (zH0 - b_ * zF0)); das.append(math.sqrt(2 / (1 + b_ * b_)) * a_)
    res.append(zH0 - (a_ + b_ * zF0))
check("fixed-ROC predicted endpoint dd'", st.mean(pred), -0.117, 3)
check("observed endpoint dd'", st.mean(c_[-1]["dprime"] - BASE["dprime"] for c_ in _cells("seq").values()), -0.121, 3)
check("base d_a on sequential slopes", st.mean(dab), 2.170, 3)
check("sequential d_a", st.mean(das), 2.159, 3)
check("base residual from sequential lines (mean)", st.mean(res), 0.011, 3)
check("base residual from sequential lines (sd)", st.pstdev(res), 0.025, 3)
# residual RMSE against an independent-binomial floor
for arm, wr, wf in (("seq", 0.018, 0.024), ("joint", 0.007, 0.023), ("anchor", 0.032, 0.024)):
    rm, fl = [], []
    for pts in _cells(arm).values():
        a_, b_, r_, xs = _fit(pts); rm.append(r_)
        fl.append(math.sqrt(st.mean([(p_["H"] * (1 - p_["H"]) / 4500) / _phi(z(p_["H"])) ** 2
                                     + b_ * b_ * (p_["FA"] * (1 - p_["FA"]) / 4500) / _phi(z(p_["FA"])) ** 2 for p_ in pts])) * math.sqrt(4 / 6))
    check(arm + " residual RMSE", st.mean(rm), wr, 3)
    check(arm + " independent-binomial floor", st.mean(fl), wf, 3)
    if arm == "anchor":
        check("anchor RMSE / floor", st.mean(rm) / st.mean(fl), 1.3, 1)
# accuracy-maximizing criterion on each run's own fitted ROC
def _opt(a_, b_):
    best = max(((0.5 * (_Phi(a_ + b_ * (-4 + i * 0.001)) + 1 - _Phi(-4 + i * 0.001)), -4 + i * 0.001) for i in range(8001)))
    zf = best[1]; return -0.5 * (a_ + b_ * zf + zf), best[0]
for arm, wc, wgap in (("seq", 0.15, 0.001), ("joint", 0.18, 0.003), ("anchor", None, 0.017)):
    co, gap, dist = [], [], []
    for pts in _cells(arm).values():
        a_, b_, r_, xs = _fit(pts); c_o, acc_o = _opt(a_, b_)
        co.append(c_o); gap.append(acc_o - acc(pts[-1])); dist.append(abs(pts[-1]["c"] - c_o))
    if wc is not None:
        check(arm + " accuracy-maximizing c (mean)", st.mean(co), wc, 2)
    check(arm + " endpoint accuracy below its peak", st.mean(gap), wgap, 3)
    if arm == "anchor":
        check("anchor endpoint distance from its peak c", st.mean(dist), 0.38, 2)
check("Fisher exact 0/9 vs 3/9, one-sided", math.comb(9, 3) / math.comb(18, 3), 0.10, 2)
check("anchor share of SEQ->JOINT gap, n=3 joint (%)",
      100 * (0.781 - 0.364) / (0.781 - st.mean([0.2400, 0.2157, 0.1254])), 71, 0)
# unit-level permutation tests (seeds averaged; seed 17, B = 20000)
def _eta(lab, val):
    g = _dd(list)
    for l_, v_ in zip(lab, val): g[l_].append(v_)
    m_ = st.mean(val); return sum(len(x) * (st.mean(x) - m_) ** 2 for x in g.values()) / sum((v_ - m_) ** 2 for v_ in val)
def _perm(lab, val, B=20000):
    o_ = _eta(lab, val); r_ = _rnd.Random(17); v_ = list(val); ge = 0
    for _ in range(B):
        r_.shuffle(v_); ge += _eta(lab, v_) >= o_ - 1e-12
    return o_, (ge + 1) / (B + 1)
ORD = {"o1": ["ArxivQA", "CLEVR-Math", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"],
       "o2": ["VizWiz", "ImageNet-R", "IconQA", "Flickr30k", "CLEVR-Math", "ArxivQA"],
       "o3": ["ImageNet-R", "VizWiz", "ArxivQA", "CLEVR-Math", "Flickr30k", "IconQA"]}
for arm, wt, wpt, wd, wpd in (("seq", 0.883, 0.001, 0.059, 0.96), ("anchor", 0.624, 0.08, 0.093, 0.89)):
    u = _dd(list)
    for k, v in ARMS.items():
        if k.startswith(arm + "|"):
            cs = [v[str(s_)]["pope"]["c"] for s_ in range(1, 7)]
            for s_ in range(2, 7): u[(k.split("|")[1], s_)].append(cs[s_ - 1] - cs[s_ - 2])
    ks = sorted(u); vals = [st.mean(u[k_]) for k_ in ks]
    et, pt = _perm([ORD[o][s_ - 1] for o, s_ in ks], vals); ed, pd = _perm([s_ for o, s_ in ks], vals)
    check(arm + " unit-level task eta2", et, wt, 3); check(arm + " unit-level task p", pt, wpt, 3 if wpt < 0.01 else 2)
    check(arm + " unit-level depth eta2", ed, wd, 3); check(arm + " unit-level depth p", pd, wpd, 2)
fu, pos = _dd(list), {}
for k, v in ARMS.items():
    if not k.startswith("seq|"): continue
    o = k.split("|")[1]
    tr = {v[str(s_)]["task_trained"]: s_ for s_ in range(1, 7)}
    for t, kt in tr.items():
        pos[(o, t)] = kt
        if v[str(kt)]["tasks"][t]["open_ended"]: continue
        for s_ in range(kt + 1, 7):
            fu[(o, t, s_)].append(v[str(kt)]["tasks"][t]["acc"] - v[str(s_)]["tasks"][t]["acc"])
ks = sorted(fu); vals = [st.mean(fu[k_]) for k_ in ks]
check("forgetting units", len(ks), 30, 0)
for name, lab, we, wp in (("elapsed", [s_ - pos[(o, t)] for o, t, s_ in ks], 0.399, 0.009),
                          ("task", [t for o, t, s_ in ks], 0.175, 0.17),
                          ("absolute depth", [s_ for o, t, s_ in ks], 0.131, 0.46)):
    e_, p_ = _perm(lab, vals)
    check("forgetting unit-level " + name + " eta2", e_, we, 3)
    check("forgetting unit-level " + name + " p", p_, wp, 3 if wp < 0.01 else 2)

print("\nz-ROC model comparison (maximum likelihood; separate script)")
out = subprocess.run([sys.executable, os.path.join(HERE, "independent_zroc_modelcomp_check.py")],
                     capture_output=True, text=True, cwd=os.path.dirname(HERE)).stdout
for line in out.splitlines():
    if line.startswith(("seq", "anchor", "joint")):
        print("  " + line)
want = {"seq": ("2.750", "9/9"), "anchor": ("5.827", "6/9"), "joint": (None, "2/2")}
for arm, (g2, aic) in want.items():
    line = next((l for l in out.splitlines() if l.startswith(arm)), "")
    ok = (g2 is None or ("mean G2 " + g2) in line) and ("(G2<8): " + aic) in line
    print("  %-52s %s" % (arm + " model comparison", "OK" if ok else "MISMATCH"))
    if not ok:
        FAILS.append(arm + " model comparison")

print("\n%s" % ("ALL HEADLINE QUANTITIES REPRODUCE" if not FAILS else "MISMATCHES: %s" % FAILS))
sys.exit(1 if FAILS else 0)
