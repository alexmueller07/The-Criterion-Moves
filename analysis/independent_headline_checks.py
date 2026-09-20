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
bound = (worst[0] / 9000) / (1 / math.sqrt(2 * math.pi))
check("worst-case criterion movement", bound, 0.028, 3)
wc = ARMS[worst[1][0]]
rng = max(wc[str(s)]["pope"]["c"] for s in range(1, 7)) - min(wc[str(s)]["pope"]["c"] for s in range(1, 7))
check("that cell's criterion range", rng, 0.395, 3)
check("margin (x)", rng / bound, 14, 0)

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
    check("Qwen endpoint dc from base", st.mean(Q[s]["5"]["c"] - QB_C for s in Q), -0.079, 3)
    check("Qwen endpoint dd' from base", st.mean(Q[s]["5"]["dprime"] - QB_D for s in Q), -0.048, 3)
    check("Qwen worst |dd'| from base (margin 0.30)",
          max(max(abs(Q[s][k]["dprime"] - QB_D) for k in usable[s]) for s in Q), 0.084, 3)
    for seed, want in (("s17", 87.7), ("s23", 11.8), ("s31", 79.0)):
        check("Qwen stage-4 parse rate, seed " + seed[1:] + " (%)",
              100 * Q[seed]["4"]["parse_rate"], want, 1)
        if Q[seed]["4"]["usable"]:
            FAILS.append("stage 4 of " + seed + " should be flagged unusable")

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
