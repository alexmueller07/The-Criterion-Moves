"""Aggregate the robustness runs against the paper's pre-stated confirmation targets.

Arms (checkpoint-dir naming in results/):
  SEQ  seed17: S1..S4        seed23: seq_s2_k1..4   seed31: seq_s3_k1..4
  G    seed17: G1..G4        seed23: g_s2_k1..4     seed31: g_s3_k1..4
  SEQ-REV seed17: seq_rev_k1..4  (order: vizwiz->flickr->textvqa->scienceqa)
  G-REV   seed17: g_rev_k1..4
  Controls: vw_only_k1, tv_only_k1 (single-stage from base)
  Base: S0. Joint: J1..J4. ER: E1..E4.

Confirmation targets (stated in the paper as in-progress):
  T1  criterion dose-response replicates across seeds (SEQ) and order (REV):
      per-stage yes-rate/criterion moves track the stage's answer statistics.
  T2  reverse-order endpoint: after final ScienceQA (no yes/no content) the
      criterion ends near base (|c - c(S0)| small vs the VizWiz-final endpoint).
  T3  anchor criterion-freeze replicates across seeds: sum|dc| (G) << sum|dc| (SEQ).
  T4  S1->S3 length-controlled CHAIR rise: per-seed sign + pooled contrast.
  T5  single-task controls: c(vw_only) vs c(S4); c(tv_only) vs c(S2) -
      does the last stage alone reproduce the sequential endpoint criterion?
Missing checkpoints are reported as ABSENT, never imputed.
"""
import json
import os
import sys
from statistics import NormalDist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pilot"))
from metrics_pope import parse_yn
from metrics_chair import extract_object_mentions, extract_objects

ND = NormalDist()
RES = "results"

SEQ_ARMS = {"seed17": ["S1", "S2", "S3", "S4"],
            "seed23": [f"seq_s2_k{k}" for k in range(1, 5)],
            "seed31": [f"seq_s3_k{k}" for k in range(1, 5)]}
G_ARMS = {"seed17": ["G1", "G2", "G3", "G4"],
          "seed23": [f"g_s2_k{k}" for k in range(1, 5)],
          "seed31": [f"g_s3_k{k}" for k in range(1, 5)]}
REV = {"SEQ-REV": [f"seq_rev_k{k}" for k in range(1, 5)],
       "G-REV": [f"g_rev_k{k}" for k in range(1, 5)]}
STAGE_TASKS_FWD = ["scienceqa", "textvqa", "flickr", "vizwiz"]
STAGE_TASKS_REV = ["vizwiz", "flickr", "textvqa", "scienceqa"]


def clip(p, n):
    return min(max(p, 1 / (2 * n)), 1 - 1 / (2 * n))


def sdt(ck):
    p = os.path.join(RES, ck, "pope_gen.jsonl")
    if not os.path.exists(p):
        return None
    hits = fas = nyes = nno = say = n = 0
    for line in open(p):
        r = json.loads(line)
        pred = parse_yn(r["output"])
        if pred is None:
            continue
        n += 1
        say += pred == "yes"
        if r["gt"] == "yes":
            nyes += 1
            hits += pred == "yes"
        else:
            nno += 1
            fas += pred == "yes"
    H, FA = clip(hits / nyes, nyes), clip(fas / nno, nno)
    zh, zf = ND.inv_cdf(H), ND.inv_cdf(FA)
    return {"H": H, "FA": FA, "dprime": zh - zf, "c": -0.5 * (zh + zf),
            "yes": say / n}


_GT = None


def chair60(ck):
    global _GT
    p = os.path.join(RES, ck, "chair_gen.jsonl")
    if not os.path.exists(p):
        return None
    if _GT is None:
        _GT = json.load(open(os.path.join(RES, "coco_gt.json")))
    M = H = 0
    per = {}
    for line in open(p):
        r = json.loads(line)
        cid = str(r["coco_id"])
        gto = set(_GT[cid]["objects"])
        for c in _GT[cid]["captions"]:
            gto |= extract_objects(c)
        m = extract_object_mentions(" ".join(r["output"].split()[:60]))
        h = sum(1 for x in m if x not in gto)
        M += len(m)
        H += h
        per[cid] = (len(m), h)
    return {"chair_i60": H / max(1, M), "per_image": per}


def fmt(v, spec=".3f"):
    return "ABSENT" if v is None else format(v, spec)


report = {"targets": {}}
print("=" * 70)
print("T1/T3: criterion trajectories and swing sums per seed")
S0 = sdt("S0")
print(f"  S0 base: c={S0['c']:+.3f} yes={S0['yes']*100:.1f}%")
swing_sums = {}
for name, arms in (("SEQ", SEQ_ARMS), ("G", G_ARMS)):
    for seed, cks in arms.items():
        vals = [sdt(c) for c in cks]
        cs = [S0["c"]] + [v["c"] if v else None for v in vals]
        if any(v is None for v in cs):
            print(f"  {name}/{seed}: ABSENT checkpoints {[c for c, v in zip(cks, vals) if v is None]}")
            continue
        swings = [abs(cs[i + 1] - cs[i]) for i in range(4)]
        post = sum(swings[1:])
        swing_sums[(name, seed)] = (sum(swings), post)
        traj = " ".join(f"{v:+.2f}" for v in cs[1:])
        print(f"  {name}/{seed}: c per stage [{traj}]  sum|dc|={sum(swings):.3f}  post-settle={post:.3f}")
report["targets"]["T3_swing_sums"] = {f"{k[0]}/{k[1]}": v for k, v in swing_sums.items()}
seq_posts = [v[1] for k, v in swing_sums.items() if k[0] == "SEQ"]
g_posts = [v[1] for k, v in swing_sums.items() if k[0] == "G"]
if seq_posts and g_posts:
    print(f"  T3 verdict: post-settle swings SEQ {min(seq_posts):.3f}-{max(seq_posts):.3f} "
          f"vs G {min(g_posts):.3f}-{max(g_posts):.3f} across seeds")

print("=" * 70)
print("T1-rev/T2: reverse-order trajectories (stages: vizwiz->flickr->textvqa->scienceqa)")
for name, cks in REV.items():
    vals = [sdt(c) for c in cks]
    line = " ".join(f"{c}:c={fmt(v['c'] if v else None, '+.3f')},yes={fmt(v['yes']*100 if v else None, '.1f')}%"
                    for c, v in zip(cks, vals))
    print(f"  {name}: {line}")
    if vals[-1] is not None:
        d_end = abs(vals[-1]["c"] - S0["c"])
        fwd_end = abs(sdt("S4")["c"] - S0["c"]) if name == "SEQ-REV" else abs(sdt("G4")["c"] - S0["c"])
        print(f"    T2: |c_end - c_base| = {d_end:.3f} (forward-order endpoint gap was {fwd_end:.3f})")
        report["targets"].setdefault("T2", {})[name] = {"end_gap": d_end, "fwd_gap": fwd_end}

print("=" * 70)
print("T4: S1->S3 length-controlled CHAIR rise per seed (stage-1 vs stage-3 ckpt)")
for seed, cks in SEQ_ARMS.items():
    a, b = chair60(cks[0]), chair60(cks[2])
    if a is None or b is None:
        print(f"  SEQ/{seed}: ABSENT")
        continue
    print(f"  SEQ/{seed}: chair_i@60 stage1={a['chair_i60']:.4f} stage3={b['chair_i60']:.4f} "
          f"delta={b['chair_i60']-a['chair_i60']:+.4f}")
    report["targets"].setdefault("T4", {})[seed] = {
        "s1": a["chair_i60"], "s3": b["chair_i60"],
        "delta": b["chair_i60"] - a["chair_i60"]}

print("=" * 70)
print("T5: single-task controls vs sequential endpoints")
for ctrl, ref, label in (("vw_only_k1", "S4", "VizWiz-only vs S4"),
                         ("tv_only_k1", "S2", "TextVQA-only vs S2")):
    cv, rv = sdt(ctrl), sdt(ref)
    if cv is None:
        print(f"  {label}: control ABSENT")
        continue
    print(f"  {label}: c={cv['c']:+.3f} vs {rv['c']:+.3f} (base {S0['c']:+.3f}); "
          f"yes {cv['yes']*100:.1f}% vs {rv['yes']*100:.1f}%")
    report["targets"].setdefault("T5", {})[label] = {"ctrl_c": cv["c"], "ref_c": rv["c"]}

os.makedirs("analysis/diag", exist_ok=True)
json.dump(report, open("analysis/diag/robustness_aggregation.json", "w"), indent=1)
print("written analysis/diag/robustness_aggregation.json")
