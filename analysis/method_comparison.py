"""Four-arm method comparison against the pre-stated predictions in
PILOT_FINDINGS.md: SEQ (S) vs JOINT (J) vs Experience Replay (E) vs
faithfulness anchor (F). Computes, per checkpoint: POPE pooled H/FA/d'/c and
yes-rate; TextVQA refusal-leakage rate; per-task accuracy; CHAIR summary.
Then the four prediction verdicts + the ER kill-check.
"""
import json
import os
import sys
from statistics import NormalDist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pilot"))
from metrics_pope import parse_yn

ND = NormalDist()
ARMS = {"S": ["S0", "S1", "S2", "S3", "S4"],
        "E": ["S0", "E1", "E2", "E3", "E4"],
        "F": ["S0", "F1", "F2", "F3", "F4"],
        "G": ["S0", "G1", "G2", "G3", "G4"],
        "J": ["S0", "J1", "J2", "J3", "J4"]}
TASKS = ["scienceqa", "textvqa", "flickr", "vizwiz"]


def clip(p, n):
    return min(max(p, 1 / (2 * n)), 1 - 1 / (2 * n))


def sdt(ck):
    hits = fas = nyes = nno = say_yes = n = 0
    for line in open(f"results/{ck}/pope_gen.jsonl"):
        r = json.loads(line)
        pred = parse_yn(r["output"])
        if pred is None:
            continue
        n += 1
        say_yes += pred == "yes"
        if r["gt"] == "yes":
            nyes += 1
            hits += pred == "yes"
        else:
            nno += 1
            fas += pred == "yes"
    H, FA = clip(hits / nyes, nyes), clip(fas / nno, nno)
    zh, zf = ND.inv_cdf(H), ND.inv_cdf(FA)
    return {"H": round(H, 4), "FA": round(FA, 4),
            "dprime": round(zh - zf, 3), "c": round(-0.5 * (zh + zf), 3),
            "yes_rate": round(say_yes / n, 4)}


def leakage(ck):
    n = hit = 0
    for line in open(f"results/{ck}/textvqa_gen.jsonl"):
        r = json.loads(line)
        n += 1
        hit += r["output"].strip().lower() == "unanswerable"
    return round(hit / n, 4)


def tasks(ck):
    return {t: json.load(open(f"results/{ck}/task_{t}.json"))["score"] for t in TASKS}


def chair(ck):
    d = json.load(open(f"results/{ck}/chair.json"))
    return {"chair_i": d["chair_i"], "tok": d["mean_new_tokens"]}


rows = {}
for arm, cks in ARMS.items():
    for ck in cks:
        if ck not in rows:
            rows[ck] = {**sdt(ck), "leak_tvqa": leakage(ck), **chair(ck), "tasks": tasks(ck)}

print(f"{'ckpt':4} {'d’':>6} {'c':>7} {'yes%':>6} {'leakTV%':>8} {'CHAIRi':>7} {'tok':>6}  new-task-acc")
stage_task = {1: "scienceqa", 2: "textvqa", 3: "flickr", 4: "vizwiz"}
for arm in ("S", "E", "F", "G", "J"):
    for i, ck in enumerate(ARMS[arm]):
        r = rows[ck]
        nt = f"{r['tasks'][stage_task[i]]:.3f} ({stage_task[i]})" if i else ""
        print(f"{ck:4} {r['dprime']:>6.3f} {r['c']:>7.3f} {r['yes_rate']*100:>5.1f}% "
              f"{r['leak_tvqa']*100:>7.1f}% {r['chair_i']:>7.4f} {r['tok']:>6.1f}  {nt}")
    print()

print("=== prediction verdicts ===")
for arm in ("S", "E", "F", "G"):
    cs = [rows[ck]["c"] for ck in ARMS[arm]]
    swings = [abs(cs[i + 1] - cs[i]) for i in range(4)]
    print(f"{arm}: |dc| per stage {['%.3f' % s for s in swings]}  sum={sum(swings):.3f}  "
          f"endpoint c={cs[-1]:+.3f}  leak4={rows[ARMS[arm][4]]['leak_tvqa']*100:.1f}%")
new_acc = {arm: [rows[ARMS[arm][k]]["tasks"][stage_task[k]] for k in (1, 2, 3, 4)]
           for arm in ("S", "E", "F", "G")}
print("plasticity (new-task acc per stage):")
for arm, v in new_acc.items():
    print(f"  {arm}: {['%.3f' % x for x in v]}  mean={sum(v)/4:.3f}")

out = {"rows": rows, "arms": ARMS}
json.dump(out, open("analysis/diag/method_comparison.json", "w"), indent=1)
print("written analysis/diag/method_comparison.json")
