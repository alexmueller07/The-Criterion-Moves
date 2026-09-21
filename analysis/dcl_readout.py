"""DCL (MLLM-CL domain continual learning) readout: the second benchmark.

Scores the DCL sequential arm and its matched joint arm with fs_common.score_pope
-- the same pooled-POPE scorer behind every UCIT number -- so c, d' and F1 mean
exactly what they mean in the paper. The untuned base is the shared LLaVA base
cell: POPE is the instrument on both benchmarks, only the training stream varies.

Endpoint, fixed in FULLSTUDY_PREREG.md (2026-09-20) before any DCL number
existed: MEAN post-settling per-step |delta c|, sum_{k>=2}|c_k - c_{k-1}| / (K-1).
UCIT sums five transitions and DCL four, so a sum is not comparable across the two
and would flatter DCL's shorter stream by construction. "Reproduces" means SEQ's
per-step drift exceeds its matched JOINT arm's, in the same direction as UCIT.
One seed, one ordering: existence and direction only, never a magnitude comparison
between benchmarks, and no pooling with UCIT cells.

    python3 dcl_readout.py            # on the cluster; writes readout/dcl_readout.json
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fs_common as F  # noqa: E402

ROOT = os.path.expanduser("~/cl-halluc")
RES = os.path.join(ROOT, "results_fs")
OUT = os.path.join(ROOT, "readout", "dcl_readout.json")
PARSE_FLOOR = 0.95
UCIT_PER_STEP = {"SEQ": 0.7814 / 5, "JOINT (n=2)": 0.2279 / 5}


def done(d):
    return os.path.exists(os.path.join(d, "EVAL_DONE")) and os.path.exists(os.path.join(d, "pope_gen.jsonl"))


def seq_cells(tag):
    out = {}
    for d in glob.glob(os.path.join(RES, tag + "_k*")):
        m = re.search(r"_k(\d+)$", d)
        if m and done(d):
            out[int(m.group(1))] = d
    return out


def joint_cells(tag):
    """Joint checkpoints are stage-indexed by sorted step order; _final is the last."""
    steps = []
    for d in glob.glob(os.path.join(RES, tag + "_step*")):
        m = re.search(r"_step(\d+)$", d)
        if m and done(d):
            steps.append((int(m.group(1)), d))
    fin = os.path.join(RES, tag + "_final")
    cells = {i + 1: d for i, (_, d) in enumerate(sorted(steps))}
    if done(fin):
        cells[len(cells) + 1] = fin
    return cells


def trajectory(cells):
    rows = []
    for k in F.contiguous_prefix(set(cells)):
        p = F.score_pope(os.path.join(cells[k], "pope_gen.jsonl"))
        p["stage"] = k
        p["parse_rate"] = round(p["n_parsed"] / p["n_total"], 4)
        p["usable"] = p["parse_rate"] >= PARSE_FLOOR
        rows.append(p)
    return rows


def summarize(name, rows, n_stages):
    cs = [r["c"] for r in rows]
    sw = F.swings(cs)
    # swings() over stages 1..K gives K-1 transitions, all of them post-settling
    # (the settling step is base -> S1, which is not in this list). Same fix as
    # backbone_trajectory_readout.py: indexing from 2 would silently drop one.
    post = sum(sw) if sw else None
    s = {"arm": name, "stages": [r["stage"] for r in rows], "complete": len(rows) == n_stages,
         "all_usable": all(r["usable"] for r in rows),
         "c": cs, "dprime": [r["dprime"] for r in rows], "f1": [r["f1"] for r in rows],
         "yes_rate": [r["yes_rate"] for r in rows], "parse_rate": [r["parse_rate"] for r in rows],
         "post_settling_sum": round(post, 4) if post is not None else None,
         "per_step": round(post / len(sw), 4) if sw else None,
         "endpoint_c": cs[-1] if cs else None, "endpoint_f1": rows[-1]["f1"] if rows else None,
         "endpoint_dprime": rows[-1]["dprime"] if rows else None}
    return s


def main():
    man = json.load(open(os.path.join(ROOT, "data_dcl", "dcl_manifest.json")))
    order = man["orders"]["d1"]
    base = F.score_pope(os.path.join(RES, "llava15_base", "pope_gen.jsonl"))
    seq = summarize("SEQ", trajectory(seq_cells("fsD_seq_d1_s17")), len(order))
    jnt = summarize("JOINT", trajectory(joint_cells("fsD_joint_d1_s17")), len(order))
    res = {"benchmark": "MLLM-CL/DCL", "order_d1": order, "seed": 17,
           "base": {"c": base["c"], "dprime": base["dprime"], "f1": base["f1"]},
           "answer_stats": {d: man["tasks"][d]["answer_stats"] for d in order},
           "SEQ": seq, "JOINT": jnt, "ucit_per_step_reference": UCIT_PER_STEP}
    verdict = "INCOMPLETE"
    if seq["complete"] and jnt["complete"]:
        if not (seq["all_usable"] and jnt["all_usable"]):
            verdict = "UNUSABLE (format collapse below parse floor)"
        else:
            verdict = ("REPRODUCES" if seq["per_step"] > jnt["per_step"] else "DOES NOT REPRODUCE")
    res["verdict"] = verdict
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    print("DCL d1 order: %s   base c %+.4f" % (" -> ".join(order), base["c"]))
    for s in (seq, jnt):
        print("\n%s  stages %s  complete=%s  usable=%s" % (s["arm"], s["stages"], s["complete"], s["all_usable"]))
        for i, k in enumerate(s["stages"]):
            print("  S%d %-4s c %+.4f  d' %.3f  F1 %.4f  yes %.3f  parsed %.3f"
                  % (k, order[k - 1], s["c"][i], s["dprime"][i], s["f1"][i], s["yes_rate"][i], s["parse_rate"][i]))
        if s["per_step"] is not None:
            print("  post-settling sum %.4f over %d transitions -> per-step %.4f"
                  % (s["post_settling_sum"], len(s["stages"]) - 1, s["per_step"]))
    print("\nUCIT per-step reference: SEQ %.4f  JOINT %.4f" % (UCIT_PER_STEP["SEQ"], UCIT_PER_STEP["JOINT (n=2)"]))
    print("VERDICT (prereg 2026-09-20): %s" % verdict)


if __name__ == "__main__":
    main()
