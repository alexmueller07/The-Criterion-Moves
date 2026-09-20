"""Language-prior share of criterion drift (lit-sweep-1 diagnostic, "M1-blind").

For each checkpoint k of an arm we have two POPE logit dumps (pilot/eval_gen.py
--dump_logits): the real one (pope_logits.jsonl) and a BLIND twin
(pope_logits_blind.jsonl) where every image was replaced by constant gray, so the
decision statistic g = z_yes - z_no carries no visual evidence. Under the paper's
additive-bias model a stage shifts g by a class-independent b_k. The blind shift
  b_k^blind = mean g_blind(k) - mean g_blind(base)
is the part of that bias the model applies with NO image; the ratio
  rho_k = b_k^blind / b_k            (b_k = mean g_real(k) - mean g_real(base))
is the language-prior share of the drift. rho ~ 1: drift lives in the answer prior
(image-independent) -> the label-free/blind probe is a valid, image-domain-free
estimator and the anchor-eval image-overlap objection dissolves. rho << 1: drift is
evidence-conditional -> a blind probe cannot see it.

Also reports the blind criterion trajectory c_blind(k) (yes/no thresholding of the
blind gaps against POPE labels) and d'_blind (should sit near 0: no evidence).

usage: blind_prior_share.py --results <results_fs> --runtag <RUNTAG> --base <label>
       [--stages k1,k2,...] --out <json>
Files: <results>/<label>/pope_logits.jsonl and pope_logits_blind.jsonl.
Stdlib only; ABSENT cells are reported, never imputed.
"""
import argparse
import json
import math
import os
from statistics import NormalDist, mean, pstdev

ND = NormalDist()


def load(path):
    rows = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[str(r["id"])] = r
    return rows


def _z(p, n):
    p = min(max(p, 1.0 / (2 * n)), 1 - 1.0 / (2 * n))
    return ND.inv_cdf(p)


def sdt(rows):
    yes = [r for r in rows.values() if str(r.get("gt", "")).lower() == "yes"]
    no = [r for r in rows.values() if str(r.get("gt", "")).lower() == "no"]
    if not yes or not no:
        return None
    H = sum(r["gap"] > 0 for r in yes) / len(yes)
    FA = sum(r["gap"] > 0 for r in no) / len(no)
    zh, zf = _z(H, len(yes)), _z(FA, len(no))
    return {"H": H, "FA": FA, "dprime": zh - zf, "c": -0.5 * (zh + zf),
            "mean_gap": mean(r["gap"] for r in rows.values()),
            "std_gap": pstdev([r["gap"] for r in rows.values()]), "n": len(rows)}


def cell(results, label):
    real = load(os.path.join(results, label, "pope_logits.jsonl"))
    blind = load(os.path.join(results, label, "pope_logits_blind.jsonl"))
    return real, blind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--runtag", required=True)
    ap.add_argument("--base", required=True, help="base label, e.g. llava15_base")
    ap.add_argument("--stages", default=None, help="comma list, default k1..k6 that exist")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base_real, base_blind = cell(a.results, a.base)
    out = {"runtag": a.runtag, "base": a.base, "stages": {}, "notes": []}
    if base_real is None or base_blind is None:
        out["notes"].append("BASE ABSENT: need both pope_logits.jsonl and pope_logits_blind.jsonl for the base")
        json.dump(out, open(a.out, "w"), indent=1)
        print(json.dumps(out, indent=1))
        return
    b_real0, b_blind0 = sdt(base_real), sdt(base_blind)
    out["base_stats"] = {"real": b_real0, "blind": b_blind0}
    stages = a.stages.split(",") if a.stages else [f"k{i}" for i in range(1, 7)]
    for k in stages:
        label = f"{a.runtag}_{k}"
        real, blind = cell(a.results, label)
        if real is None or blind is None:
            out["stages"][k] = "ABSENT"
            continue
        # align ids across the four dumps
        ids = set(real) & set(blind) & set(base_real) & set(base_blind)
        if len(ids) < 100:
            out["stages"][k] = {"status": "TOO_FEW_ALIGNED", "n": len(ids)}
            continue
        g = lambda rows: mean(rows[i]["gap"] for i in ids)
        b_k = g(real) - g(base_real)
        b_blind = g(blind) - g(base_blind)
        rho = b_blind / b_k if abs(b_k) > 1e-6 else float("nan")
        # per-item: does the blind shift explain the real shift item by item?
        d_real = [real[i]["gap"] - base_real[i]["gap"] for i in ids]
        d_blind = [blind[i]["gap"] - base_blind[i]["gap"] for i in ids]
        mr, mb = mean(d_real), mean(d_blind)
        cov = sum((x - mr) * (y - mb) for x, y in zip(d_real, d_blind)) / len(ids)
        vr = sum((x - mr) ** 2 for x in d_real) / len(ids)
        vb = sum((y - mb) ** 2 for y in d_blind) / len(ids)
        corr = cov / math.sqrt(vr * vb) if vr > 0 and vb > 0 else float("nan")
        out["stages"][k] = {
            "b_real": b_k, "b_blind": b_blind, "rho_prior_share": rho,
            "item_corr_real_vs_blind_shift": corr,
            "real": sdt({i: real[i] for i in ids}), "blind": sdt({i: blind[i] for i in ids}),
            "n_aligned": len(ids),
        }
    # summary line
    rhos = [v["rho_prior_share"] for v in out["stages"].values()
            if isinstance(v, dict) and "rho_prior_share" in v and not math.isnan(v["rho_prior_share"])]
    if rhos:
        out["summary"] = {"rho_min": min(rhos), "rho_max": max(rhos), "rho_mean": mean(rhos),
                          "reading": ("prior-borne (rho~1): blind probe valid, image-overlap objection dissolves"
                                      if mean(rhos) > 0.7 else
                                      "mixed" if mean(rhos) > 0.3 else
                                      "evidence-conditional (rho<<1): blind probe cannot see the drift")}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[blind] {a.runtag}: " + ", ".join(
        f"{k}: rho={v['rho_prior_share']:+.2f} (b_real={v['b_real']:+.2f}, b_blind={v['b_blind']:+.2f}, d'_blind={v['blind']['dprime']:.2f})"
        if isinstance(v, dict) and "rho_prior_share" in v else f"{k}: {v if isinstance(v, str) else v.get('status')}"
        for k, v in out["stages"].items()))
    if "summary" in out:
        print("[blind] summary:", out["summary"]["reading"], f"(rho mean {out['summary']['rho_mean']:+.2f})")


if __name__ == "__main__":
    main()
