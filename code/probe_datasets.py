"""Probe HF dataset repos before building anything: splits, sizes, columns,
one example's field types/values (no images saved). Output: probe_report.json.
Candidate repos per asset are tried in order; first success wins.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.gate_checks import run_all_gates

CANDIDATES = {
    "scienceqa": ["derek-thomas/ScienceQA"],
    "textvqa": ["lmms-lab/textvqa", "facebook/textvqa"],
    "flickr": ["nlphuji/flickr30k", "lmms-lab/flickr30k"],
    "vizwiz": ["lmms-lab/VizWiz-VQA", "HuggingFaceM4/VizWiz", "Multimodal-Fatima/VizWiz"],
    "pope": ["lmms-lab/POPE"],
}


def describe_example(ex):
    out = {}
    for k, v in ex.items():
        t = type(v).__name__
        if t in ("str", "int", "float", "bool"):
            s = str(v)
            out[k] = {"type": t, "value": s[:120]}
        elif isinstance(v, list):
            out[k] = {"type": f"list[{len(v)}]",
                      "head": [str(x)[:60] for x in v[:3]]}
        else:
            out[k] = {"type": t, "repr": repr(v)[:120]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_all_gates(args.out, min_free_gb=30.0, need_gpu=False)
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    report = {}
    for asset, repos in CANDIDATES.items():
        report[asset] = []
        for repo in repos:
            entry = {"repo": repo}
            try:
                cfgs = get_dataset_config_names(repo)
                entry["configs"] = cfgs[:10]
                cfg = cfgs[0] if cfgs and cfgs[0] != "default" else None
                splits = get_dataset_split_names(repo, cfg)
                entry["splits"] = splits
                split_info = {}
                for sp in splits:
                    try:
                        ds = load_dataset(repo, cfg, split=f"{sp}[:2]")
                        split_info[sp] = {"columns": ds.column_names,
                                          "example": describe_example(ds[0])}
                    except Exception as e:
                        split_info[sp] = {"error": repr(e)[:300]}
                entry["split_info"] = split_info
                entry["ok"] = True
            except Exception as e:
                entry["ok"] = False
                entry["error"] = repr(e)[:300]
            report[asset].append(entry)
            if entry.get("ok"):
                break

    path = os.path.join(args.out, "probe_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)[:8000], flush=True)
    print(f"[probe] written {path}", flush=True)


if __name__ == "__main__":
    main()
