"""Interventional dose manipulation for the criterion-drift mechanism.

The pilot's mechanism claim (analysis/PILOT_FINDINGS.md) is that each stage's
labelled ANSWER STATISTICS (yes/no imbalance, refusal fraction) are the dose
that moves the POPE criterion c. This script builds a variant of the pilot suite
whose training targets carry a CONTROLLED answer prior while every other aspect
of the task (prompts, images, row content, n_train, stage order) stays fixed:

  neutral    balance yes/no counts (subsample the majority to the minority),
             cap refusal-class rows at <= --neutral_refusal_cap (2%), refill
             from answerable non-yes/no rows so n_train is preserved.
  amplified  the opposite: raise the refusal fraction of refusal-bearing tasks
             (VizWiz) to --amp_refusal_frac (60%), push the yes:no ratio of
             yes/no-bearing tasks (TextVQA) toward yes to --amp_yes_ratio (4:1)
             by subsampling 'no'; n_train preserved.

Rows are SELECTED, never edited: every written line is a byte-identical copy of
a source line (the script re-reads its own output and asserts this). Refilling
to n_train when the supply of eligible unique rows runs out duplicates rows
(--refill dup, default; every duplicate is counted in the manifest recipe) or,
with --refill shrink, writes fewer rows and records the smaller n_train.

Output is a DROP-IN DATA ROOT (default <data_root>/tasks_<variant>):
  <out_root>/tasks/<t>/train.jsonl          the variant training file
  <out_root>/tasks/<t>/{images,val.jsonl,...} relative symlinks to the source
  <out_root>/{grounding,pope,chair,...}     relative symlinks to the source
  <out_root>/<variant>_manifest.json        pilot manifest with n_train and
                                            answer_stats recomputed from the
                                            WRITTEN files + variant/recipe
so an arm runs with DATADIR=<out_root> MANIFEST=<out_root>/<variant>_manifest.json
and fullstudy/run_arm.py's hard-coded  $DATADIR/tasks/<t>/train.jsonl  resolves
without code changes (relative image paths inside rows resolve through the
images symlink). ER buffer dirs (er*) are deliberately NOT linked so a replay
arm on a variant rebuilds its buffers from the variant files.

Deterministic: selection uses only random.Random(str).random() (string seeding
is hash-randomisation-proof; random() is the one stream Python guarantees
stable across versions). Two runs, or two interpreters, give identical bytes.

Cluster (vgi2, data at ~/cl-halluc/data, tasks/<t>/train.jsonl):
    python code/fullstudy/neutralize_prior.py --data_root ~/cl-halluc/data \
        --manifest ~/cl-halluc/data/pilot_manifest.json --variant neutral
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

# pilot/metrics_task.norm_answer is the repo's canonical answer normalisation
# (lowercase, strip articles + punctuation, collapse whitespace). Two layouts:
# repo (this file in fullstudy/, metrics_task in ../pilot/) and cluster
# (~/cl-halluc/code/fullstudy/ with metrics_task.py at ../, because the sync
# ships pilot/* as code/*; see run_arm.py's CODE/TRAIN convention).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "..", "pilot"), os.path.join(_HERE, "..")):
    if os.path.isfile(os.path.join(_cand, "metrics_task.py")):
        sys.path.insert(0, _cand)
        break
else:
    raise SystemExit("metrics_task.py not found next to fullstudy/ (repo: pilot/, "
                     "cluster: ~/cl-halluc/code/)")
from metrics_task import norm_answer  # noqa: E402

SEED = 17
VARIANTS = ("neutral", "amplified")
CLASSES = ("yes", "no", "refusal", "other")
_ER_DIR = re.compile(r"^er\d*$")   # run_arm.py ER buffer dirs: er100, er500, ...

# Refusal class = the pilot's refusal string 'unanswerable' (data_prep.py
# build_vizwiz: majority-vote lowercase answer; the VizWiz prompt instructs
# "respond with 'Unanswerable'") plus the abstention variants the VizWiz answer
# vocabulary and pilot/method/policy_anchor.REFUSAL_PHRASES cover. Matching is
# on norm_answer(target): substring for 'unanswerable', exact for the rest.
REFUSAL_EXACT = frozenset({
    "unsuitable", "unsuitable image", "unknown", "unclear", "not answerable",
    "cannot be answered", "can t be answered", "cant be answered",
    "i don t know", "i dont know", "dont know", "don t know",
})


def classify(target):
    """Answer class of a training target: yes | no | refusal | other."""
    n = norm_answer(target)
    if "unanswerable" in n or n in REFUSAL_EXACT:
        return "refusal"
    if n == "yes":
        return "yes"
    if n == "no":
        return "no"
    return "other"


def answer_stats(targets):
    """Frozen-manifest answer statistics, PILOT definition: fraction of targets
    that are EXACTLY yes / no / unanswerable after strip().lower(), and mean
    target length in words (analysis/diag/criterion_vs_stats.py TRAIN_STATS,
    which reproduces fullstudy/pilot_manifest.json). Kept deliberately
    identical so variant stats are comparable to the original manifest."""
    n = len(targets)
    if n == 0:
        return {"yes_frac": 0.0, "no_frac": 0.0, "refusal_frac": 0.0,
                "mean_target_words": 0.0}
    yes = no = una = 0
    words = 0
    for t in targets:
        a = t.strip().lower()
        yes += a == "yes"
        no += a == "no"
        una += a == "unanswerable"
        words += len(a.split())
    return {"yes_frac": round(yes / n, 4), "no_frac": round(no / n, 4),
            "refusal_frac": round(una / n, 4),
            "mean_target_words": round(words / n, 2)}


# ---------------------------------------------------------------------------
# deterministic selection primitives (random() stream only)
# ---------------------------------------------------------------------------

def _order(rng, items):
    """Seeded random permutation of items via sort-by-random-key."""
    keys = [rng.random() for _ in items]
    return [items[i] for i in sorted(range(len(items)), key=keys.__getitem__)]


def subsample(rng, items, k):
    """k distinct items (all of them if k >= len)."""
    k = max(0, min(k, len(items)))
    return _order(rng, items)[:k]


def oversample(rng, pool, k):
    """k items drawn from pool with the most even multiplicity possible: whole
    shuffled passes over the pool, so no item is repeated before every item has
    been used (max multiplicity ceil(k/len(pool)))."""
    if k <= 0:
        return []
    if not pool:
        raise ValueError("oversample from empty pool")
    out = []
    while len(out) < k:
        out.extend(_order(rng, pool)[:k - len(out)])
    return out


# ---------------------------------------------------------------------------
# per-task selection
# ---------------------------------------------------------------------------

def _counts(idx_by_class):
    return {c: len(idx_by_class[c]) for c in CLASSES}


def _refill(rng, pool, deficit, refill):
    """Rows to append so the file reaches n. Returns (rows, n_dup, shortfall)."""
    if deficit <= 0:
        return [], 0, 0
    if refill == "shrink" or not pool:
        return [], 0, deficit
    rows = oversample(rng, pool, deficit)
    return rows, len(rows), 0


def select_rows(classes, variant, rng, n_target=None, refill="dup",
                neutral_refusal_cap=0.02, amp_refusal_frac=0.60,
                amp_yes_ratio=4.0, amp_yes_frac=None, refusal_task_min=0.10):
    """Decide which source row indices to write (with multiplicity).

    classes: list of class labels, one per source row (index = row position).
    Returns (indices_with_multiplicity, recipe). Kept unique rows appear first
    in source order; refill duplicates are appended (the HF Trainer's seeded
    RandomSampler shuffles within a stage, so file order is not a confound)."""
    n = len(classes)
    n_target = n if n_target is None else n_target
    by = {c: [i for i, cl in enumerate(classes) if cl == c] for c in CLASSES}
    src = _counts(by)
    keep = {c: list(by[c]) for c in CLASSES}     # unique kept rows per class
    extra = []                                    # duplicated rows (indices)
    dup = {c: 0 for c in CLASSES}
    targets = {}
    rule = "unchanged"
    shortfall = 0

    if variant == "neutral":
        if src["yes"] + src["no"] > 0 or src["refusal"] > 0:
            rule = "neutral:balance_yesno+cap_refusal"
            k = min(src["yes"], src["no"])
            cap = int(math.floor(neutral_refusal_cap * n_target))
            targets = {"yesno_each": k, "refusal_cap": cap}
            keep["yes"] = subsample(rng, by["yes"], k)
            keep["no"] = subsample(rng, by["no"], k)
            keep["refusal"] = subsample(rng, by["refusal"], min(src["refusal"], cap))
            deficit = n_target - sum(len(v) for v in keep.values())
            rows, d, shortfall = _refill(rng, by["other"], deficit, refill)
            extra, dup["other"] = rows, d
    elif variant == "amplified":
        if src["refusal"] / max(1, n) >= refusal_task_min:
            rule = "amplified:refusal"
            u_target = int(round(amp_refusal_frac * n_target))
            targets = {"refusal_target": u_target, "refusal_frac": amp_refusal_frac}
            if refill == "shrink" and src["refusal"] < u_target:
                # no duplicates: n shrinks so that all refusal rows are 60%
                n_new = int(math.floor(src["refusal"] / amp_refusal_frac))
                u_target = src["refusal"]
                shortfall = n_target - n_new
                n_target = n_new
                targets["n_shrunk_to"] = n_new
            # non-refusal budget is n - u_target; refusal rows fill the rest,
            # duplicated (dup) when the supply is below u_target.
            u_keep = min(u_target, src["refusal"])
            keep["refusal"] = subsample(rng, by["refusal"], u_keep)
            non_ref = [i for i in range(n) if classes[i] != "refusal"]
            nr_keep = set(subsample(rng, non_ref, n_target - u_target))
            for c in ("yes", "no", "other"):
                keep[c] = [i for i in by[c] if i in nr_keep]
            deficit = n_target - sum(len(v) for v in keep.values())
            rows, d, short2 = _refill(rng, by["refusal"], deficit, refill)
            extra, dup["refusal"] = rows, d
            shortfall += short2
        elif src["yes"] + src["no"] > 0:
            rule = "amplified:yes_ratio"
            if amp_yes_frac is None:
                y_keep = src["yes"]
            else:
                y_keep = int(round(amp_yes_frac * n_target))
            n_keep = min(src["no"], int(math.floor(y_keep / amp_yes_ratio)))
            targets = {"yes_ratio": amp_yes_ratio, "yes_target": y_keep,
                       "no_target": n_keep}
            keep["no"] = subsample(rng, by["no"], n_keep)
            if y_keep > src["yes"]:                      # explicit --amp_yes_frac
                keep["yes"] = list(by["yes"])
                rows, d, short_y = _refill(rng, by["yes"], y_keep - src["yes"], refill)
                extra += rows
                dup["yes"] = d
                shortfall += short_y
                o_room = n_target - len(keep["yes"]) - d - len(keep["no"]) - src["refusal"]
                keep["other"] = subsample(rng, by["other"], max(0, o_room))
            else:
                keep["yes"] = subsample(rng, by["yes"], y_keep)
            deficit = n_target - sum(len(v) for v in keep.values()) - len(extra)
            rows, d, short2 = _refill(rng, by["other"], deficit, refill)
            extra += rows
            dup["other"] += d
            shortfall += short2
    else:
        raise ValueError(f"unknown variant {variant}")

    kept_sorted = sorted(i for v in keep.values() for i in v)
    out = kept_sorted + extra
    written = {c: len(keep[c]) + dup[c] for c in CLASSES}
    recipe = {
        "rule": rule,
        "n_source": n,
        "n_written": len(out),
        "n_unique_written": len(kept_sorted),
        "n_shortfall": shortfall,
        "counts_source": src,
        "counts_kept_unique": {c: len(keep[c]) for c in CLASSES},
        "removed": {c: src[c] - len(keep[c]) for c in CLASSES},
        "duplicated": dup,
        "counts_written": written,
        "targets": targets,
        "unchanged": out == list(range(n)),
    }
    return out, recipe


# ---------------------------------------------------------------------------
# file plumbing
# ---------------------------------------------------------------------------

def read_lines(path):
    """Raw JSONL lines without the trailing newline (blank lines dropped)."""
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_symlink(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def link_assets(data_root, out_root, task, skip_names=("train.jsonl",)):
    """Link every non-train entry of tasks/<t>/ and every top-level source
    entry except tasks/, variant roots, ER buffers and manifests."""
    src_task = os.path.join(data_root, "tasks", task)
    dst_task = os.path.join(out_root, "tasks", task)
    os.makedirs(dst_task, exist_ok=True)
    for name in sorted(os.listdir(src_task)):
        if name in skip_names:
            continue
        relative_symlink(os.path.join(src_task, name), os.path.join(dst_task, name))


def link_top_level(data_root, out_root):
    out_real = os.path.realpath(out_root)
    linked = []
    for name in sorted(os.listdir(data_root)):
        src = os.path.join(data_root, name)
        if name == "tasks" or name.startswith("tasks_") or _ER_DIR.match(name):
            continue
        if name.endswith("manifest.json") or os.path.realpath(src) == out_real:
            continue
        if name.startswith("."):
            continue
        relative_symlink(src, os.path.join(out_root, name))
        linked.append(name)
    return linked


def build(args):
    import random
    man = json.load(open(args.manifest))
    tasks = list(man["tasks"]) if args.tasks == "all" else args.tasks.split(",")
    for t in tasks:
        if t not in man["tasks"]:
            raise SystemExit(f"task {t} not in manifest")
    out_root = args.out_root or os.path.join(args.data_root, f"tasks_{args.variant}")
    mpath = os.path.join(out_root, f"{args.variant}_manifest.json")
    if not args.dry_run:
        if os.path.exists(mpath) and not args.force:
            raise SystemExit(f"{mpath} exists; refusing to overwrite a frozen variant "
                             "without --force (queued jobs read data at START).")
        os.makedirs(os.path.join(out_root, "tasks"), exist_ok=True)

    new_man = json.loads(json.dumps(man))       # deep copy
    new_man["benchmark"] = f"{man.get('benchmark', 'pilot-suite')} / prior-{args.variant}"
    new_man["variant"] = args.variant
    new_man["seed"] = args.seed
    new_man["refill_policy"] = args.refill
    new_man["source_manifest"] = os.path.abspath(args.manifest)
    new_man["source_data_root"] = os.path.abspath(args.data_root)
    new_man["generated_by"] = "fullstudy/neutralize_prior.py"
    new_man["answer_stats_definition"] = (
        "pilot: fraction of targets exactly yes/no/unanswerable after strip().lower(); "
        "mean_target_words = mean whitespace word count. Selection classes use "
        "metrics_task.norm_answer (see recipe.counts_*).")
    new_man["params"] = {
        "neutral_refusal_cap": args.neutral_refusal_cap,
        "amp_refusal_frac": args.amp_refusal_frac,
        "amp_yes_ratio": args.amp_yes_ratio,
        "amp_yes_frac": args.amp_yes_frac,
        "refusal_task_min": args.refusal_task_min,
    }
    new_man["note"] = (f"{man.get('note', '')} | VARIANT {args.variant}: answer prior "
                       "manipulated by row selection only (design_notes/"
                       "mechanism_intervention.md). n_train/answer_stats recomputed "
                       "from the written files.")
    new_man["recipe"] = {}

    print(f"[np] variant={args.variant} seed={args.seed} refill={args.refill} "
          f"out_root={out_root}", flush=True)
    for t in tasks:
        src_path = os.path.join(args.data_root, "tasks", t, "train.jsonl")
        if not os.path.exists(src_path):
            raise SystemExit(f"missing {src_path}")
        lines = read_lines(src_path)
        rows = [json.loads(l) for l in lines]
        classes = [classify(r["target"]) for r in rows]
        n_manifest = man["tasks"][t]["n_train"]
        if len(lines) != n_manifest:
            print(f"[np] WARN {t}: file has {len(lines)} rows, manifest n_train "
                  f"{n_manifest}; using the file count", flush=True)
        rng = random.Random(f"{args.seed}:{args.variant}:{t}")
        sel, recipe = select_rows(
            classes, args.variant, rng, n_target=len(lines), refill=args.refill,
            neutral_refusal_cap=args.neutral_refusal_cap,
            amp_refusal_frac=args.amp_refusal_frac,
            amp_yes_ratio=args.amp_yes_ratio, amp_yes_frac=args.amp_yes_frac,
            refusal_task_min=args.refusal_task_min)
        recipe["n_train_manifest_source"] = n_manifest
        recipe["source_file"] = os.path.abspath(src_path)
        recipe["sha256_source"] = sha256_file(src_path)
        stats_src = answer_stats([r["target"] for r in rows])
        stats_new = answer_stats([rows[i]["target"] for i in sel])
        recipe["answer_stats_source"] = stats_src
        print(f"[np] {t}: {recipe['rule']} n={recipe['n_source']}->{recipe['n_written']} "
              f"(unique {recipe['n_unique_written']}, dup {sum(recipe['duplicated'].values())}, "
              f"shortfall {recipe['n_shortfall']}) src={recipe['counts_source']} "
              f"written={recipe['counts_written']} stats {stats_src} -> {stats_new}",
              flush=True)
        if args.dry_run:
            continue
        dst_dir = os.path.join(out_root, "tasks", t)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "train.jsonl")
        tmp = dst + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for i in sel:
                f.write(lines[i] + "\n")
        os.replace(tmp, dst)
        # read-back audit: count, byte-identical rows, stats from the file itself
        back = read_lines(dst)
        assert len(back) == recipe["n_written"], (t, len(back), recipe["n_written"])
        src_set = set(lines)
        assert all(l in src_set for l in back), f"{t}: a written row is not a source row"
        stats_file = answer_stats([json.loads(l)["target"] for l in back])
        assert stats_file == stats_new, (t, stats_file, stats_new)
        recipe["sha256_written"] = sha256_file(dst)
        if recipe["unchanged"]:
            assert recipe["sha256_written"] == recipe["sha256_source"], t
        if args.link_assets:
            link_assets(args.data_root, out_root, t)
        new_man["tasks"][t]["n_train"] = len(back)
        new_man["tasks"][t]["n_train_source"] = n_manifest
        new_man["tasks"][t]["answer_stats"] = stats_file
        new_man["recipe"][t] = recipe

    if args.dry_run:
        print("[np] dry run: nothing written", flush=True)
        return None
    if args.link_assets:
        new_man["linked_top_level"] = link_top_level(args.data_root, out_root)
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(new_man, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, mpath)                       # written LAST = success marker
    print(f"[np] manifest frozen at {mpath}", flush=True)
    print("[np] ALL OK", flush=True)
    return mpath


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True,
                    help="source data root containing tasks/<t>/train.jsonl")
    ap.add_argument("--manifest", required=True, help="pilot_manifest.json")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--out_root", default=None,
                    help="drop-in data root to write (default <data_root>/tasks_<variant>)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--refill", choices=("dup", "shrink"), default="dup",
                    help="dup: keep n_train by duplicating eligible rows (recorded); "
                         "shrink: never duplicate, record the smaller n_train")
    ap.add_argument("--neutral_refusal_cap", type=float, default=0.02)
    ap.add_argument("--amp_refusal_frac", type=float, default=0.60)
    ap.add_argument("--amp_yes_ratio", type=float, default=4.0)
    ap.add_argument("--amp_yes_frac", type=float, default=None,
                    help="optional: also raise the yes fraction (duplicates yes rows)")
    ap.add_argument("--refusal_task_min", type=float, default=0.10,
                    help="a task with source refusal_frac >= this gets the refusal "
                         "amplification rule; otherwise the yes-ratio rule")
    ap.add_argument("--tasks", default="all", help="comma subset of manifest tasks")
    ap.add_argument("--no_link_assets", dest="link_assets", action="store_false",
                    help="do not symlink images/val/top-level assets into out_root")
    ap.add_argument("--force", action="store_true", help="overwrite an existing variant")
    ap.add_argument("--dry_run", action="store_true", help="print the recipe only")
    args = ap.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    main()
