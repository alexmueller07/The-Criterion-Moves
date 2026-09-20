"""Shared loaders for pilot results directories.

Schemas mirror pilot/metrics_pope.py, pilot/metrics_chair.py, pilot/metrics_task.py
(see also pilot/jobs/sb_eval.sbatch for file naming):

  results/<CKPT>/pope.json            list of rows, one per POPE category + "all":
                                      {ckpt, category, n, accuracy, precision, recall,
                                       f1, yes_rate, parse_fail_rate}
  results/<CKPT>/pope_gen.jsonl       raw generations; rows carry {category, output,
                                      n_new_tokens, truncated, ...} -- used here only
                                      to compute POPE truncation rates (audit column).
  results/<CKPT>/chair.json           {ckpt, n_captions, chair_s, chair_i,
                                       mentions_per_caption, mean_new_tokens,
                                       truncation_rate}
  results/<CKPT>/chair_per_image.jsonl  per-image rows (bootstrap input; not needed here)
  results/<CKPT>/task_<t>.json        {metric, score, n, parse_fail_rate, ckpt, task,
                                       mean_new_tokens, truncation_rate}

Missing files/checkpoints are reported on stderr and returned as None -- never
fabricated, never silently coerced.
"""
import json
import os
import sys

CKPTS = ["S0", "S1", "S2", "S3", "S4", "J1", "J2", "J3", "J4"]
SEQ_CKPTS = ["S0", "S1", "S2", "S3", "S4"]      # SEQ trajectory (S0 = shared base)
JOINT_CKPTS = ["S0", "J1", "J2", "J3", "J4"]    # JOINT trajectory (S0 = shared base)
STAGE_LABELS = ["S0", "S1/J1", "S2/J2", "S3/J3", "S4/J4"]

TASKS = ["scienceqa", "textvqa", "flickr", "vizwiz"]  # pilot training order
TASK_METRIC = {"scienceqa": "mc_acc", "textvqa": "vqa_acc",
               "flickr": "caption_uf1", "vizwiz": "vqa_acc"}
POPE_SPLITS = ["random", "popular", "adversarial"]


def warn(msg):
    print("[results_io] WARNING: %s" % msg, file=sys.stderr)


def _load_json(path):
    """Load a JSON file; missing or unreadable -> None (with a warning)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        warn("unreadable %s (%r) -- treating as missing" % (path, e))
        return None


def _pope_truncation(path):
    """Truncation rate per POPE category (+ 'all') from pope_gen.jsonl.

    metrics_pope.py does not compute this, but the raw generations carry the
    audit fields, so we derive it here rather than leaving the audit column
    empty. Returns {category: rate} or None if the file is absent/unreadable.
    """
    if not os.path.exists(path):
        return None
    n = {}
    trunc = {}
    bad = 0
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    cat = r.get("category", "unknown")
                    t = int(bool(r["truncated"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    bad += 1
                    continue
                for c in (cat, "all"):
                    n[c] = n.get(c, 0) + 1
                    trunc[c] = trunc.get(c, 0) + t
    except OSError as e:
        warn("unreadable %s (%r)" % (path, e))
        return None
    if bad:
        warn("%s: %d malformed lines skipped in truncation audit" % (path, bad))
    if not n:
        return None
    return {c: round(trunc[c] / float(n[c]), 4) for c in n}


def load_checkpoint(results_dir, ckpt):
    """Load one checkpoint dir. Returns None if the dir itself is absent."""
    d = os.path.join(results_dir, ckpt)
    if not os.path.isdir(d):
        return None
    out = {"ckpt": ckpt, "chair": _load_json(os.path.join(d, "chair.json"))}
    pope_rows = _load_json(os.path.join(d, "pope.json"))
    if pope_rows is not None:
        try:
            out["pope"] = {r["category"]: r for r in pope_rows}
        except (TypeError, KeyError) as e:
            warn("%s/pope.json has unexpected shape (%r) -- ignored" % (d, e))
            out["pope"] = None
    else:
        out["pope"] = None
    out["tasks"] = {t: _load_json(os.path.join(d, "task_%s.json" % t)) for t in TASKS}
    out["pope_trunc"] = _pope_truncation(os.path.join(d, "pope_gen.jsonl"))
    for name, key in (("chair.json", "chair"), ("pope.json", "pope")):
        if out[key] is None:
            warn("%s/%s missing -- will appear as gaps, not values" % (d, name))
    return out


def load_results(results_dir):
    """Load all canonical checkpoints. {ckpt: row-or-None}; dies if none exist."""
    if not os.path.isdir(results_dir):
        raise SystemExit("[results_io] results dir does not exist: %s" % results_dir)
    table = {}
    n_found = 0
    for c in CKPTS:
        row = load_checkpoint(results_dir, c)
        if row is None:
            warn("checkpoint %s absent under %s -- skipped" % (c, results_dir))
        else:
            n_found += 1
        table[c] = row
    extra = sorted(x for x in os.listdir(results_dir)
                   if os.path.isdir(os.path.join(results_dir, x)) and x not in CKPTS)
    if extra:
        warn("non-canonical subdirs ignored: %s" % ", ".join(extra))
    if n_found == 0:
        raise SystemExit("[results_io] no checkpoints (%s) found under %s"
                         % ("/".join(CKPTS), results_dir))
    print("[results_io] loaded %d/%d checkpoints from %s"
          % (n_found, len(CKPTS), results_dir))
    return table


# ---- safe accessors: every miss is None, never a KeyError, never a default value


def chair_metric(table, ckpt, key):
    row = table.get(ckpt)
    if not row or not row["chair"]:
        return None
    return row["chair"].get(key)


def pope_metric(table, ckpt, category, key):
    row = table.get(ckpt)
    if not row or not row["pope"] or category not in row["pope"]:
        return None
    return row["pope"][category].get(key)


def pope_truncation(table, ckpt, category):
    row = table.get(ckpt)
    if not row or not row["pope_trunc"]:
        return None
    return row["pope_trunc"].get(category)


def task_field(table, ckpt, task, key):
    row = table.get(ckpt)
    if not row or not row["tasks"].get(task):
        return None
    return row["tasks"][task].get(key)


def task_score(table, ckpt, task):
    return task_field(table, ckpt, task, "score")
