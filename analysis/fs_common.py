"""Shared helpers for the full-study readout pipeline (aggregate -> figures ->
SOTA table). Ships alongside fs_aggregate.py / make_fs_figures.py /
make_sota_table.py so the three scripts run unchanged from EITHER layout:

  local   <repo>/analysis/<script>.py        pilot scorers in <repo>/pilot/
  cluster ~/cl-halluc/code/analysis/<script>.py  pilot scorers flat in ~/cl-halluc/code/

Contents
  * SDT primitives (clipped_rate, _z) -- verbatim copies of
    analysis/diag/signal_detection.py, which is hard-wired to the pilot's
    results/<ckpt>/ layout and cannot be imported on the cluster.
  * Layout detection + argparse defaults for --results/--bench/--coco_gt/
    --pope_dir/--manifest_ucit/--manifest_pilot (explicit flags always win;
    relative paths resolve against the CURRENT WORKING DIRECTORY, never the
    script dir).
  * Pilot-scorer import shim (parse_yn, CHAIR extractor, norm_answer, SCORERS).
  * RUNTAG / cell-directory parsing for every family:
      fs{L,Q}_<arm>_o<N>_s<seed>     UCIT suite (6 stages)
      ps{L,Q}_<arm>_o<N>_s<seed>     pilot suite (4 stages)
      lamF_<weight>_o<N>_s<seed>     anchor-strength sweep (UCIT, LLaVA)
      mth_<candidate>_o<N>_s<seed>   method candidates (pilot suite)
    cell dirs: <RUNTAG>_k<N> | <RUNTAG>_[ckpt_]step<N> | <RUNTAG>_final (joint)
    (sb_fs_eval strips the 'ckpt_' prefix when it labels joint checkpoints, so both spellings must parse -- 10 JOINT cells were silently 'unparsed' until 2026-09-09)
  * Suite definitions (tasks, orderings, caption tasks) from the frozen
    manifests when present, else the committed inline lists.
  * A JSONL reader that fails loud with file:line context and recovers the one
    known benign corruption (a resumed eval appending a full record directly
    after a partial line from a killed job), reporting it as an audit count.

Python 3.9 + stdlib only.
"""
import json
import os
import random
import re
import sys
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
_z = NormalDist().inv_cdf


class MalformedResult(Exception):
    """A present result file is corrupt/unexpected -- abort loudly, never average."""


# ---------------------------------------------------------------------------
# SDT primitives (verbatim from analysis/diag/signal_detection.py)
# ---------------------------------------------------------------------------
def clipped_rate(k, n):
    """Rate k/n clipped to [1/(2n), 1-1/(2n)]. Returns (rate_for_z, fired)."""
    raw = k / n
    lo = 1.0 / (2.0 * n)
    hi = 1.0 - lo
    if raw < lo:
        return lo, True
    if raw > hi:
        return hi, True
    return raw, False


def sdt_from_counts(hits, nyes, fas, nno):
    """(dprime, c, clipped_flag) from pooled counts; raw H/FA are reported
    separately by the caller (clip only affects the z-transform)."""
    H_z, h_clip = clipped_rate(hits, nyes)
    FA_z, fa_clip = clipped_rate(fas, nno)
    zH, zFA = _z(H_z), _z(FA_z)
    flag = ("H" if h_clip else "") + ("FA" if fa_clip else "") or "none"
    return zH - zFA, -0.5 * (zH + zFA), flag


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------
CLUSTER_ROOT_FALLBACK = Path("/home/alexmueller/cl-halluc")


def _is_cluster_root(p):
    return p is not None and p.is_dir() and (
        (p / "results_fs").is_dir() or (p / "data" / "pope").is_dir())


def _is_repo_root(p):
    return p is not None and (p / "fullstudy").is_dir() and (p / "pilot").is_dir()


def find_layout():
    """Return a dict describing where things live. Order of preference:
    $CLH_ROOT (must look like a cluster root) > ~/cl-halluc > the script's
    grandparent (cluster code/analysis/ sits two below the root) >
    /home/alexmueller/cl-halluc > local repo (script parent dir).
    A layout is only chosen if its marker directories exist."""
    cands = []
    env = os.environ.get("CLH_ROOT")
    if env:
        cands.append(Path(env).expanduser())
    cands.append(Path.home() / "cl-halluc")
    if len(HERE.parents) >= 2:
        cands.append(HERE.parents[1])
    cands.append(CLUSTER_ROOT_FALLBACK)
    for c in cands:
        if _is_cluster_root(c):
            return _cluster_layout(c)
    repo = HERE.parent
    if _is_repo_root(repo):
        return _local_layout(repo)
    # Unknown: still return something usable (all flags must then be explicit).
    return {"name": "unknown", "root": HERE.parent, "results": None, "bench": None,
            "coco_gt": None, "pope_dir": None, "manifest_ucit": None,
            "manifest_pilot": None, "out_dir": Path.cwd() / "readout",
            "agg_default": Path.cwd() / "readout" / "fs_aggregate.json",
            "pilot_code": [HERE.parent, HERE.parent / "pilot"]}


def _cluster_layout(root):
    return {
        "name": "cluster", "root": root,
        "results": root / "results_fs",
        "bench": root / "results_fs_bench",
        "coco_gt": root / "data" / "chair" / "coco_gt.json",
        "pope_dir": root / "data" / "pope",
        "manifest_ucit": root / "data_ucit" / "ucit_manifest.json",
        "manifest_pilot": root / "data" / "pilot_manifest.json",
        "out_dir": root / "readout",
        "agg_default": root / "readout" / "fs_aggregate.json",
        "pilot_code": [root / "code", root / "code" / "pilot"],
    }


def _local_layout(repo):
    return {
        "name": "local", "root": repo,
        "results": repo / "fullstudy" / "results_fs",
        "bench": repo / "fullstudy" / "results_fs_bench",
        "coco_gt": repo / "results" / "coco_gt.json",
        "pope_dir": None,
        "manifest_ucit": repo / "fullstudy" / "ucit_manifest.json",
        "manifest_pilot": repo / "fullstudy" / "pilot_manifest.json",
        "out_dir": repo / "analysis" / "readout",
        "agg_default": repo / "analysis" / "diag" / "fs_aggregate.json",
        "pilot_code": [repo / "pilot"],
    }


LAYOUT_FLAGS = ("results", "bench", "coco_gt", "pope_dir", "manifest_ucit",
                "manifest_pilot")


def add_layout_args(ap, flags=LAYOUT_FLAGS):
    help_ = {
        "results": "results_fs dir (cells <RUNTAG>_k<N>, <backbone>_base)",
        "bench": "results_fs_bench dir (<RUNTAG>_bench/{objhal,mme}.json)",
        "coco_gt": "CHAIR ground truth json (cluster: data/chair/coco_gt.json)",
        "pope_dir": "POPE asset dir holding prompts.jsonl (row-count audit only)",
        "manifest_ucit": "frozen ucit_manifest.json (orders/tasks); inline fallback",
        "manifest_pilot": "pilot_manifest.json (orders/tasks/answer stats)",
    }
    for f in flags:
        ap.add_argument("--" + f, default=None,
                        help=help_[f] + " [default: auto-detected layout]")


def resolve_layout_args(args, flags=LAYOUT_FLAGS):
    """Fill unset flags from the detected layout; make every path absolute
    relative to CWD. Returns the layout dict (for logging)."""
    lay = find_layout()
    for f in flags:
        v = getattr(args, f, None)
        if v is None or v == "":
            setattr(args, f, str(lay[f]) if lay.get(f) is not None else None)
        else:
            setattr(args, f, os.path.abspath(os.path.expanduser(v)))
    return lay


# ---------------------------------------------------------------------------
# Pilot scorer import shim
# ---------------------------------------------------------------------------
_SCORERS = None


def import_pilot_scorers():
    """Import the audited pilot scorers from wherever they live. Returns a dict
    {parse_yn, extract_object_mentions, extract_objects, norm_answer, SCORERS}.
    Tries, in order: modules already importable (PYTHONPATH), <script>/../pilot
    (local), <script>/.. (cluster: code/analysis/.. == code/), and the
    detected layout's pilot_code dirs."""
    global _SCORERS
    if _SCORERS is not None:
        return _SCORERS
    cands = [HERE.parent / "pilot", HERE.parent]
    try:
        cands += list(find_layout().get("pilot_code", []))
    except Exception:  # layout detection must never block scoring
        pass
    for c in cands:
        if (c / "metrics_pope.py").is_file() and str(c) not in sys.path:
            sys.path.append(str(c))
    try:
        from metrics_pope import parse_yn
        from metrics_chair import extract_object_mentions, extract_objects
        from metrics_task import norm_answer, SCORERS
    except ImportError as e:
        raise SystemExit(
            "[fs_common] cannot import the pilot scorers (metrics_pope/"
            "metrics_chair/metrics_task): %s\n  searched: %s\n  set PYTHONPATH "
            "to the directory holding metrics_pope.py" % (e, [str(c) for c in cands]))
    _SCORERS = {"parse_yn": parse_yn,
                "extract_object_mentions": extract_object_mentions,
                "extract_objects": extract_objects,
                "norm_answer": norm_answer, "SCORERS": SCORERS}
    return _SCORERS


# ---------------------------------------------------------------------------
# RUNTAG families and cell names
# ---------------------------------------------------------------------------
FAMILY_RE = re.compile(
    r"^(?P<fam>fs[LQ]|ps[LQ]|lamF|mth)_(?P<arm>.+?)_(?P<order>o\d+)_s(?P<seed>\d+)$")
CELL_RE = re.compile(
    r"^(?P<runtag>.+?)_(?:k(?P<k>\d+)|(?:ckpt_)?step(?P<step>\d+)|(?P<final>final))$")
BACKBONE_OF_LETTER = {"L": "llava15", "Q": "qwen25vl"}
BASE_DIRS = {"llava15": "llava15_base", "qwen25vl": "qwen25vl_base"}
FAMILY_SUITE = {"fs": "ucit", "ps": "pilot", "lamF": "ucit", "mth": "pilot"}
# lamF/mth carry no backbone letter; the queue runs them on LLaVA (the pilot
# backbone). Override with --family_backbone lamF=qwen25vl,mth=... if that changes.
FAMILY_BACKBONE_DEFAULT = {"lamF": "llava15", "mth": "llava15"}
BACKBONE_LABEL = {"llava15": "LLaVA-1.5-7B", "qwen25vl": "Qwen2.5-VL-7B"}


def parse_family_backbone(spec):
    out = dict(FAMILY_BACKBONE_DEFAULT)
    if spec:
        for kv in spec.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split("=")
            if v not in BASE_DIRS:
                raise SystemExit("--family_backbone: unknown backbone %r" % v)
            out[k.strip()] = v.strip()
    return out


def parse_runtag(runtag, family_backbone=None):
    """-> dict(runtag, family, arm, order, seed, backbone, suite, faith_weight)
    or None when the name is not one of ours."""
    m = FAMILY_RE.match(runtag)
    if not m:
        return None
    fam, arm, order, seed = m.group("fam"), m.group("arm"), m.group("order"), int(m.group("seed"))
    fb = family_backbone or FAMILY_BACKBONE_DEFAULT
    faith_weight = None
    if fam in ("fsL", "fsQ", "psL", "psQ"):
        backbone = BACKBONE_OF_LETTER[fam[-1]]
        family = fam[:2]
    elif fam == "lamF":
        try:
            faith_weight = float(arm)
        except ValueError:
            return None
        backbone = fb.get("lamF", "llava15")
        family = "lamF"
        arm = "lamF_" + arm          # arm token keeps the weight, e.g. lamF_0.02
    else:  # mth
        backbone = fb.get("mth", "llava15")
        family = "mth"
        arm = "mth_" + arm
    return {"runtag": runtag, "family": family, "arm": arm, "order": order,
            "seed": seed, "backbone": backbone, "suite": FAMILY_SUITE[family],
            "faith_weight": faith_weight}


def parse_cell_name(name):
    """-> (runtag, kind, value): kind in {'k','step','final'}; value int|None."""
    m = CELL_RE.match(name)
    if not m:
        return None
    if m.group("k") is not None:
        return m.group("runtag"), "k", int(m.group("k"))
    if m.group("step") is not None:
        return m.group("runtag"), "step", int(m.group("step"))
    return m.group("runtag"), "final", None


def arm_family(arm):
    """Scorecard family of an arm token: lamF_0.02 -> lamF, single_X -> single,
    mth_x -> mth, everything else itself."""
    for pfx in ("lamF_", "mth_", "single_"):
        if arm.startswith(pfx):
            return pfx[:-1]
    return arm


# Canonical scorecard row order; unknown arms are appended after these.
ARM_ORDER = ["seq", "anchor", "critp", "anchorcrit", "cecf", "er", "er500",
             "ewc", "lwf", "joint"]


def arm_sort_key(arm):
    if arm in ARM_ORDER:
        return (0, ARM_ORDER.index(arm), 0.0, arm)
    if arm.startswith("lamF_"):
        try:
            return (1, 0, float(arm[5:]), arm)
        except ValueError:
            return (1, 1, 0.0, arm)
    if arm.startswith("mth_"):
        return (2, 0, 0.0, arm)
    if arm.startswith("single_"):
        return (4, 0, 0.0, arm)
    return (3, 0, 0.0, arm)


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------
# UCIT canonical order (fullstudy/ucit_prep.py ORDER_CANONICAL; o3 = the
# random.Random(41).shuffle permutation, exactly as ucit_prep freezes it).
UCIT_ORDER_O1 = ["ArxivQA", "CLEVR-Math", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"]
UCIT_CAPTION_TASKS = {"Flickr30k", "VizWiz"}      # open-ended: containment-EM is a floor
PILOT_ORDERS = {"o1": ["scienceqa", "textvqa", "flickr", "vizwiz"],
                "o2": ["vizwiz", "flickr", "textvqa", "scienceqa"]}
PILOT_CAPTION_TASKS = {"flickr"}                   # caption_uf1 is a trajectory signal
PILOT_TASKS = set(PILOT_ORDERS["o1"])


def _ucit_inline_orders():
    o3 = list(UCIT_ORDER_O1)
    random.Random(41).shuffle(o3)
    return {"o1": list(UCIT_ORDER_O1), "o2": list(reversed(UCIT_ORDER_O1)), "o3": o3}


def load_suite(suite, manifest_path):
    """Suite descriptor: {name, tasks:{t:{...}}, orders:{o:[t..]}, n_stages,
    caption_tasks, source}. Reads the frozen manifest when it exists, else the
    committed inline lists (source says which)."""
    man = None
    if manifest_path and os.path.isfile(str(manifest_path)):
        with open(manifest_path) as f:
            man = json.load(f)
    if suite == "ucit":
        if man:
            orders = dict(man["orders"])
            tasks = dict(man["tasks"])
            src = str(manifest_path)
        else:
            orders = _ucit_inline_orders()
            tasks = {t: {} for t in UCIT_ORDER_O1}
            src = "inline (ucit_prep.py ORDER_CANONICAL; o3 = Random(41).shuffle)"
        caption = set(UCIT_CAPTION_TASKS)
    elif suite == "pilot":
        if man:
            orders = dict(man["orders"])
            tasks = dict(man["tasks"])
            src = str(manifest_path)
        else:
            orders = {k: list(v) for k, v in PILOT_ORDERS.items()}
            tasks = {t: {} for t in PILOT_ORDERS["o1"]}
            src = "inline (pilot_manifest.json orders)"
        caption = set(PILOT_CAPTION_TASKS)
    else:
        raise ValueError("unknown suite %r" % suite)
    n_stages = len(next(iter(orders.values())))
    return {"name": suite, "tasks": tasks, "orders": orders, "n_stages": n_stages,
            "caption_tasks": sorted(caption), "source": src}


def order_tasks(suite_info, order_tag):
    return list(suite_info["orders"].get(order_tag) or []) or None


def suite_of_task_files(task_names):
    """Data-driven suite guess from the task gen files present in a cell."""
    s = set(task_names)
    if s & PILOT_TASKS:
        return "pilot"
    if s & set(UCIT_ORDER_O1):
        return "ucit"
    return None


# ---------------------------------------------------------------------------
# JSONL reader (fail loud; recover the one known benign corruption)
# ---------------------------------------------------------------------------
_REC_START = '{"id"'


def iter_jsonl(path, audit=None):
    """Yield (lineno, obj). A line that is not valid JSON is retried from its
    LAST '{"id"' occurrence: eval_gen.py appends complete records after a
    partial line left by a killed job, so the tail of such a line is a full
    record and the head is a duplicate of an id regenerated later. Recovered
    lines are counted in audit['recovered_lines']; anything else raises
    MalformedResult with file:line context."""
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
                continue
            except json.JSONDecodeError as e:
                err = e
            j = line.rfind(_REC_START)
            if j > 0:
                try:
                    obj = json.loads(line[j:])
                    if audit is not None:
                        audit["recovered_lines"] = audit.get("recovered_lines", 0) + 1
                    yield i, obj
                    continue
                except json.JSONDecodeError:
                    pass
            raise MalformedResult("%s:%d: invalid JSON (%s)" % (path, i, err))


def read_rows_dedup(path, audit=None):
    """All rows of a gen file, deduplicated on 'id' (last occurrence wins;
    duplicates counted in audit['dup_ids']). Rows without 'id' are kept as-is."""
    rows = {}
    order = []
    extra = []
    for _ln, r in iter_jsonl(path, audit):
        rid = r.get("id")
        if rid is None:
            extra.append(r)
            continue
        if rid in rows:
            if audit is not None:
                audit["dup_ids"] = audit.get("dup_ids", 0) + 1
        else:
            order.append(rid)
        rows[rid] = r
    return [rows[i] for i in order] + extra


def count_jsonl_ids(path):
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def fmt(v, spec="+.3f", absent="ABSENT"):
    return absent if v is None else format(v, spec)


def rng_pair(vals):
    return [round(min(vals), 4), round(max(vals), 4)] if vals else None


def mean(vals):
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Per-gen-file scorers. Counts are assembled HERE; every numerically
# load-bearing primitive (parse_yn, the CHAIR extractor, norm_answer, the pilot
# task scorers, the SDT clip + z) is the audited one imported above. These are
# the single implementations the aggregate, the figure fallback and the SOTA
# table all call, so no number can be computed two different ways.
# ---------------------------------------------------------------------------
POPE_FILE = "pope_gen.jsonl"
CHAIR_FILE = "chair_gen.jsonl"
NON_TASK_GEN = {"pope_gen.jsonl", "chair_gen.jsonl", "amber_gen.jsonl"}
CHAIR_K_WORDS = 60   # pre-registered length-control budget (FULLSTUDY_PREREG E3)
DOSE_ACTIVE = 0.02   # a stage is criterion-active when |yes-no| + refusal >= this


def task_gen_files(cell_dir):
    """Task names of the <task>_gen.jsonl files in a cell (POPE/CHAIR excluded)."""
    return sorted(p.name[:-len("_gen.jsonl")] for p in Path(cell_dir).glob("*_gen.jsonl")
                  if p.name not in NON_TASK_GEN)


def _tok_trunc(r):
    return (r.get("n_new_tokens") or 0), int(bool(r.get("truncated", False)))


def score_pope(path, audit=None):
    """Pooled-POPE signal detection over parsed rows -> audit dict. Yes is the
    positive class; parse failures are counted, never coerced. f1/precision/
    recall are the exact 'all'-category values of pilot/metrics_pope.py."""
    parse_yn = import_pilot_scorers()["parse_yn"]
    hits = fas = nyes = nno = say = parsed = nfail = ntot = 0
    tp = fp = fn = 0
    tok = trunc = 0
    for r in read_rows_dedup(str(path), audit):
        if "gt" not in r or "output" not in r:
            raise MalformedResult("%s: POPE row missing 'gt'/'output' (id=%r)"
                                  % (path, r.get("id")))
        gt = r["gt"]
        if gt not in ("yes", "no"):
            raise MalformedResult("%s: unexpected POPE gt=%r (id=%r)" % (path, gt, r.get("id")))
        ntot += 1
        t, tr = _tok_trunc(r)
        tok += t
        trunc += tr
        pred = parse_yn(r["output"])
        if pred is None:
            nfail += 1
            continue
        parsed += 1
        if pred == "yes":
            say += 1
        if gt == "yes":
            nyes += 1
            if pred == "yes":
                hits += 1
                tp += 1
            else:
                fn += 1
        else:
            nno += 1
            if pred == "yes":
                fas += 1
                fp += 1
    if ntot == 0:
        raise MalformedResult("%s: empty POPE gen file" % path)
    if nyes == 0 or nno == 0:
        raise MalformedResult("%s: degenerate POPE (nyes=%d, nno=%d)" % (path, nyes, nno))
    d, c, flag = sdt_from_counts(hits, nyes, fas, nno)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "H": round(hits / nyes, 4), "FA": round(fas / nno, 4),
        "dprime": round(d, 4), "c": round(c, 4),
        "yes_rate": round(say / parsed, 4),
        "n_total": ntot, "n_parsed": parsed, "parse_fail": nfail,
        "n_gt_yes": nyes, "n_gt_no": nno, "clipped": flag,
        "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        "mean_new_tokens": round(tok / ntot, 1), "truncation_rate": round(trunc / ntot, 4),
    }


_COCO_GT_CACHE = {}


def load_coco_gt(path):
    """{str(coco_id): {objects, captions}} (cached per path); None if absent."""
    if not path or not os.path.isfile(str(path)):
        return None
    key = os.path.abspath(str(path))
    if key not in _COCO_GT_CACHE:
        with open(key) as f:
            _COCO_GT_CACHE[key] = json.load(f)
    return _COCO_GT_CACHE[key]


def score_chair(path, coco_gt, audit=None, k_words=CHAIR_K_WORDS):
    """Per-mention CHAIR_i over full captions AND over the first k_words
    whitespace words (the E3 length-controlled quantity, identical truncation to
    analysis/diag/length_controlled_chair.py::chair_over). GT per image =
    instance objects UNION objects the same extractor finds in the GT captions.
    The private '_per_image_k' map {coco_id: (n_mentions@k, n_hal@k)} feeds the
    paired bootstrap and is stripped before serialization. Fails loud on a
    coco_id absent from the GT table."""
    sc = import_pilot_scorers()
    ext_m, ext_o = sc["extract_object_mentions"], sc["extract_objects"]
    n_caps = n_ment = n_hal = n_hal_caps = 0
    n_ment_k = n_hal_k = n_over = 0
    tok = trunc = words = 0
    per_k = {}
    gt_cache = {}
    for r in read_rows_dedup(str(path), audit):
        if "coco_id" not in r or "output" not in r:
            raise MalformedResult("%s: CHAIR row missing 'coco_id'/'output' (id=%r)"
                                  % (path, r.get("id")))
        cid = str(r["coco_id"])
        if cid not in coco_gt:
            raise MalformedResult("%s: coco_id %s absent from the CHAIR GT table" % (path, cid))
        gto = gt_cache.get(cid)
        if gto is None:
            gto = set(coco_gt[cid]["objects"])
            for cap in coco_gt[cid].get("captions", []):
                gto |= ext_o(cap)
            gt_cache[cid] = gto
        out = r["output"]
        m = ext_m(out)
        h = sum(1 for x in m if x not in gto)
        n_caps += 1
        n_ment += len(m)
        n_hal += h
        n_hal_caps += int(h > 0)
        ws = out.split()
        words += len(ws)
        if len(ws) > k_words:
            n_over += 1
            mk = ext_m(" ".join(ws[:k_words]))
            hk = sum(1 for x in mk if x not in gto)
        else:
            mk, hk = m, h
        n_ment_k += len(mk)
        n_hal_k += hk
        per_k[cid] = (len(mk), hk)
        t, tr = _tok_trunc(r)
        tok += t
        trunc += tr
    if n_caps == 0:
        raise MalformedResult("%s: empty CHAIR gen file" % path)
    return {
        "chair_i": round(n_hal / max(1, n_ment), 4),
        "n_captions": n_caps, "n_mentions": n_ment, "n_hal_mentions": n_hal,
        "mentions_per_caption": round(n_ment / max(1, n_caps), 3),
        "chair_s": round(n_hal_caps / n_caps, 4),
        "chair_i60": round(n_hal_k / max(1, n_ment_k), 4), "k_words": k_words,
        "n_mentions60": n_ment_k, "n_hal_mentions60": n_hal_k,
        "frac_over_budget": round(n_over / n_caps, 4),
        "mean_words": round(words / n_caps, 1),
        "mean_new_tokens": round(tok / n_caps, 1), "truncation_rate": round(trunc / n_caps, 4),
        "_per_image_k": per_k,
    }


def score_task(path, task, suite, audit=None):
    """Per-task held-out accuracy.
    pilot suite: the audited pilot/metrics_task.py scorer for that task
      (scienceqa mc_acc, textvqa/vizwiz vqa_acc, flickr caption_uf1) -- the
      same numbers analysis/method_comparison.py reports for the pilot arms.
    ucit suite: normalized containment-EM against meta.answers (a floor on the
      open-ended caption tasks Flickr30k/VizWiz, flagged open_ended).
    refusal_rate = fraction of outputs whose normalized form contains
    'unanswerable' (the leakage audit on every short-answer task)."""
    sc = import_pilot_scorers()
    norm, scorers = sc["norm_answer"], sc["SCORERS"]
    rows = read_rows_dedup(str(path), audit)
    if not rows:
        raise MalformedResult("%s: empty task gen file" % path)
    n = len(rows)
    empty = refusal = tok = trunc = 0
    for r in rows:
        if "output" not in r:
            raise MalformedResult("%s: task row missing 'output' (id=%r)" % (path, r.get("id")))
        p = norm(r["output"])
        empty += int(not p)
        refusal += int("unanswerable" in p)
        t, tr = _tok_trunc(r)
        tok += t
        trunc += tr
    if suite == "pilot" and task in scorers:
        try:
            res = scorers[task](rows)
        except KeyError as e:
            raise MalformedResult("%s: pilot scorer for %s needs meta field %s" % (path, task, e))
        out = {"acc": res["score"], "n": res["n"], "empty_pred": empty,
               "metric": res["metric"], "open_ended": task in PILOT_CAPTION_TASKS,
               "parse_fail_rate": res.get("parse_fail_rate", 0.0)}
    else:
        ok = 0
        for r in rows:
            meta = r.get("meta") or {}
            if "answers" not in meta:
                raise MalformedResult("%s: task row missing meta.answers (id=%r)"
                                      % (path, r.get("id")))
            pred = norm(r["output"])
            anss = [norm(a) for a in meta["answers"]]
            if any(a and (pred == a or a in pred) for a in anss):
                ok += 1
        out = {"acc": round(ok / n, 4), "n": n, "empty_pred": empty,
               "metric": "norm-containment-EM",
               "open_ended": task in UCIT_CAPTION_TASKS, "parse_fail_rate": 0.0}
    out.update({"refusal_rate": round(refusal / n, 4),
                "mean_new_tokens": round(tok / n, 1),
                "truncation_rate": round(trunc / n, 4)})
    return out


def strip_private(cell):
    """Copy of a scored cell without the in-memory-only '_' keys (bootstrap inputs)."""
    out = {}
    for k, v in cell.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        if isinstance(v, dict):
            out[k] = strip_private(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Paired eval-set bootstrap (ratio of sums over shared image ids)
# ---------------------------------------------------------------------------
def boot_ratio_gap(per_a, per_b, B=10000, seed=17):
    """Paired bootstrap of the CHAIR_i gap (b - a) over the image ids both cells
    share: resample images with replacement, ratio-of-sums CHAIR_i per arm per
    resample, percentile CI with the pilot's index convention
    (pilot/compare_endpoints.py: gaps[int(0.025B)], gaps[min(B-1, int(0.975B))]).
    Uses numpy when importable (fast), else a stdlib loop. Divergence regime,
    named: the two paths draw DIFFERENT resample streams for the same seed, so
    CI endpoints agree only to bootstrap tolerance; 'impl' records which ran."""
    ids = sorted(set(per_a) & set(per_b))
    n = len(ids)
    if n == 0:
        return None
    ma = [per_a[i][0] for i in ids]
    ha = [per_a[i][1] for i in ids]
    mb = [per_b[i][0] for i in ids]
    hb = [per_b[i][1] for i in ids]
    point = sum(hb) / max(1, sum(mb)) - sum(ha) / max(1, sum(ma))
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(B, n))
        MA = np.asarray(ma, dtype=float)[idx].sum(1)
        HA = np.asarray(ha, dtype=float)[idx].sum(1)
        MB = np.asarray(mb, dtype=float)[idx].sum(1)
        HB = np.asarray(hb, dtype=float)[idx].sum(1)
        gaps = np.sort(HB / np.maximum(MB, 1.0) - HA / np.maximum(MA, 1.0)).tolist()
        impl = "numpy"
    except ImportError:
        rng = random.Random(seed)
        gaps = []
        rr = range(n)
        for _ in range(B):
            s = rng.choices(rr, k=n)
            sma = sum(ma[i] for i in s)
            sha = sum(ha[i] for i in s)
            smb = sum(mb[i] for i in s)
            shb = sum(hb[i] for i in s)
            gaps.append(shb / max(1, smb) - sha / max(1, sma))
        gaps.sort()
        impl = "stdlib"
    lo = gaps[int(0.025 * B)]
    hi = gaps[min(B - 1, int(0.975 * B))]
    return {"gap": round(point, 5), "ci": [round(lo, 5), round(hi, 5)], "B": B,
            "n_images": n, "seed": seed, "impl": impl,
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------
# Cell provenance, stage helpers, dose table, external-benchmark loader
# ---------------------------------------------------------------------------
def backbone_from_gate_info(cell_dir):
    """Backbone recorded by eval_gen.py in the cell's gate_info.json argv
    ('--backbone <name>'); None if absent. The data-driven source for families
    whose RUNTAG carries no backbone letter (lamF_*, mth_*)."""
    p = Path(cell_dir) / "gate_info.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            j = json.load(f)
    except ValueError:
        return None
    argv = j.get("argv") or []
    for i in range(len(argv) - 1):
        if argv[i] == "--backbone" and argv[i + 1] in BASE_DIRS:
            return argv[i + 1]
    return None


def contiguous_prefix(stages):
    """Stage indices 1..m that are all present (endpoint math needs consecutive
    stages; a gap after m makes later swings span two stages)."""
    ks = []
    k = 1
    while k in stages:
        ks.append(k)
        k += 1
    return ks


def swings(vals):
    return [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]


def post_settle_sum(sw):
    """Sigma|dc| with the first (settling) step dropped; None if < 2 steps."""
    return sum(sw[1:]) if len(sw) >= 2 else None


def dose_of(stats):
    """Stage dose from frozen answer statistics: magnitude = |yes-no| + refusal
    (the pilot's dose variable); direction = sign(yes - no - refusal): +1 pushes
    the criterion liberal (yes), -1 conservative (no/abstain); 0 = zero-dose.
    None when the manifest carries no stats (inline fallback)."""
    if not stats:
        return None
    y = float(stats.get("yes_frac", 0.0) or 0.0)
    n = float(stats.get("no_frac", 0.0) or 0.0)
    r = float(stats.get("refusal_frac", 0.0) or 0.0)
    mag = abs(y - n) + r
    active = mag >= DOSE_ACTIVE
    direction = 0 if not active else (1 if (y - n - r) > 0 else -1)
    return {"direction": direction, "magnitude": round(mag, 4), "active": active,
            "yes_frac": y, "no_frac": n, "refusal_frac": r}


BENCH_DONE = "BENCH_DONE"
_BENCH_CHAIR_KEYS = ("chair_i", "chair_s", "n_captions", "mentions_per_caption",
                     "mean_new_tokens", "truncation_rate")


def bench_label(runtag):
    """results_fs_bench dir name for a runtag (sb_fs_bench.sbatch LABEL)."""
    return runtag if runtag in BASE_DIRS.values() else runtag + "_bench"


def load_bench(bench_dir, label):
    """Endpoint external-benchmark summaries for one bench dir, or None if the
    dir is absent. objhal/noncoco = pilot/metrics_chair.py summaries; mme = the
    'subtask == all' row of pilot/metrics_mme.py plus per-subtask scores. A
    present-but-unreadable file raises MalformedResult."""
    d = Path(bench_dir) / label
    if not d.is_dir():
        return None
    out = {"dir": label, "done": (d / BENCH_DONE).exists(),
           "objhal": None, "noncoco": None, "mme": None, "amberdisc": None}
    for name in ("objhal", "noncoco"):
        p = d / (name + ".json")
        if p.is_file():
            try:
                with open(p) as f:
                    j = json.load(f)
            except ValueError as e:
                raise MalformedResult("%s: invalid JSON (%s)" % (p, e))
            if not isinstance(j, dict) or "chair_i" not in j:
                raise MalformedResult("%s: not a metrics_chair summary" % p)
            out[name] = {k: j.get(k) for k in _BENCH_CHAIR_KEYS}
    p = d / "mme.json"
    if p.is_file():
        try:
            with open(p) as f:
                rows = json.load(f)
        except ValueError as e:
            raise MalformedResult("%s: invalid JSON (%s)" % (p, e))
        if not isinstance(rows, list):
            raise MalformedResult("%s: expected a list of subtask rows" % p)
        allrow = next((r for r in rows if r.get("subtask") == "all"), None)
        if allrow is None:
            raise MalformedResult("%s: no 'all' summary row" % p)
        out["mme"] = {
            "score": allrow.get("score"), "acc": allrow.get("acc"),
            "full_800_scale": allrow.get("full_800_scale"),
            "subtasks_present": allrow.get("subtasks_present"),
            "other_rate": allrow.get("other_rate"), "n_questions": allrow.get("n_questions"),
            "per_subtask": {r["subtask"]: r.get("score") for r in rows
                            if r.get("subtask") != "all"},
        }
    p = d / "amberdisc.json"
    if p.is_file():
        out["amberdisc"] = {"present": True}
    return out


def discover_bench(bench_dir):
    """{label: load_bench(...)} for every dir under results_fs_bench (or {})."""
    out = {}
    bd = Path(bench_dir) if bench_dir else None
    if bd is None or not bd.is_dir():
        return out
    for entry in sorted(bd.iterdir()):
        if entry.is_dir():
            out[entry.name] = load_bench(bd, entry.name)
    return out
