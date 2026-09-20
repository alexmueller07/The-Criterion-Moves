"""CPU tests for eval_gen.run_dump (the --dump_logits instrument) with a mocked
model / processor / backbone. No GPU, no download, no real forward.

What is pinned:
  * position gather under RIGHT padding: the recorded z_yes/z_no come from the
    last REAL token (attention_mask.sum-1), never from the padded tail;
  * gap == z_yes - z_no and its sign agrees with the argmax decision whenever
    ' Yes'/' No' are the top-2 tokens; a third token winning is reported as
    argmax_is == "other" (the logit-level analogue of a parse failure);
  * OOM-halving: a model that OOMs above batch 2 still yields every row, in
    order, with values identical to a batch-1 run;
  * floats are cast to the model dtype, every processor output is forwarded;
  * the right-padding precondition and the logit/input length assert fire;
  * method.faith_loss.resolve_answer_token_ids resolves in context and rejects
    an unstable tokenizer.

Torch/PIL guarded (skips cleanly on a light interpreter, test_baselines
convention). Python 3.9; no pytest needed:

    python3 pilot/tests/test_logit_dump.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from PIL import Image
    from eval_gen import run_dump
    from method.faith_loss import resolve_answer_token_ids
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _skip(name):
    print(f"SKIP {name}: torch/PIL not installed")


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------
PAD, EOS = 0, 1
DISTRACTOR = 2   # a vocab row that wins argmax only at pad positions / 'oth' rows


class FakeTokenizer:
    """Word-level tokenizer with a growing vocab; ids 0..2 reserved."""

    def __init__(self):
        self.vocab = {"<pad>": PAD, "</s>": EOS, "<distractor>": DISTRACTOR}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = PAD
        self.eos_token_id = EOS
        self.padding_side = "right"

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab)
            self.inv[self.vocab[w]] = w
        return self.vocab[w]

    def encode_words(self, text):
        return [self._id(w) for w in text.split()]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode_words(text)}

    def decode(self, ids):
        return " ".join(self.inv[int(i)] for i in ids)


class UnstableTokenizer(FakeTokenizer):
    """tok(stub + ' Yes') does NOT extend tok(stub): the last stub word merges
    with the answer (the failure mode the in-context assert exists for)."""

    def __call__(self, text, add_special_tokens=False):
        words = text.split()
        if len(words) >= 2 and words[-1] in ("Yes", "No"):
            words = words[:-2] + [words[-2] + words[-1]]
        return {"input_ids": [self._id(w) for w in words]}


class FakeProcessor:
    def __init__(self, tok):
        self.tokenizer = tok

    def __call__(self, text, images, return_tensors="pt", padding=True):
        ids = [self.tokenizer.encode_words(t) for t in text]
        T = max(len(x) for x in ids)
        B = len(ids)
        input_ids = torch.full((B, T), PAD, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        for b, x in enumerate(ids):
            if self.tokenizer.padding_side == "right":
                input_ids[b, :len(x)] = torch.tensor(x)
                attn[b, :len(x)] = 1
            else:
                input_ids[b, T - len(x):] = torch.tensor(x)
                attn[b, T - len(x):] = 1
        pix = torch.stack([torch.tensor(list(im.getdata())[:4], dtype=torch.float32)
                           for im in images])  # (B, 4, 3) float32
        return {"input_ids": input_ids, "attention_mask": attn,
                "pixel_values": pix, "extra_flag": torch.ones(B, dtype=torch.long)}


class FakeBackbone:
    def build_prompt(self, p):
        return f"USER: <image> {p} ASSISTANT:"

    def anchor_stub(self):
        return "USER: hi ASSISTANT:"


class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeModel(torch.nn.Module):
    """logits[b,t,yes] = s_b * (id[b,t] + 0.5*t); logits[b,t,no] = 0.3*id[b,t];
    everything else -10, EXCEPT the distractor row gets +100 at pad positions
    (so a gather that lands on the padded tail is loudly wrong) and +1000 at
    every position of rows containing the word 'oth'. s_b = -1 for rows
    containing 'neg' (so gap < 0 there), else +1."""

    def __init__(self, tok, yes_id, no_id, oom_above=None, extra_len=0,
                 dtype=torch.bfloat16):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.tok, self.yes_id, self.no_id = tok, yes_id, no_id
        self.oom_above, self.extra_len = oom_above, extra_len
        self.calls, self.seen_float_dtypes, self.seen_keys = [], set(), set()

    def forward(self, use_cache=False, **inputs):
        assert use_cache is False
        self.seen_keys |= set(inputs)
        for v in inputs.values():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                self.seen_float_dtypes.add(v.dtype)
        ids = inputs["input_ids"]
        B, T = ids.shape
        self.calls.append(B)
        if self.oom_above is not None and B > self.oom_above:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        V = len(self.tok.vocab)
        neg = self.tok.vocab.get("neg", -1)
        oth = self.tok.vocab.get("oth", -1)
        logits = torch.full((B, T + self.extra_len, V), -10.0)
        t = torch.arange(T, dtype=torch.float32)
        for b in range(B):
            s = -1.0 if bool((ids[b] == neg).any()) else 1.0
            logits[b, :T, self.yes_id] = s * (ids[b].float() + 0.5 * t)
            logits[b, :T, self.no_id] = 0.3 * ids[b].float()
            logits[b, :T, DISTRACTOR] = torch.where(ids[b] == PAD, 100.0, -10.0)
            if bool((ids[b] == oth).any()):
                logits[b, :T, DISTRACTOR] = 1000.0
        return _Out(logits.to(self.p.dtype))


def _fixture(n_rows):
    tmp = tempfile.mkdtemp(prefix="logit_dump_")
    rows = []
    for i in range(n_rows):
        rel = f"img_{i}.png"
        Image.new("RGB", (4, 4), color=(i * 7 % 256, 3, 5)).save(os.path.join(tmp, rel))
        # variable-length prompts so right padding is non-trivial; 'neg' rows
        # flip the sign of the yes logit; one 'oth' row makes a third token win
        words = ["Is", "there", "a"] + ["obj%d" % j for j in range(i % 4)]
        if i % 3 == 1:
            words.append("neg")
        if i == n_rows - 1:
            words.append("oth")
        rows.append({"id": f"pope_x_{i}", "image": rel,
                     "prompt": " ".join(words) + " ?",
                     "gt": "yes" if i % 2 == 0 else "no", "category": "adversarial"})
    return tmp, rows


def _setup(n_rows=7, **model_kw):
    tok = FakeTokenizer()
    bb = FakeBackbone()
    yes_id, no_id = resolve_answer_token_ids(tok, bb)
    proc = FakeProcessor(tok)
    model = FakeModel(tok, yes_id, no_id, **model_kw)
    tmp, rows = _fixture(n_rows)
    return tok, bb, proc, model, tmp, rows, yes_id, no_id


def _expected(tok, bb, row, yes_id):
    """Hand-computed z_yes/z_no at the last REAL position of this row."""
    ids = tok.encode_words(bb.build_prompt(row["prompt"]))
    last = len(ids) - 1
    s = -1.0 if "neg" in row["prompt"].split() else 1.0
    z_yes = s * (ids[last] + 0.5 * last)
    z_no = 0.3 * ids[last]
    return z_yes, z_no, len(ids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_resolve_answer_ids_in_context():
    if not HAVE_DEPS:
        return _skip("test_resolve_answer_ids_in_context")
    tok, bb = FakeTokenizer(), FakeBackbone()
    yes_id, no_id = resolve_answer_token_ids(tok, bb)
    assert yes_id == tok.vocab["Yes"] and no_id == tok.vocab["No"], (yes_id, no_id)
    assert yes_id != no_id
    try:
        resolve_answer_token_ids(UnstableTokenizer(), bb)
    except AssertionError as e:
        assert "context tokenization unstable" in str(e), e
    else:
        raise AssertionError("unstable tokenizer must be rejected")


def test_position_gather_under_right_padding():
    if not HAVE_DEPS:
        return _skip("test_position_gather_under_right_padding")
    tok, bb, proc, model, tmp, rows, yes_id, no_id = _setup(7)
    recs = run_dump(rows, model, proc, bb, tmp, batch=7, yes_id=yes_id, no_id=no_id,
                    device="cpu", adapter_label="none", log=lambda m: None)
    assert [r["id"] for r in recs] == [r["id"] for r in rows]
    assert model.calls == [7], model.calls
    lengths = set()
    for r, rec in zip(rows, recs):
        zy, zn, n = _expected(tok, bb, r, yes_id)
        lengths.add(n)
        # bf16 model dtype: compare at bf16 resolution
        assert abs(rec["z_yes"] - zy) <= 0.02 * max(1.0, abs(zy)), (rec, zy)
        assert abs(rec["z_no"] - zn) <= 0.02 * max(1.0, abs(zn)), (rec, zn)
        assert rec["n_input_tokens"] == n, (rec, n)
        assert abs(rec["gap"] - (rec["z_yes"] - rec["z_no"])) < 1e-9
        assert "image" not in rec and "prompt" not in rec
        assert rec["gt"] == r["gt"] and rec["category"] == r["category"]
    assert len(lengths) > 1, "fixture must exercise real padding (varied lengths)"
    # A gather at the padded tail would have put the distractor (+100) on top.
    assert all(rec["argmax_id"] != DISTRACTOR for rec in recs[:-1]), recs


def test_gap_sign_vs_argmax():
    if not HAVE_DEPS:
        return _skip("test_gap_sign_vs_argmax")
    tok, bb, proc, model, tmp, rows, yes_id, no_id = _setup(7)
    recs = run_dump(rows, model, proc, bb, tmp, batch=4, yes_id=yes_id, no_id=no_id,
                    device="cpu", log=lambda m: None)
    signs = set()
    for rec in recs[:-1]:
        assert rec["argmax_is"] in ("yes", "no"), rec
        assert (rec["gap"] > 0) == (rec["argmax_is"] == "yes"), rec
        assert rec["argmax_token"] == ("Yes" if rec["gap"] > 0 else "No"), rec
        signs.add(rec["gap"] > 0)
    assert signs == {True, False}, "fixture must produce both gap signs"
    last = recs[-1]
    assert last["argmax_is"] == "other" and last["argmax_id"] == DISTRACTOR, last
    assert last["argmax_token"] == "<distractor>", last
    assert abs(last["gap"] - (last["z_yes"] - last["z_no"])) < 1e-9


def test_oom_halving_matches_batch1():
    if not HAVE_DEPS:
        return _skip("test_oom_halving_matches_batch1")
    tok, bb, proc, model, tmp, rows, yes_id, no_id = _setup(7, oom_above=2)
    logs = []
    streamed = []
    recs = run_dump(rows, model, proc, bb, tmp, batch=8, yes_id=yes_id, no_id=no_id,
                    device="cpu", log=logs.append, on_rec=streamed.append)
    assert [r["id"] for r in recs] == [r["id"] for r in rows]
    assert len(streamed) == len(rows) and streamed[0] is recs[0]
    # 7 -> OOM -> 4,3 -> OOM,OOM -> 2,2,2,1 (each forwarded once)
    assert model.calls == [7, 4, 2, 2, 3, 2, 1], model.calls
    assert any("OOM at batch 7 -> splitting to 4" in m for m in logs), logs
    ref_model = FakeModel(tok, yes_id, no_id)
    ref = run_dump(rows, ref_model, proc, bb, tmp, batch=1, yes_id=yes_id, no_id=no_id,
                   device="cpu", log=lambda m: None)
    for a, b in zip(recs, ref):
        for k in ("z_yes", "z_no", "gap", "argmax_id", "argmax_is", "n_input_tokens"):
            assert a[k] == b[k], (k, a, b)
    # batch 1 must not be retried on OOM: it raises
    oom1 = FakeModel(tok, yes_id, no_id, oom_above=0)
    try:
        run_dump(rows[:1], oom1, proc, bb, tmp, batch=1, yes_id=yes_id, no_id=no_id,
                 device="cpu", log=lambda m: None)
    except torch.cuda.OutOfMemoryError:
        pass
    else:
        raise AssertionError("OOM at batch 1 must propagate")


def test_dtype_cast_and_passthrough():
    if not HAVE_DEPS:
        return _skip("test_dtype_cast_and_passthrough")
    tok, bb, proc, model, tmp, rows, yes_id, no_id = _setup(3)
    run_dump(rows, model, proc, bb, tmp, batch=3, yes_id=yes_id, no_id=no_id,
             device="cpu", log=lambda m: None)
    assert model.seen_float_dtypes == {torch.bfloat16}, model.seen_float_dtypes
    assert {"input_ids", "attention_mask", "pixel_values", "extra_flag"} <= model.seen_keys


def test_preconditions_fire():
    if not HAVE_DEPS:
        return _skip("test_preconditions_fire")
    tok, bb, proc, model, tmp, rows, yes_id, no_id = _setup(3)
    tok.padding_side = "left"
    try:
        run_dump(rows, model, proc, bb, tmp, batch=3, yes_id=yes_id, no_id=no_id,
                 device="cpu", log=lambda m: None)
    except AssertionError as e:
        assert "right padding" in str(e), e
    else:
        raise AssertionError("left padding must be rejected")
    tok.padding_side = "right"
    bad = FakeModel(tok, yes_id, no_id, extra_len=1)
    try:
        run_dump(rows, bad, proc, bb, tmp, batch=3, yes_id=yes_id, no_id=no_id,
                 device="cpu", log=lambda m: None)
    except AssertionError as e:
        assert "image-token expansion" in str(e), e
    else:
        raise AssertionError("logit/input length mismatch must be rejected")


def main():
    test_resolve_answer_ids_in_context()
    test_position_gather_under_right_padding()
    test_gap_sign_vs_argmax()
    test_oom_halving_matches_batch1()
    test_dtype_cast_and_passthrough()
    test_preconditions_fire()
    if HAVE_DEPS:
        print("LOGIT DUMP OK: in-context id resolution, right-padded position "
              "gather, gap sign vs argmax (+'other'), OOM-halving == batch-1, "
              "dtype cast/passthrough, precondition asserts.")


if __name__ == "__main__":
    main()
