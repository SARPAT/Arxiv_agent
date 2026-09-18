"""Checkpoint B's analysis logic, offline. No TypeSafe API, no Qdrant.

The thing most worth protecting here is the reporting contract: every
sweep row must carry in-corpus retention *and* out-of-corpus rejection.
A threshold that rejects all 50 out-of-corpus questions while dropping 20
good in-corpus answers is a worse system, and a table showing only the
rejection column would make it look like an improvement.
"""
import json
from pathlib import Path

import eval.jev_separation as js

# --- 1. The golden set's real labels, not invented ones ------------------
assert js.category_of({"type": "in_corpus"}) == "in_corpus"
assert js.category_of({"type": "out_of_corpus", "subtype": "unrelated"}) == "unrelated"
assert (
    js.category_of({"type": "out_of_corpus", "subtype": "adjacent_uncovered"})
    == "adjacent_uncovered"
)
# An out-of-corpus row with no subtype must not silently become in_corpus.
assert js.category_of({"type": "out_of_corpus"}) == "out_of_corpus_unlabelled"
print("PASSED: category_of() maps the golden set's own type/subtype labels.")

# And those labels really are what the file contains.
rows = [json.loads(line) for line in open("eval/golden_set.jsonl") if line.strip()]
cats = {js.category_of(r) for r in rows}
assert cats == {"in_corpus", "unrelated", "adjacent_uncovered"}, cats
assert sum(1 for r in rows if js.category_of(r) == "in_corpus") == 150
assert sum(1 for r in rows if js.category_of(r) == "unrelated") == 20
assert sum(1 for r in rows if js.category_of(r) == "adjacent_uncovered") == 30
print("PASSED: golden set really is 150 in_corpus / 20 unrelated / 30 adjacent_uncovered.")

# --- 2. Aggregators ------------------------------------------------------
agg = js.aggregate([0.1, 0.9, 0.5, 0.4])
assert agg["max"] == 0.9, agg
assert abs(agg["mean"] - 0.475) < 1e-9, agg
assert agg["count_above"] == 2, agg  # 0.9 and 0.5 are >= 0.5
assert js.aggregate([]) == {"max": 0.0, "mean": 0.0, "count_above": 0}
print("PASSED: aggregate() computes max / mean / count_above_0.5, empty-safe.")

# --- 3. THE REPORTING CONTRACT: both columns, always --------------------
questions = [
    {"category": "in_corpus", "aggregates": {"is_relevant": {"max": m, "mean": m, "count_above": 1}}}
    for m in (0.95, 0.90, 0.85, 0.80)
] + [
    {"category": "unrelated", "aggregates": {"is_relevant": {"max": m, "mean": m, "count_above": 0}}}
    for m in (0.40, 0.35)
] + [
    {"category": "adjacent_uncovered", "aggregates": {"is_relevant": {"max": m, "mean": m, "count_above": 0}}}
    for m in (0.70, 0.45)
]

rows_out = js.sweep(questions, "max", "is_relevant")
assert len(rows_out) == 13, len(rows_out)
assert rows_out[0]["threshold"] == 0.30 and rows_out[-1]["threshold"] == 0.90
for row in rows_out:
    for cat in ("in_corpus", "unrelated", "adjacent_uncovered"):
        assert f"{cat}_retained_pct" in row, (cat, row)
        assert f"{cat}_rejected_pct" in row, (cat, row)
        # The two must be complements - a row cannot report a rejection
        # rate that isn't the mirror of what it retained.
        assert abs(row[f"{cat}_retained_pct"] + row[f"{cat}_rejected_pct"] - 100.0) < 1e-9
print("PASSED: every sweep row carries retention AND rejection for every category,")
print("        and they are exact complements - the trap the spec names is unreachable.")

# --- 4. Sweep arithmetic on known data ----------------------------------
# in_corpus  = 0.95 0.90 0.85 0.80
# unrelated  = 0.40 0.35
# adjacent   = 0.70 0.45
at_75 = next(r for r in rows_out if r["threshold"] == 0.75)
assert at_75["in_corpus_retained"] == 4 and at_75["in_corpus_retained_pct"] == 100.0
assert at_75["unrelated_rejected_pct"] == 100.0           # both below 0.75
assert at_75["adjacent_uncovered_rejected_pct"] == 100.0  # 0.70 and 0.45 both below

at_50 = next(r for r in rows_out if r["threshold"] == 0.50)
assert at_50["in_corpus_retained"] == 4
assert at_50["unrelated_rejected_pct"] == 100.0           # 0.40, 0.35 below
assert at_50["adjacent_uncovered_rejected_pct"] == 50.0   # 0.70 clears, 0.45 does not

at_90 = next(r for r in rows_out if r["threshold"] == 0.90)
assert at_90["in_corpus_retained"] == 2                   # only 0.95 and 0.90
assert at_90["in_corpus_retained_pct"] == 50.0
print("PASSED: sweep arithmetic is correct on hand-checked data at 0.50 / 0.75 / 0.90.")

# --- 5. State shape is exactly the four locked fields -------------------
state = js.build_state("what is attention?", {
    "title": "Attention Is All You Need", "chunk_type": "body",
    "text": "The Transformer...", "paper_key": "attention",
    "cosine_score": 0.8, "chunk_id": "abc",
})
assert set(state) == {"query", "passage"}, state
assert set(state["passage"]) == {"title", "chunk_type", "text"}, state["passage"]
print("PASSED: state carries only query + {title, chunk_type, text} - no drift.")

# --- 6. Cache key identity ----------------------------------------------
k1 = js.Cache.key("jev-latest", "ic_001", "chunk-a", "hash1")
assert k1 != js.Cache.key("jev-other", "ic_001", "chunk-a", "hash1")
assert k1 != js.Cache.key("jev-latest", "ic_002", "chunk-a", "hash1")
assert k1 != js.Cache.key("jev-latest", "ic_001", "chunk-b", "hash1")
assert k1 != js.Cache.key("jev-latest", "ic_001", "chunk-a", "hash2")
print("PASSED: cache key varies with model, question, chunk and question wording.")

# Changing the instruction wording must bust the cache, or a re-run would
# serve answers to a question no longer being asked.
h = js._questions_hash()
assert isinstance(h, str) and len(h) == 12, h
print("PASSED: _questions_hash() is a stable 12-char digest of the instructions.")

# --- 7. Latency and token reporting -------------------------------------
pairs = [{"seconds": s, "input_tokens": 100, "output_tokens": 2} for s in
         (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
lat = js.latency_summary(pairs)
assert lat["n_calls"] == 10 and lat["max_seconds"] == 1.0, lat
assert "projection_assumption" in lat and "NOT measured" in lat["projection_assumption"]
print("PASSED: latency reports p50/p95/max and labels the fan-out as an assumption.")

tok = js.token_summary(pairs, None, None)
assert tok["total_input_tokens"] == 1000
assert tok["estimated_input_cost_usd"] is None and "No rate supplied" in tok["cost_note"]
tok2 = js.token_summary(pairs, 1.25, "2026-09-18")
assert tok2["estimated_input_cost_usd"] == round(1000 / 1_000_000 * 1.25, 4)
assert tok2["rate_as_of"] == "2026-09-18"
print("PASSED: cost is reported only with an explicit rate + date, never hardcoded.")

# --- 8. This checkpoint ships nothing -----------------------------------
repo = Path("eval/jev_separation.py").resolve().parent.parent
assert "typesafe" not in (repo / "requirements.txt").read_text().lower()
assert "matplotlib" not in (repo / "requirements.txt").read_text().lower()
cfg = (repo / "app/config.py").read_text()
assert "typesafe" not in cfg.lower() and "TYPESAFE" not in cfg
assert "TYPESAFE" not in (repo / "render.yaml").read_text()
print("PASSED: no SDK in requirements.txt, no TYPESAFE_API_KEY in config or render.yaml.")

# --- 9. --stub needs no credentials and no Qdrant -----------------------
# run() used to reach load_golden_set() through eval.run_eval, which
# imports rag.pipeline -> app.config, whose Settings validates REDIS_URL /
# QDRANT_URL / QDRANT_API_KEY at import time. That made the documented
# `--stub` invocation die on a pydantic ValidationError on a clean
# checkout, which is exactly what --stub exists to avoid.
src_text = Path("eval/jev_separation.py").read_text()
assert "from eval.run_eval import" not in src_text, (
    "jev_separation imports eval.run_eval again - that drags in app.config "
    "and breaks --stub on a machine with no Qdrant/Redis credentials"
)
golden = js.load_golden_set()
assert len(golden) == 200 and golden[0]["id"] == "ic_001", len(golden)
print("PASSED: golden set loads without importing app.config (--stub needs no creds).")

# --- 10. count_above is swept over chunk counts, not over THRESHOLDS ----
# It is a count in 0..TOP_K, not a score in [0, 1]. Swept against
# 0.30-0.90 every row collapsed to "at least 1 chunk" and all 13 came back
# identical - a table that reads as a flat result rather than as an
# aggregator that was never actually swept.
count_qs = [
    {"category": "in_corpus", "aggregates": {"is_relevant": {"count_above": n}}}
    for n in (8, 6, 3, 1)
] + [
    {"category": "unrelated", "aggregates": {"is_relevant": {"count_above": n}}}
    for n in (0, 1)
]
count_rows = js.sweep(count_qs, "count_above", "is_relevant")
assert [r["threshold"] for r in count_rows] == list(range(1, js.TOP_K + 1)), count_rows
assert len({r["in_corpus_retained"] for r in count_rows}) > 1, (
    "count_above sweep is flat - the cut is being ignored"
)
at_1 = count_rows[0]
assert at_1["in_corpus_retained"] == 4 and at_1["unrelated_retained"] == 1
at_4 = next(r for r in count_rows if r["threshold"] == 4)
assert at_4["in_corpus_retained"] == 2          # 8 and 6 clear 4
assert at_4["unrelated_rejected_pct"] == 100.0  # neither 0 nor 1 does
for row in count_rows:
    for cat in ("in_corpus", "unrelated"):
        assert abs(row[f"{cat}_retained_pct"] + row[f"{cat}_rejected_pct"] - 100.0) < 1e-9
print("PASSED: count_above is swept over 1..TOP_K chunks and actually varies,")
print("        still carrying retention AND rejection on every row.")

# --- 11. The adapter really parses a genuine SystemOneResponse ----------
# The SDK is measurement-only and not in requirements.txt, so this section
# skips cleanly when it is absent rather than failing the suite.
try:
    from typesafe_sdk import NoulAnswer, SystemOneResponse, Usage
except ImportError:
    print("SKIPPED: typesafe-sdk not installed - adapter parsing not checked.")
else:
    class _FakeClient:
        def system_one(self, state, questions):
            return SystemOneResponse(
                model="jev-1.2.0-20260901",
                usage=Usage(input_tokens=431, output_tokens=2),
                answers={
                    "is_relevant": NoulAnswer(type="noul", noul=0.87),
                    "contains_answer_evidence": NoulAnswer(type="noul", noul=0.61),
                },
            )

        def close(self):
            pass

    scorer = js.JevScorer.__new__(js.JevScorer)
    scorer._client = _FakeClient()
    scorer._questions = js._questions()
    scorer.model = "jev-latest"
    out = scorer.score(js.build_state("what is attention?", {
        "title": "Attention Is All You Need",
        "chunk_type": "body",
        "text": "The Transformer...",
    }))
    assert out["is_relevant"] == 0.87 and out["contains_answer_evidence"] == 0.61, out
    assert out["input_tokens"] == 431 and out["output_tokens"] == 2, out
    # The versioned id must come off the response, never be assumed to be
    # the id that was requested.
    assert out["model_version"] == "jev-1.2.0-20260901", out
    assert isinstance(out["seconds"], float), out
    print("PASSED: JevScorer parses a genuine SystemOneResponse; model id read"
          " from the response.")

    # The stub-mode fallback literal in _questions_hash() must stay byte-
    # identical to the real Noul text, or the same wording would hash two
    # ways depending on whether the SDK happened to be installed.
    with_sdk = js._questions_hash()
    real = js._questions
    js._questions = lambda: (_ for _ in ()).throw(ImportError("no sdk"))
    try:
        without_sdk = js._questions_hash()
    finally:
        js._questions = real
    assert without_sdk == with_sdk, (
        "the ImportError fallback text in _questions_hash() has drifted from "
        "the real Noul instructions - the same wording would hash two ways"
    )
    print("PASSED: _questions_hash() fallback matches the real instruction text.")

# --- 12. A warm cache serves every pair and makes no calls --------------
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / "cache.json"
    c1 = js.Cache(cache_path)
    key = js.Cache.key("jev-latest", "ic_001", "chunk-a", "hash1")
    assert c1.get(key) is None and c1.misses == 1
    c1.put(key, {"is_relevant": 0.9})
    c1.save()

    c2 = js.Cache(cache_path)
    assert c2.get(key) == {"is_relevant": 0.9}, "warm cache did not serve the pair"
    assert c2.hits == 1 and c2.misses == 0, (c2.hits, c2.misses)
print("PASSED: Cache round-trips to disk; a warm re-run serves hits, not calls.")

print("\nALL CHECKPOINT B ANALYSIS TESTS PASSED")
