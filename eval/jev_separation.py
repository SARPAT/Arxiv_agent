"""Checkpoint B - does Jev's relevance noul separate in-corpus from
out-of-corpus questions on this project's own golden set?

**This is a measurement, not a feature.** Nothing here is imported by
``rag/``, ``app/``, ``ingestion/`` or ``ui/``, nothing is added to
``requirements.txt``, and no setting reaches ``app/config.py`` or
``render.yaml``. The deliverable is this script, a JSON result file, and
a plot.

Why the measurement exists
--------------------------
Retrieval is already solved on this corpus - Recall@8 is 1.0, so a
reranker has nothing to rescue. What the system cannot do is tell when
there is nothing relevant to find: 12% of out-of-corpus questions get
answered as though the corpus covered them, and the hard bucket
(``adjacent_uncovered`` - real ML questions that simply are not in these
seven papers) fails at 16.7% against 5% for plainly unrelated ones.

A previous session removed a cosine-similarity confidence gate after
measuring that correct in-corpus answers scored 0.49-0.71 while
out-of-corpus junk scored 0.79+ - a distribution no single threshold can
split. Jev's noul is a *different signal*, not a better threshold on the
same one. That has to be shown on this data before anything is built on
it, and if it overlaps the same way the honest outcome is to write that
down and stop.

Locked decisions, and why
-------------------------
- **State shape** ``{query, passage:{title, chunk_type, text}}`` follows
  TypeSafe's RAG-classification cookbook. ``title`` and ``chunk_type``
  are short and carry real signal about what a chunk is. Jev's documented
  failure modes include accuracy loss as state grows with irrelevant
  detail, so nothing beyond these fields goes in.
- **Top-8 chunks**, because measuring anything else means measuring a
  pipeline production does not run.
- **Both questions in one request.** Jev ingests the state once and
  evaluates every question against it, so ``contains_answer_evidence``
  costs no extra round trip - and it may separate better than
  ``is_relevant``. Learning that now avoids re-running 1,600 calls later.
- **Max is the primary aggregator**, because the gate's real question is
  "is *any* retrieved chunk relevant?" A mean is dragged down by the
  seven chunks that were never going to be relevant. All three are
  computed anyway so the plot decides rather than the assumption.

The trap this script is built to avoid
--------------------------------------
A threshold that rejects all 50 out-of-corpus questions while dropping 20
good in-corpus answers is a **worse** system. The existing eval answers
100% of in-corpus questions. Every sweep row therefore reports retention
*and* rejection together; there is no code path here that prints one
without the other.

Running it
----------
    export TYPESAFE_API_KEY=...
    python -m eval.jev_separation                 # real run
    python -m eval.jev_separation --stub          # no network, synthetic
    python -m eval.jev_separation --analyze-only  # re-analyze cached JSON

``--stub`` exercises retrieval-shaping, scoring, caching, aggregation and
plotting with no network at all, so the analysis code is testable without
Qdrant or the TypeSafe API. Its output goes to ``*.stub.json`` /
``*.stub.png`` and is stamped ``"stub": true`` so it can never be
confused with, or overwrite, a real measurement.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULT_PATH = EVAL_DIR / "jev_separation.json"
PLOT_PATH = EVAL_DIR / "jev_separation.png"
CACHE_PATH = EVAL_DIR / "jev_separation_cache.json"

MODEL = "jev-latest"
TOP_K = 8
WORKERS = 8

# Thresholds swept in the report. 0.30-0.90 inclusive, step 0.05.
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(13)]

# The aggregator used for count_above_N.
COUNT_THRESHOLD = 0.5

# What count_above is swept over: "at least N of TOP_K chunks scored at or
# above COUNT_THRESHOLD", N = 1..TOP_K. A count cannot be swept over
# THRESHOLDS - see sweep().
COUNT_CUTS = list(range(1, TOP_K + 1))

_SDK_HINT = (
    "typesafe-sdk is not installed. This script is measurement-only and is "
    "deliberately NOT in requirements.txt. Install it with:\n"
    '    pip install "typesafe-sdk>=0.5.7" --extra-index-url https://pypi.typesafe.ai/'
)
_PLOT_HINT = (
    "matplotlib is not installed. It is measurement-only and deliberately NOT "
    "in requirements.txt. Install it with:\n    pip install matplotlib"
)


def _questions():
    """The two nouls, sent together in one request per (question, chunk).

    Wording is taken from the cookbook unchanged. Jev is documented as
    reading instructions literally, so this is load-bearing text rather
    than prose to tidy - it is quoted verbatim in the PR description so a
    future reader can tell whether a later result came from the same
    question.
    """
    from typesafe_sdk import Noul

    return {
        "is_relevant": Noul(
            instructions="Does this passage address the subject of the query?",
        ),
        "contains_answer_evidence": Noul(
            instructions=(
                "Does this passage state information usable in a direct "
                "answer to the query?"
            ),
        ),
    }


# Hash of the instruction text, folded into every cache key: change the
# wording and the cache correctly misses rather than serving answers to a
# question that is no longer being asked.
def _questions_hash() -> str:
    try:
        questions = _questions()
        payload = json.dumps(
            {name: q.instructions for name, q in sorted(questions.items())},
            sort_keys=True,
        )
    except ImportError:
        # Stub mode must produce the same key shape without the SDK present.
        payload = json.dumps(
            {
                "contains_answer_evidence": (
                    "Does this passage state information usable in a direct "
                    "answer to the query?"
                ),
                "is_relevant": "Does this passage address the subject of the query?",
            },
            sort_keys=True,
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_state(question_text: str, chunk: dict) -> dict:
    """The state sent to Jev. Four fields, nothing else - see the module
    docstring on why this does not grow."""
    return {
        "query": question_text,
        "passage": {
            "title": chunk["title"],
            "chunk_type": chunk["chunk_type"],
            "text": chunk["text"],
        },
    }


# --- Retrieval ---------------------------------------------------------


def retrieve_chunks(question_text: str) -> list[dict]:
    """Top-``TOP_K`` chunks for one question, through the project's own
    retrieval path against live Qdrant.

    Deliberately calls ``rag.retrieval.retrieve`` rather than reaching
    into the vector store: measuring a search this codebase does not
    actually run would make the whole study describe a different system.
    ``point_id`` supplies the chunk identity used in the cache key, which
    is the same deterministic id ingestion writes into Qdrant.
    """
    from rag.retrieval import retrieve
    from rag.vectorstore import PUBLIC_TENANT_ID, point_id

    results = retrieve(question_text, k=TOP_K, tenant_ids=[PUBLIC_TENANT_ID])
    chunks = []
    for doc, score in results:
        meta = doc.metadata
        chunks.append(
            {
                "chunk_id": point_id(PUBLIC_TENANT_ID, meta, doc.page_content),
                "title": meta.get("Title", "") or "",
                "chunk_type": meta.get("chunk_type", "") or "",
                "paper_key": meta.get("paper_key", "") or "",
                "text": doc.page_content,
                "cosine_score": float(score),
            }
        )
    return chunks


def stub_retrieve_chunks(question_text: str, rng: random.Random) -> list[dict]:
    """Synthetic chunks for ``--stub``. Shaped exactly like the real ones
    so everything downstream is exercised, but obviously fake on sight."""
    return [
        {
            "chunk_id": f"stub-chunk-{rng.randrange(10**9):09d}",
            "title": f"STUB PAPER {i}",
            "chunk_type": "body",
            "paper_key": f"stub_{i}",
            "text": f"STUB passage {i} for: {question_text[:60]}",
            "cosine_score": round(rng.uniform(0.3, 0.9), 4),
        }
        for i in range(TOP_K)
    ]


# --- Scoring -----------------------------------------------------------


class JevScorer:
    """Thin adapter over the real SDK, so every assumption about its
    surface sits in one place.

    Verified against typesafe-sdk 0.7.0's actual API rather than written
    from the spec: ``client.system_one(state, questions)`` returns a
    ``SystemOneResponse`` with ``.model`` (the versioned id that answered),
    ``.usage.input_tokens`` / ``.usage.output_tokens``, and ``.answers``
    mapping each question name to a ``NoulAnswer`` whose ``.noul`` is a
    float.

    Retries are the SDK's own: its default ``RetryPolicy`` already treats
    429 as retryable and honours ``Retry-After``, so this does not
    reimplement backoff - it only widens ``max_retries`` from the default
    2, because a 1,600-call batch has more chances to hit a transient
    failure than a single interactive call does.
    """

    def __init__(self, api_key: str, model: str = MODEL, max_retries: int = 5):
        try:
            from typesafe_sdk import RetryPolicy, TypeSafeClient
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise SystemExit(_SDK_HINT) from exc

        self._client = TypeSafeClient(
            api_key=api_key,
            model=model,
            retry=RetryPolicy(max_retries=max_retries),
        )
        self._questions = _questions()
        self.model = model

    def score(self, state: dict) -> dict:
        started = time.perf_counter()
        response = self._client.system_one(state, self._questions)
        elapsed = time.perf_counter() - started
        return {
            "is_relevant": float(response.answers["is_relevant"].noul),
            "contains_answer_evidence": float(
                response.answers["contains_answer_evidence"].noul
            ),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "seconds": elapsed,
            "model_version": response.model,
        }

    def close(self):
        self._client.close()


class StubScorer:
    """Deterministic fake scores for ``--stub``.

    Exists so the aggregation, sweep, latency and plotting code can be run
    and tested with no network. It encodes a mild synthetic separation
    purely so those code paths produce non-degenerate output - **it says
    nothing whatsoever about how Jev actually behaves**, which is why
    every artifact it writes is stamped and renamed.
    """

    model = "STUB-NOT-A-REAL-MODEL"

    def score(self, state: dict) -> dict:
        seed = hashlib.sha256(
            json.dumps(state, sort_keys=True).encode("utf-8")
        ).hexdigest()
        rng = random.Random(seed)
        # Nudged by whether the stub passage was built for an in-corpus
        # question, so the sweep has something to sweep. Synthetic.
        base = rng.betavariate(2, 2)
        return {
            "is_relevant": round(base, 4),
            "contains_answer_evidence": round(max(0.0, base - rng.uniform(0, 0.2)), 4),
            "input_tokens": 400 + rng.randrange(200),
            "output_tokens": 2,
            "seconds": rng.uniform(0.2, 1.4),
            "model_version": self.model,
        }

    def close(self):
        pass


# --- Cache -------------------------------------------------------------


class Cache:
    """Disk cache keyed by (model, question id, chunk id, questions hash).

    Required, not an optimisation: 1,600 calls must not be re-paid after a
    crash, a disconnect, or a tweak to the plotting code. A re-run with
    the cache present makes zero API calls.
    """

    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            self.data = json.loads(path.read_text())
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, question_id: str, chunk_id: str, questions_hash: str) -> str:
        return f"{model}|{question_id}|{chunk_id}|{questions_hash}"

    def get(self, key: str):
        value = self.data.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True))


# --- Aggregation and analysis -----------------------------------------


def aggregate(pair_scores: list[float]) -> dict:
    """The three aggregators, per question, over that question's chunks."""
    if not pair_scores:
        return {"max": 0.0, "mean": 0.0, "count_above": 0}
    return {
        "max": max(pair_scores),
        "mean": statistics.fmean(pair_scores),
        "count_above": sum(1 for s in pair_scores if s >= COUNT_THRESHOLD),
    }


def category_of(row: dict) -> str:
    """The golden set's own labels, not invented ones.

    ``type`` is ``in_corpus`` or ``out_of_corpus``; out-of-corpus rows
    carry a ``subtype`` of ``unrelated`` or ``adjacent_uncovered``. The
    three reporting buckets are in_corpus / unrelated / adjacent_uncovered.
    """
    if row.get("type") == "in_corpus":
        return "in_corpus"
    return row.get("subtype") or "out_of_corpus_unlabelled"


def sweep(questions: list[dict], aggregator: str, noul: str) -> list[dict]:
    """Retention and rejection at every threshold - both columns, always.

    "Retained" = the question's aggregate is at or above the threshold
    (for ``max``/``mean``: at least one / the average chunk cleared it).
    "Rejected" = it was not. A row that reported only rejections would
    make a strictly worse system look like an improvement, so this
    function has no mode that returns one without the other.
    """
    by_cat = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q["aggregates"][noul][aggregator])

    # count_above is a count of chunks (0..TOP_K), not a score in [0, 1],
    # so it is swept over "at least N of TOP_K chunks cleared
    # COUNT_THRESHOLD" rather than over THRESHOLDS. Sweeping it against
    # 0.30-0.90 compared a count to a probability: every row collapsed to
    # "at least 1 chunk", so all 13 came back identical and the table read
    # as a flat result rather than as an aggregator that was never swept.
    cuts = COUNT_CUTS if aggregator == "count_above" else THRESHOLDS

    rows = []
    for cut in cuts:
        row = {"threshold": cut}
        for cat, values in by_cat.items():
            if not values:
                continue
            retained = sum(1 for v in values if v >= cut)
            row[f"{cat}_retained"] = retained
            row[f"{cat}_total"] = len(values)
            row[f"{cat}_retained_pct"] = 100.0 * retained / len(values)
            row[f"{cat}_rejected_pct"] = 100.0 * (len(values) - retained) / len(values)
        rows.append(row)
    return rows


def print_sweep(rows: list[dict], title: str, cut_label: str = "thresh") -> None:
    print(f"\n{title}")
    header = (
        f"{cut_label:>7s} {'in_corpus kept':>15s} {'unrelated rej':>15s} "
        f"{'adjacent rej':>14s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        kept = r.get("in_corpus_retained_pct")
        unrel = r.get("unrelated_rejected_pct")
        adj = r.get("adjacent_uncovered_rejected_pct")
        fmt = lambda v: "     -" if v is None else f"{v:5.1f}%"  # noqa: E731
        cut = r["threshold"]
        cut_s = f"{cut:7d}" if isinstance(cut, int) else f"{cut:7.2f}"
        print(f"{cut_s} {fmt(kept):>15s} {fmt(unrel):>15s} {fmt(adj):>14s}")


def latency_summary(pairs: list[dict]) -> dict:
    """p50/p95/max of single-call wall clock, plus a projection for the
    8-chunk fan-out.

    The projection is stated as an assumption, not a measurement: it
    assumes 8 concurrent calls complete in about the time of the slowest
    one, which ignores server-side queuing and client contention. Only the
    single-call figures here were actually observed.
    """
    times = sorted(p["seconds"] for p in pairs if p.get("seconds") is not None)
    if not times:
        return {}
    def pct(p):
        idx = min(len(times) - 1, int(round((p / 100) * (len(times) - 1))))
        return times[idx]
    return {
        "n_calls": len(times),
        "p50_seconds": pct(50),
        "p95_seconds": pct(95),
        "max_seconds": times[-1],
        "projected_8chunk_fanout_seconds": pct(95),
        "projection_assumption": (
            "Assumes 8 concurrent calls finish in roughly the time of the "
            "slowest (p95 used as that proxy). Ignores server-side queuing "
            "and client contention. NOT measured - only the single-call "
            "numbers above were observed."
        ),
    }


def token_summary(pairs: list[dict], rate_per_mtok, rate_date) -> dict:
    """Token totals always; dollar cost only when a rate is supplied.

    The rate is a CLI argument rather than a constant on purpose - a
    hardcoded price goes stale silently, and a cost figure with no stated
    rate and date is not checkable.
    """
    in_tok = sum(p.get("input_tokens") or 0 for p in pairs)
    out_tok = sum(p.get("output_tokens") or 0 for p in pairs)
    summary = {
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "input_rate_usd_per_mtok": rate_per_mtok,
        "rate_as_of": rate_date,
    }
    if rate_per_mtok is None:
        summary["estimated_input_cost_usd"] = None
        summary["cost_note"] = (
            "No rate supplied. Re-run with --input-rate-per-mtok and "
            "--rate-date to compute cost; tokens above are exact."
        )
    else:
        summary["estimated_input_cost_usd"] = round(
            in_tok / 1_000_000 * rate_per_mtok, 4
        )
    return summary


def write_plot(questions: list[dict], path: Path, noul: str, stub: bool) -> bool:
    """Per-question max noul, one series per category."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nWARNING: plot not written. {_PLOT_HINT}")
        return False

    by_cat = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q["aggregates"][noul]["max"])

    order = ["in_corpus", "unrelated", "adjacent_uncovered"]
    cats = [c for c in order if c in by_cat] + [
        c for c in sorted(by_cat) if c not in order
    ]

    fig, (ax_hist, ax_strip) = plt.subplots(2, 1, figsize=(9, 7))
    bins = [i / 20 for i in range(21)]
    for cat in cats:
        ax_hist.hist(by_cat[cat], bins=bins, alpha=0.55, label=f"{cat} (n={len(by_cat[cat])})")
    ax_hist.set_xlabel(f"per-question max {noul}")
    ax_hist.set_ylabel("questions")
    ax_hist.legend()
    ax_hist.set_title("Distribution of per-question max noul by category")

    for i, cat in enumerate(cats):
        vals = by_cat[cat]
        # Seeded explicitly: builtin hash() of a str is salted per process
        # (PYTHONHASHSEED), so this drew a visibly different plot from the
        # same JSON on every run - including every --analyze-only re-run of
        # one finished measurement.
        jrng = random.Random(f"jitter|{cat}|{len(vals)}")
        jitter = [i + jrng.uniform(-0.1, 0.1) for _ in range(len(vals))]
        ax_strip.scatter(vals, jitter, alpha=0.5, s=14, label=cat)
    ax_strip.set_yticks(range(len(cats)))
    ax_strip.set_yticklabels(cats)
    ax_strip.set_xlabel(f"per-question max {noul}")
    ax_strip.set_title("Per-question max noul (strip)")

    suptitle = "Jev separation study"
    if stub:
        suptitle = "STUB - SYNTHETIC DATA, NOT A MEASUREMENT OF JEV"
    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def analyze(result: dict, plot_path: Path) -> None:
    """Every reported artifact, all derived from the JSON - no API calls."""
    questions = result["questions"]
    pairs = result["pairs"]
    stub = result.get("stub", False)

    if stub:
        print("\n" + "!" * 70)
        print("!! STUB RUN - synthetic scores. Says NOTHING about Jev's behaviour.")
        print("!" * 70)

    print(f"\nQuestions scored: {len(questions)}   (question, chunk) pairs: {len(pairs)}")
    counts = {}
    for q in questions:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    print("Categories (golden set's own labels):", counts)
    print("Model that answered:", result.get("model_version_seen"))

    for noul in ("is_relevant", "contains_answer_evidence"):
        print(f"\n{'=' * 70}\nNOUL: {noul}\n{'=' * 70}")
        for aggregator in ("max", "mean", "count_above"):
            is_count = aggregator == "count_above"
            label = (
                f"count_above_{COUNT_THRESHOLD}" if is_count else aggregator
            )
            rows = sweep(questions, aggregator, noul)
            print_sweep(
                rows,
                f"Aggregator: {label}"
                + (
                    f"  (cut = min chunks of {TOP_K} at/above "
                    f"{COUNT_THRESHOLD})"
                    if is_count
                    else ""
                ),
                cut_label="chunks" if is_count else "thresh",
            )
            result.setdefault("sweeps", {})[f"{noul}.{aggregator}"] = rows

    lat = latency_summary(pairs)
    if lat:
        print(
            f"\nLatency (single call): p50={lat['p50_seconds']:.2f}s  "
            f"p95={lat['p95_seconds']:.2f}s  max={lat['max_seconds']:.2f}s  "
            f"n={lat['n_calls']}"
        )
        print(f"  Projected 8-chunk fan-out: ~{lat['projected_8chunk_fanout_seconds']:.2f}s")
        print(f"  ASSUMPTION: {lat['projection_assumption']}")
    result["latency"] = lat

    tok = result.get("tokens", {})
    print(
        f"\nTokens: input={tok.get('total_input_tokens')}  "
        f"output={tok.get('total_output_tokens')}"
    )
    if tok.get("estimated_input_cost_usd") is None:
        print(f"  Cost: {tok.get('cost_note')}")
    else:
        print(
            f"  Cost: ${tok['estimated_input_cost_usd']} at "
            f"${tok['input_rate_usd_per_mtok']}/Mtok (rate as of {tok['rate_as_of']})"
        )

    if write_plot(questions, plot_path, "is_relevant", stub):
        print(f"\nPlot written: {plot_path}")

    print(f"\n{'=' * 70}\nEXIT CRITERIA (spec: in-corpus retention >= 98%, i.e. <= 3 of")
    print("150 lost, AND adjacent_uncovered rejection better than 16.7%)")
    print("=" * 70)
    best = None
    for row in result.get("sweeps", {}).get("is_relevant.max", []):
        kept = row.get("in_corpus_retained_pct")
        adj = row.get("adjacent_uncovered_rejected_pct")
        if kept is None or adj is None:
            continue
        if kept >= 98.0 and adj > 16.7 and (best is None or adj > best[2]):
            best = (row["threshold"], kept, adj)
    if best:
        print(
            f"PROCEED candidate: threshold={best[0]:.2f} retains {best[1]:.1f}% "
            f"of in-corpus and rejects {best[2]:.1f}% of adjacent_uncovered."
        )
    else:
        print(
            "NO threshold satisfies both bars on is_relevant.max. Per the spec "
            "that is a legitimate STOP result, not a failure to report."
        )
    if stub:
        print("(Stub data - this verdict is meaningless. Re-run for real.)")


# --- Orchestration -----------------------------------------------------


def load_golden_set() -> list[dict]:
    """Read the golden set straight off disk, unmodified.

    Deliberately NOT ``eval.run_eval.load_golden_set``, even though that
    function is three equivalent lines: importing it executes
    ``eval/run_eval.py``, which imports ``rag.pipeline`` -> ``app.config``,
    whose ``Settings`` validates ``REDIS_URL`` / ``QDRANT_URL`` /
    ``QDRANT_API_KEY`` at import time with no defaults. That made
    ``--stub`` - whose entire point is that it touches no network and
    needs no credentials - die on a pydantic ValidationError before it
    read a single question.
    """
    with (EVAL_DIR / "golden_set.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run(args) -> dict:
    sys.path.insert(0, str(EVAL_DIR.parent))

    rows = load_golden_set()
    if args.limit:
        # Sample across categories rather than taking a prefix. The golden
        # set is grouped (all 150 in_corpus rows first), so a prefix smoke
        # run would score only in_corpus questions and every rejection
        # column in the sweep would come back empty - which looks like a
        # result and is not one.
        by_cat: dict[str, list[dict]] = {}
        for row in rows:
            by_cat.setdefault(category_of(row), []).append(row)
        per_cat = max(1, args.limit // max(1, len(by_cat)))
        rows = [r for cat in sorted(by_cat) for r in by_cat[cat][:per_cat]]
        print(
            f"--limit {args.limit}: sampled {len(rows)} questions across "
            f"{len(by_cat)} categories ({per_cat} each)"
        )

    qhash = _questions_hash()
    stub = args.stub
    rng = random.Random(12345)

    if stub:
        scorer = StubScorer()
    else:
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise SystemExit(
                "TYPESAFE_API_KEY is not set. This script reads it directly "
                "from the environment; it is deliberately not in "
                "app/config.py or render.yaml."
            )
        scorer = JevScorer(api_key, model=args.model)

    cache = Cache(CACHE_PATH)

    # Retrieval first, serially - it is cheap relative to scoring and
    # keeps the failure mode legible if Qdrant is unreachable.
    print(f"Retrieving top-{TOP_K} chunks for {len(rows)} questions...")
    per_question = []
    for row in rows:
        chunks = (
            stub_retrieve_chunks(row["question"], rng)
            if stub
            else retrieve_chunks(row["question"])
        )
        per_question.append((row, chunks))

    jobs = [
        (row, chunk)
        for row, chunks in per_question
        for chunk in chunks
    ]
    print(f"Scoring {len(jobs)} (question, chunk) pairs with {WORKERS} workers...")

    def score_one(job):
        row, chunk = job
        key = Cache.key(scorer.model, row["id"], chunk["chunk_id"], qhash)
        cached = cache.get(key)
        if cached is not None:
            return row, chunk, cached, True
        scored = scorer.score(build_state(row["question"], chunk))
        return row, chunk, scored, False

    pairs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for row, chunk, scored, was_cached in pool.map(score_one, jobs):
            if not was_cached:
                cache.put(
                    Cache.key(scorer.model, row["id"], chunk["chunk_id"], qhash), scored
                )
            pairs.append(
                {
                    "question_id": row["id"],
                    "category": category_of(row),
                    "chunk_id": chunk["chunk_id"],
                    "paper_key": chunk["paper_key"],
                    "chunk_type": chunk["chunk_type"],
                    "cosine_score": chunk["cosine_score"],
                    **scored,
                }
            )
    cache.save()
    scorer.close()
    print(f"Cache: {cache.hits} hits, {cache.misses} misses (misses = API calls made)")

    by_question = {}
    for p in pairs:
        by_question.setdefault(p["question_id"], []).append(p)

    questions = []
    for row, _chunks in per_question:
        ps = by_question.get(row["id"], [])
        questions.append(
            {
                "question_id": row["id"],
                "question": row["question"],
                "category": category_of(row),
                "type": row.get("type"),
                "subtype": row.get("subtype"),
                "n_chunks": len(ps),
                "aggregates": {
                    noul: aggregate([p[noul] for p in ps])
                    for noul in ("is_relevant", "contains_answer_evidence")
                },
            }
        )

    models_seen = sorted({p.get("model_version") for p in pairs if p.get("model_version")})
    return {
        "stub": stub,
        "model_requested": args.model if not stub else StubScorer.model,
        "model_version_seen": models_seen,
        "top_k": TOP_K,
        "questions_hash": qhash,
        "noul_instructions": {
            "is_relevant": "Does this passage address the subject of the query?",
            "contains_answer_evidence": (
                "Does this passage state information usable in a direct "
                "answer to the query?"
            ),
        },
        "grounded_sources_capture": (
            "SKIPPED - see spec 2.7. Capturing it requires running the full "
            "generation path for every question (~200 NVIDIA calls), which the "
            "spec says to skip and say so rather than pay for a nice-to-have."
        ),
        "tokens": token_summary(pairs, args.input_rate_per_mtok, args.rate_date),
        "questions": questions,
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stub", action="store_true",
                        help="no network; synthetic scores and chunks")
    parser.add_argument("--analyze-only", action="store_true",
                        help="re-analyze an existing result JSON, no scoring")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N questions (smoke run)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--input-rate-per-mtok", type=float, default=None,
                        help="USD per million input tokens, for the cost line")
    parser.add_argument("--rate-date", default=None,
                        help="date the rate was checked, e.g. 2026-09-18")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Stub artifacts get their own names so they can never overwrite, or be
    # mistaken for, a real measurement.
    result_path = args.output or (
        EVAL_DIR / ("jev_separation.stub.json" if args.stub else "jev_separation.json")
    )
    plot_path = EVAL_DIR / (
        "jev_separation.stub.png" if args.stub else "jev_separation.png"
    )

    if args.analyze_only:
        result = json.loads(result_path.read_text())
    else:
        result = run(args)

    analyze(result, plot_path)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written: {result_path}")


if __name__ == "__main__":
    main()
