# ReconAI

**Track 4 — AI Finance Controller · Razorpay AI Buildathon**

An agent that closes one finance-ops loop end to end: it reconciles a merchant's
**internal ledger** against their **bank statement** and **payment-gateway
settlement report**, auto-matches everything that rules can explain, uses an LLM
*only* on the leftover exceptions to explain why each one didn't match, and then
**takes a bounded, gated resolution action** on each — posting an adjusting entry
to a mock ledger, raising a settlement query, or escalating to a human. It
doesn't stop at "here's a problem"; it closes the loop and reports the money it
reconciled and the money it flagged for recovery.

### The three stages

```
  MATCH            →   CLASSIFY           →   RESOLVE
  (rules, no AI)       (LLM, residue only)    (deterministic policy + gate)
  93% of records       ~7% that need           bounded action per exception,
  at 1.000 precision   judgment                gated behind human approval
```

## The core design decision: where NOT to use AI

Matching two financial records is a solved, deterministic problem — same
reference, same amount, same date. Throwing an LLM at 200 records to do that
would be slower, more expensive, non-reproducible, and impossible to audit.

So ReconAI does the opposite of the reflex "put AI on everything":

| Stage | Tool | Why |
|-------|------|-----|
| Match ledger ↔ bank ↔ gateway | **Deterministic rules** (exact, then fuzzy within a fee/rounding tolerance and a settlement-lag window) | Fast, free, reproducible, and every decision is explainable as a rule |
| Explain the *unmatched residue* | **LLM (Gemini)** | Genuine ambiguity — is this a fee, a partial refund, a timing lag, or money that never arrived? |
| Decide the money action + gate | **Deterministic policy** ([resolver.py](reconai/resolver.py)) | The LLM classifies; it **never** moves money. A rule maps cause → action, bounds the amount, and gates sensitive/large actions behind human approval |

On the synthetic batch, rules resolve **93% of records with 1.000 precision**, so
the LLM is invoked on only the **~7% that actually need judgment** — and even
then it only labels the cause. Every rupee that moves is decided by an
auditable, bounded rule. That separation is the point.

## The honesty & safety guarantees

- **A record only counts as matched when *both* legs (bank and gateway) are
  found.** A one-legged match — the exact shape of a missing settlement or a
  partial refund — is never silently accepted; it's surfaced as an exception.
- **The classifier is required to answer `cannot_determine`** when evidence is
  weak, instead of inventing a confident label. The exception list is honest, not
  cherry-picked.
- **Every money action is bounded and gated.** Sensitive actions (reversing a
  duplicate debit, chasing a missing settlement) *always* require human approval;
  money-moving actions above a configurable limit are gated; a low-confidence
  classification is escalated even when the action would be cheap. The agent
  auto-executes only what is provably safe.
- **Nothing executes on the LLM's say-so.** The model's only output is a cause
  label. A deterministic policy turns that into an action — so an LLM hallucination
  can misclassify, but it can never move money on its own.
- **Everything is scored against ground truth**, not eyeballed. `eval.py` reports
  real precision / recall / F1, because the synthetic generator records the true
  cause of every record.
- **Full audit trail.** Every decision — matched by which rule, flagged with which
  LLM cause, and which action was taken/gated and why — is written to SQLite,
  queryable per record.

## Architecture

```
data/generate_synthetic_data.py   3 sources + ground_truth.json (200 ledger records,
                                   7 injected scenarios)
reconai/
  loader.py       CSV loading
  matcher.py      deterministic 2-pass matching        ← NO AI
  llm_client.py   thin Gemini wrapper (call_json)      ← swap providers in one file
  classifier.py   exception prompt + fixed cause vocabulary + cannot_determine
  resolver.py     cause → bounded, gated action policy ← NO AI (never moves money)
  actions.py      executes/gates actions, posts to a mock ledger, tallies outcome
  audit.py        SQLite decision log
  pipeline.py     load → match → classify → resolve → execute/gate → audit → summary
app.py            Streamlit dashboard (with live approve-and-execute on gated actions)
eval.py           precision/recall/F1 + classification accuracy vs ground truth
tests/            matcher + resolver unit tests
```

## Run it

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py          # generate the batch
python eval.py                                   # matcher metrics (no API key needed)
python -m reconai.pipeline --sim                 # FULL loop incl. actions, no API key
                                                 # (--sim simulates the classifier from
                                                 #  ground truth so you can see the whole
                                                 #  resolve/gate/execute path offline)
python -m reconai.pipeline --no-llm              # matcher only, exceptions to manual review
```

To enable exception classification, add a Gemini key:

```bash
cp .env.example .env        # then put your GEMINI_API_KEY in .env
python -m reconai.pipeline  # full run: matcher + LLM on the residue
python eval.py --llm        # also scores LLM classification accuracy
streamlit run app.py        # dashboard
```

```bash
pytest tests/               # matcher correctness
```

## Metrics (synthetic batch, seed 42)

| Metric | Value |
|--------|-------|
| Ledger records | 200 |
| Auto-matched (rules) | 186 (93.0%) |
| Match precision / recall / F1 | 1.000 / 1.000 / 1.000 |
| Exceptions surfaced | 14 |
| Exception recall (true exceptions caught) | 1.000 |
| Matcher throughput | ~49,000 records/sec |
| LLM calls per run | 14 (only the residue) |
| Resolution actions planned | 14 (100% of exceptions get an action or a review) |
| Money gated for approval | ~₹368,000 |
| Money flagged for recovery | ~₹226,000 (missing settlements) |

Every one of those 14 actions is gated in this batch (all amounts exceed the
auto-approve limit, and reversals/recovery are always gated) — approving one in
the dashboard executes it and books it to the mock ledger, which is the
loop-closing moment to show on camera.

*(LLM classification-accuracy row is produced by `python eval.py --llm` once a
key is configured — see the demo video.)*

## What broke, and how I got out

*(Kept as a running log during the build — the failure-recovery criterion.)*

- **Fuzzy matcher silently matched nothing.** The candidate-scoring loop
  initialised `best_score = -1.0`, but every score is a *negative* penalty
  (`-(amount_delta + date_delta)`), so no real candidate ever beat the seed and
  fee/timing-lag records all fell through to "exception." Caught it because the
  matcher-only eval showed 0 fuzzy matches where ~60 were expected. Fixed by
  seeding with `float("-inf")`. Lesson baked into `tests/test_matcher.py` as
  explicit fee-deduction and timing-lag cases.
- *(more entries added as they happen during days 6–7)*

## Notes

- No Docker / no external services by design — pure Python + a SQLite file.
- `llm_client.py` is the only file that knows the provider; swapping Gemini for
  another model is a one-file change.
