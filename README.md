# ReconAI

**Track 4 — AI Finance Controller · Razorpay AI Buildathon**

An agent that closes one finance-ops loop end to end: it reconciles a merchant's
**internal ledger** against their **bank statement** and **payment-gateway
settlement report**, auto-matches everything that rules can explain, and uses an
LLM *only* on the leftover exceptions — to explain why each one didn't match and
what a human should do about it.

## The core design decision: where NOT to use AI

Matching two financial records is a solved, deterministic problem — same
reference, same amount, same date. Throwing an LLM at 200 records to do that
would be slower, more expensive, non-reproducible, and impossible to audit.

So ReconAI does the opposite of the reflex "put AI on everything":

| Stage | Tool | Why |
|-------|------|-----|
| Match ledger ↔ bank ↔ gateway | **Deterministic rules** (exact, then fuzzy within a fee/rounding tolerance and a settlement-lag window) | Fast, free, reproducible, and every decision is explainable as a rule |
| Explain the *unmatched residue* | **LLM (Gemini)** | Genuine ambiguity — is this a fee, a partial refund, a timing lag, or money that never arrived? |

On the synthetic batch, rules resolve **93% of records with 1.000 precision**, so
the LLM is invoked on only the **~7% that actually need judgment**. That ratio is
the point.

## The honesty guarantees

- **A record only counts as matched when *both* legs (bank and gateway) are
  found.** A one-legged match — the exact shape of a missing settlement or a
  partial refund — is never silently accepted; it's surfaced as an exception.
- **The classifier is required to answer `cannot_determine`** when evidence is
  weak, instead of inventing a confident label. The exception list is honest, not
  cherry-picked.
- **Everything is scored against ground truth**, not eyeballed. `eval.py` reports
  real precision / recall / F1, because the synthetic generator records the true
  cause of every record.
- **Full audit trail.** Every decision — matched by which rule, or flagged with
  which LLM cause and rationale — is written to SQLite, queryable per record.

## Architecture

```
data/generate_synthetic_data.py   3 sources + ground_truth.json (200 ledger records,
                                   7 injected scenarios)
reconai/
  loader.py       CSV loading
  matcher.py      deterministic 2-pass matching  ← NO AI
  llm_client.py   thin Gemini wrapper (call_json) ← swap providers in one file
  classifier.py   exception prompt + fixed cause vocabulary + cannot_determine
  audit.py        SQLite decision log
  pipeline.py     load → match → classify → audit → summary
app.py            Streamlit dashboard
eval.py           precision/recall/F1 + classification accuracy vs ground truth
tests/            matcher unit tests
```

## Run it

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py          # generate the batch
python eval.py                                   # matcher metrics (no API key needed)
python -m reconai.pipeline --no-llm              # full pipeline, matcher only
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
