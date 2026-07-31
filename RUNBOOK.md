# Runbook — customer feedback theme extraction

The operational guide for running, verifying, evaluating, and troubleshooting
the pipeline end to end. `RUN.md` is the quickstart; this document is the
complete run book.

## 1. What the pipeline does

```
reviews.json + frozen themes.json
        │
        ▼
Groq structured classification (batches of 10, checkpointed)
        │
        ▼
semantic contract validation (IDs, hierarchy, verbatim evidence, 5-cap)
        │
        ▼
out/results.json (rich)  +  out/flat.json (checker projection)
```

The model only ever chooses specific-theme leaf IDs from the frozen taxonomy;
midlevel and strategic parents are joined from the registry by code, so the
three-tier hierarchy cannot be contradicted. A review with no supported
subject gets an empty assignment list (surfaced as `no_relevant_theme` in the
rich output) — abstention is a first-class outcome, not a failure.

## 2. Prerequisites

- Python 3.11+ (no third-party packages for the submitted configuration)
- A Groq API key with access to `openai/gpt-oss-120b`

```powershell
$env:GROQ_API_KEY = "your-key"        # never logged or written to output
```

A `.env` file is git-ignored and NOT auto-loaded; export the variable in the
shell. On macOS/Linux use `export GROQ_API_KEY="your-key"`.

Free-tier note: the daily quota for 120B is 200k tokens; one full run costs
roughly 105k. Comparison runs and failed attempts count against the same
budget — see section 7 for what to do when the quota runs out mid-run.

## 3. Submitted configuration

| Setting | Value |
|---|---|
| Model | `openai/gpt-oss-120b` |
| Reasoning | low |
| Max completion tokens | 3,000 |
| Batch size | 10 reviews |
| Prompt | `classification-v5.1` (boundary rules + disclosed 5-assignment cap) |
| Taxonomy | frozen `themes.json` (4 / 14 / 31, polarity-neutral) |
| Retrieval | none (hybrid mode is opt-in, experimental) |

All of these are the checked-in defaults of `python -m feedback_themes run`.

## 4. The standard end-to-end run

```powershell
# 1. verify the code without an API key (61 tests, no network)
python -m unittest discover -s tests

# 2. classify all 223 reviews (~10 min active time on free tier)
python -m feedback_themes run

# 3. structural gate: well-formed rows, consistent three-tier hierarchy
python score.py --pred out/flat.json

# 4. semantic gate: score against the human-annotated holdout
python -m feedback_themes evaluate
```

Expected results, in order: `OK` from the test suite; clean `ROWS` and `TREE`
sections from the checker; an evaluation report written to
`out/evaluation.json`. Reference numbers for the submitted run: micro-F1
0.741, precision 0.729, recall 0.754, abstention precision/recall
0.750/0.818, evidence validity 1.000.

## 5. How the data annotation works

The structural checker cannot tell a faithful assignment from a confident
hallucination, so quality claims rest on an annotated holdout.

1. `python -m feedback_themes holdout` deterministically selects 50 reviews,
   stratified over all five ratings, avoiding taxonomy-discovery samples
   wherever a stratum allows (reused reviews are flagged
   `seen_during_discovery`). It writes `data/holdout_annotations.json` with
   `specific_theme_ids: null` per review.
2. A human replaces each `null` with the list of supported leaf IDs, judging
   only against the frozen theme definitions. The empty list `[]` is a
   deliberate annotation meaning "no supported subject" — endorsements,
   pure sentiment ("Never again."), and emoji-only reviews are annotated
   this way rather than force-fitted.
3. `python -m feedback_themes evaluate` compares the run against the
   annotations and reports precision/recall/F1 (micro and macro), exact
   theme-set match, multi-subject recall, abstention precision/recall,
   evidence validity, and the unsupported-assignment rate.

Annotation rules that matter in real cases (learned from this corpus):

- **Subjects, not sentiment.** "Reply in under an hour" and "55 minutes in a
  phone queue" carry opposite sentiment but the same subject and land in the
  same leaf. Rating is never a theme.
- **Every clause counts.** A review praising the advisor while attacking the
  portal gets both labels, each with its own verbatim evidence span.
- **Questions are subjects too.** "Do they accept sole proprietorships?"
  supports the application-requirements theme.
- **Sarcasm reads through.** "Fantastic to wait three weeks for a rejection"
  is decision turnaround.
- **Unattributed cost complaints abstain.** "Too expensive" without an
  identified fee or interest mechanism supports no pricing leaf.
- **At most five assignments** per review, keeping the most explicit
  subjects; the cap is stated in the prompt and enforced by the validator.

## 6. Comparing models or prompts

Never compare on the full corpus first; use the annotated holdout (5 batches
instead of 23):

```powershell
python -m feedback_themes run --subset data/holdout_annotations.json `
    --output-dir out/cmp-NAME --checkpoint-dir out/checkpoints-cmp-NAME `
    [--model openai/gpt-oss-20b]
python -m feedback_themes evaluate --results out/cmp-NAME/results.json `
    --output out/cmp-NAME/evaluation.json
```

Measure run-to-run stability by repeating the winning configuration and
passing `--baseline-results` to `evaluate`; the report adds the fraction of
reviews with identical assignment sets and the mean Jaccard overlap
(reference: 0.80 identical, 0.877 Jaccard for 120B-v5). The submitted
outputs always come from a full run.

## 7. Failure recovery

| Symptom | What it means | What to do |
|---|---|---|
| `Groq returned HTTP 429 … (TPM)` | Per-minute token limit | Nothing; the client retries up to 6 times using the provider's own wait guidance (ms/s/min all parsed). |
| `Groq returned HTTP 429 … (TPD) … try again in Nm` | Daily quota exhausted | Wait for the stated time (or switch to a fresh key), then `python -m feedback_themes run --resume`. Completed batches are never re-bought. |
| `Batch N: checkpoint was written under a different configuration; recomputing` | A checkpoint from an older prompt/model was found | Informational; that batch is recomputed, matching batches are still reused. |
| `HTTP 403, error code: 1010` on manual API tests | Cloudflare rejecting the default Python user-agent | Send a `User-Agent` header; the pipeline already does. |
| Semantic validation failure after 3 attempts | Model output violates the contract (unknown ID, paraphrased evidence, >5 assignments) | The batch fails loudly. Inspect the message; if the contract itself is wrong (it happened once: the undisclosed 5-cap), fix prompt/validator together and bump `PROMPT_VERSION`. |

A checkpoint is reused only when review IDs, taxonomy hash, prompt version,
model, reasoning, completion limit, and batch size all match. Bumping
`PROMPT_VERSION` intentionally invalidates old checkpoints.

## 8. Experimental hybrid retrieval (not submitted)

```powershell
pip install .[hybrid]     # fastembed, BGE-small
python -m feedback_themes run --hybrid --output-dir out/hybrid --checkpoint-dir out/checkpoints-hybrid
```

Measured candidate recall at top-12 was 79.2%, which would cap exactly the
recall the 120B model wins on — that is why it is off by default.

## 9. Artifact map

| Path | Meaning |
|---|---|
| `themes.json` | Frozen, human-reviewed taxonomy (canonical IDs) |
| `out/results.json` | Rich output: run metadata, taxonomy, assignments with evidence |
| `out/flat.json` | Deterministic lossy projection for `score.py` |
| `out/evaluation.json` | Holdout metrics for the current `out/results.json` |
| `out/cmp-*/` | Holdout comparison runs backing the NOTES quality table |
| `out/viz/` | Taxonomy tree and flat-projection charts (`make_viz.py` in the parent folder regenerates) |
| `data/holdout_annotations.json` | Human-annotated 50-review holdout (references frozen taxonomy by hash) |
| `artifacts/` | Taxonomy-discovery candidates and run metadata, kept for inspection |
| `NOTES.html` | Design decisions, measured numbers, and known failures |

## 10. Known failure modes (current, measured)

- **Coverage gaps** (model abstains or files under the nearest leaf): credit
  enquiries, account administration/closure, user and role management,
  unsolicited marketing. These need taxonomy v1.1, not prompt work.
- **Cross-channel consistency is under-recalled**: contradictions between
  channels tend to resolve to the subject being contradicted (decision,
  price) instead of the inconsistency itself.
- **Initiator confusion**: customer-requested plan changes can read as
  unilateral term changes.
- **Sentiment-rule edge**: strongly worded but subject-free verdicts are
  usually abstained correctly, but single-word subjects inside praise lists
  ("friendlier") are sometimes dropped, and vague price praise occasionally
  leaks into a fee leaf.
