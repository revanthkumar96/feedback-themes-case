# Running the submission

## Requirements

- Python 3.11 or newer
- A Groq API key
- No third-party Python packages for the submitted configuration

From a clean checkout, set the key in the process environment and run:

```powershell
$env:GROQ_API_KEY = "your-key"
python -m feedback_themes run
python score.py --pred out/flat.json
```

On macOS or Linux, use `export GROQ_API_KEY="your-key"` instead. The key is
never logged or written to output. A local `.env` file is ignored by Git but is
not automatically loaded.

The checked-in defaults reproduce the measured final configuration:

- `openai/gpt-oss-120b`
- low reasoning
- 3,000 maximum completion tokens
- batches of 10 reviews
- frozen `themes.json`
- no local embedding retrieval

The run writes `out/results.json` and `out/flat.json`. Each successful batch is
also atomically checkpointed under `out/checkpoints/`. If a provider or machine
failure interrupts the run, use:

```powershell
python -m feedback_themes run --resume
```

A checkpoint is reused only when its review IDs, model, reasoning setting,
prompt version, batch size, and taxonomy hash all match the current run; a
checkpoint written under a different configuration is reported and recomputed.

## Verification

Run the local test suite without an API key:

```powershell
python -m unittest discover -s tests -v
```

The supplied structural checker should report clean `ROWS` and `TREE` sections:

```powershell
python score.py --pred out/flat.json
```

## Semantic evaluation (no API key required)

The structural checker cannot measure faithfulness, so the pipeline ships a
human-annotated holdout workflow:

```powershell
python -m feedback_themes holdout
```

This deterministically selects a 50-review, rating-stratified holdout that
avoids the taxonomy-discovery samples wherever possible (reviews reused from
depleted rating strata are flagged `seen_during_discovery`) and writes
`data/holdout_annotations.json`. Replace each `null` `specific_theme_ids`
with the list of supported leaf IDs — an empty list records a correct
abstention — then score the run:

```powershell
python -m feedback_themes evaluate
```

It reports assignment precision/recall, micro- and macro-F1, exact theme-set
match, multi-subject recall, abstention precision/recall, evidence validity,
and the unsupported-assignment rate, and writes `out/evaluation.json`.
Pass `--baseline-results` with a second results file to measure run-to-run
stability.

For model or prompt comparison, `run --subset data/holdout_annotations.json`
classifies only the annotated holdout (5 batches instead of 23); the submitted
outputs always come from a full run.

## Experimental hybrid retrieval (not the submitted configuration)

An embedding-assisted mode ranks candidate themes locally as a soft prior
before classification. A preliminary experiment retained only 79.2% of
assignments at top-12, so it is off by default and requires an optional
dependency:

```powershell
pip install .[hybrid]
python -m feedback_themes run --hybrid --output-dir out/hybrid --checkpoint-dir out/checkpoints-hybrid
```

## Taxonomy-development commands

The submitted classification uses the reviewed, frozen `themes.json`. The
candidate-generation process is retained for inspection:

```powershell
python -m feedback_themes discover --sample-phase 0 --output artifacts/themes_phase0.json
python -m feedback_themes discover --sample-phase 1 --output artifacts/themes_phase1.json --metadata-output artifacts/taxonomy_phase1_run.json
python -m feedback_themes consolidate --candidates artifacts/themes_phase0.json artifacts/themes_phase1.json
```

Those commands generate candidates; they do not overwrite the reviewed taxonomy.
The candidate trees deliberately remain visible because their weaknesses explain
why human taxonomy review was necessary.
