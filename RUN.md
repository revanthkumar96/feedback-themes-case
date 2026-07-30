# Running the submission

## Requirements

- Python 3.11 or newer
- A Groq API key
- No third-party Python packages

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

The run writes `out/results.json` and `out/flat.json`. Each successful batch is
also atomically checkpointed under `out/checkpoints/`. If a provider or machine
failure interrupts the run, use:

```powershell
python -m feedback_themes run --resume
```

A checkpoint is accepted only when its review IDs, model, reasoning setting,
prompt version, batch size, and taxonomy hash all match the current run.

## Verification

Run the local test suite without an API key:

```powershell
python -m unittest discover -s tests -v
```

The supplied structural checker should report clean `ROWS` and `TREE` sections:

```powershell
python score.py --pred out/flat.json
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
