# Taxonomy-development artifacts

These files preserve the route to the reviewed `themes.json`:

- `themes_phase0.json` and `themes_phase1.json` are candidate trees generated
  from two disjoint, rating-balanced 40-review samples.
- `themes_consolidated_raw.json` is the model's corrected consolidation. It was
  rejected as the final taxonomy because it retained vague "overview" leaves,
  one-child hierarchy layers, and coverage gaps.
- The corresponding `*_run.json` files record sample IDs, model settings,
  tokens, latency, cost estimates, and content hashes.

The final root-level taxonomy is a human-reviewed consolidation. These artifacts
are evidence and debugging material, not runtime inputs.
