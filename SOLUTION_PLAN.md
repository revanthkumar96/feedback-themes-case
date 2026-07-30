# Proposed solution plan: hierarchical customer feedback themes

## Executive summary

My goal is to build a small, auditable pipeline that turns each review into zero or more recurring feedback themes arranged in a strict three-level hierarchy:

```text
strategic theme
└── midlevel theme
    └── specific theme
```

I would optimize for semantic faithfulness, consistency, and explainability before optimizing for sophistication. The dataset is small enough that a compact pipeline can process it well within the time and cost limits, but it is varied enough that a one-shot prompt or a clustering-only solution would hide important failure modes.

I considered three approaches:

1. embedding-based clustering and automatic cluster naming;
2. unconstrained end-to-end LLM generation;
3. a controlled hybrid pipeline that discovers a taxonomy, freezes it, and then performs evidence-backed classification.

I would choose the third approach. It preserves the flexibility needed to discover domain-specific themes while preventing labels and parent relationships from drifting during final classification. It also gives me a clear place to validate, retry, measure cost, and inspect mistakes.

The principal design decision is that the model will assign only a stable `specific_theme_id`. The midlevel and strategic themes will be resolved deterministically from the taxonomy registry. This makes an inconsistent tree impossible in the final output unless the registry itself is invalid.

---

## 1. How I interpret the task

The task is not ordinary sentiment analysis and it is not review summarization. A theme is a recurring subject that multiple customers can discuss positively, negatively, or neutrally. For example, "advisor continuity" can be mentioned by a satisfied customer who values a long-running relationship and by a dissatisfied customer whose advisor changed mid-application.

That distinction leads to five requirements:

- **Themes must describe subjects, not reactions.** Labels such as "positive experience", "frustration", or "disappointed customer" are not valid themes.
- **Themes must recur.** A label that describes only one review is probably an observation rather than a useful theme.
- **Reviews are multi-label.** A long review may discuss application speed, advisor knowledge, document handling, and account closure.
- **Abstention is valid.** A review such as "Bad." or an emoji can express sentiment without identifying a subject. Inventing a theme would be less faithful than returning no assignment.
- **The hierarchy must be globally stable.** A specific theme must always have the same midlevel parent, and a midlevel theme must always have the same strategic parent.

I would treat the published flat JSON as a derived interoperability format, not as the primary domain model. The primary representation should preserve the taxonomy, definitions, evidence, run metadata, and the difference between "no subject was present" and "classification failed."

---

## 2. Success criteria

I would consider the pipeline successful when it satisfies all of the following:

### Structural correctness

- Every assigned review ID exists in the source data.
- Every assignment references a known specific-theme ID.
- Every specific theme has exactly one midlevel parent.
- Every midlevel theme has exactly one strategic parent.
- Duplicate assignments are removed.
- The generated flat projection passes the supplied checker.

### Semantic quality

- Each assignment is supported by text from the review.
- Multi-subject reviews receive multiple assignments when appropriate.
- Vague reviews are not forced into invented categories.
- Similar subjects use the same canonical theme.
- Specific themes remain fine-grained enough to be actionable without becoming one-review descriptions.

### Operational quality

- A clean checkout can be run from one documented command.
- Raw model responses and checkpoints make failures diagnosable.
- Only invalid or uncertain batches are retried.
- Wall time, token use, model names, and cost are measured rather than estimated after the fact.
- The full run stays below the stated 25-minute and $6 limits.

### Communication quality

- The theme set is readable without opening the code.
- The output representation explains itself through stable IDs and definitions.
- Known mistakes are documented honestly, including why the architecture produced them.

---

## 3. Data decisions

### Language

I would use `content_en` as the primary classification text so the pipeline has one consistent language. I would preserve both language fields in the input layer and spot-check a few classifications against `content_no` when investigating apparent translation problems, but I would not double the normal inference cost by classifying both versions.

### Titles

The title can provide useful context when present, so I would include it in the review payload but mark it explicitly as optional. The classifier must work when it is empty.

### Ratings

I would retain the rating as metadata but not allow it to choose the theme. Ratings can help diagnose sentiment mismatches, but using them to infer subjects would encourage unsupported assignments for vague reviews.

### Multi-label behavior

The classifier may return multiple specific themes for a review. I would not impose a one-theme-per-review rule. I would set a reasonable safety ceiling, such as five assignments, to catch runaway output, but the normal prompt would emphasize completeness without duplication rather than targeting a fixed count.

### No-theme reviews

The rich output would explicitly record why a review received no assignments:

- `no_subject`: sentiment or reaction is present, but no feedback subject is identifiable;
- `irrelevant`: the text does not discuss the bank or customer experience;
- `classification_failed`: the pipeline could not obtain a valid result after retries.

Only the first two are legitimate empty classifications. `classification_failed` should make the run fail or be surfaced prominently rather than silently becoming an empty list.

---

## 4. Approaches considered

### Approach 1: embedding clustering with generated cluster names

#### How it would work

1. Split longer reviews into sentences or clauses.
2. Generate an embedding for each segment.
3. Cluster semantically similar segments.
4. Ask a language model to name each cluster.
5. Build a hierarchy by clustering the cluster names again.
6. Map cluster membership back to the originating reviews.

#### Advantages

- Relatively inexpensive after embeddings are generated.
- Provides a useful visual map of the corpus.
- Makes frequent subjects and outliers easy to inspect.
- Most steps are deterministic for a fixed model and clustering configuration.
- It does not require a predefined taxonomy.

#### Weaknesses

- Semantic similarity is not the same as thematic equivalence.
- Cluster boundaries are sensitive to distance thresholds and minimum cluster sizes.
- Very short texts have weak semantic signals.
- A segment can mention more than one subject, while normal clustering usually assigns one point to one cluster.
- Positive and negative descriptions of the same subject may separate because their surrounding language differs.
- Hierarchical clustering produces a mathematical tree, but not necessarily a useful business hierarchy.
- Naming a noisy cluster can make it appear more coherent than it really is.

#### Decision

I would not use clustering as the final classifier. I may use embeddings as a discovery aid for finding near-duplicate candidate labels and representative reviews, but every final theme and assignment should remain inspectable in natural language.

---

### Approach 2: unconstrained end-to-end LLM generation

#### How it would work

Each review would be sent to a model with instructions to return one or more complete paths:

```text
strategic > midlevel > specific
```

A later consolidation pass would attempt to merge synonyms and repair the tree.

#### Advantages

- Fastest approach to prototype.
- Handles short, implicit, and multi-subject language better than clustering alone.
- Requires little preprocessing.
- Can emit a rich explanation and evidence in the same call.
- Naturally supports abstention and multiple assignments.

#### Weaknesses

- Labels drift across calls: "approval delay", "slow approval", and "application processing time" may become separate themes.
- Consolidating after classification changes the meaning of already-produced assignments.
- Parent relationships may be inconsistent.
- A prompt that asks the model to discover, name, place, and assign themes simultaneously is difficult to evaluate.
- Large one-shot calls are hard to retry and can fail partially.
- The output can be structurally valid while semantically unsupported.
- Reproducibility is weak because the theme set may change on every run.

#### Decision

I would use an LLM for semantic extraction, but I would not let it invent taxonomy paths during the final classification stage. Discovery and assignment need to be separate operations with different constraints.

---

### Approach 3: controlled taxonomy discovery followed by evidence-backed classification

#### How it would work

The pipeline first extracts atomic candidate subjects from the corpus. It then consolidates those candidates into a reviewed, versioned three-level taxonomy. Once the taxonomy is frozen, the pipeline classifies every review against the fixed specific-theme IDs and attaches supporting evidence.

#### Advantages

- Combines open-ended discovery with controlled final output.
- Makes label definitions and hierarchy explicit.
- Prevents naming drift during classification.
- Supports multi-label review assignments.
- Makes parent paths deterministic.
- Permits targeted retries without regenerating the taxonomy.
- Separates taxonomy quality from assignment quality during evaluation.
- Produces evidence that makes manual error analysis much faster.
- Fits the small dataset and budget without model training.

#### Weaknesses

- More engineering than a one-shot prompt.
- Taxonomy quality depends on the discovery sample and consolidation decisions.
- Freezing the taxonomy can miss rare or genuinely novel subjects.
- Definitions and boundaries require deliberate review.
- A two-stage process introduces versioning and cache-invalidation concerns.

#### Decision

This is the approach I would implement. Its additional complexity directly addresses the two hardest parts of the task: developing a coherent theme set and applying it consistently. The remaining complexity can be kept modest through plain Python modules, JSON files, and a small provider abstraction.

---

## 5. Approach comparison

| Criterion | Embedding clustering | Unconstrained generation | Controlled hybrid |
|---|---:|---:|---:|
| Discovers new themes | Good | Very good | Very good |
| Handles multi-theme reviews | Weak | Good | Good |
| Stable labels | Medium | Weak | Strong |
| Guaranteed hierarchy | Weak | Weak | Strong |
| Evidence traceability | Medium | Good | Strong |
| Reproducibility | Medium | Weak | Strong |
| Implementation effort | Medium | Low | Medium |
| Cost for 223 reviews | Low | Low to medium | Low to medium |
| Ease of defending decisions | Medium | Weak | Strong |

The controlled hybrid is not the shortest implementation, but it creates the clearest relationship between input text, taxonomy decisions, final assignments, and validation.

---

## 6. Proposed architecture

I would separate the pipeline into design-time discovery and reproducible runtime classification.

```text
                         DESIGN TIME
reviews
  │
  ├─ validate and normalize
  │
  ├─ extract atomic candidate subjects + evidence
  │
  ├─ consolidate synonyms and inspect recurrence
  │
  ├─ construct the three-level hierarchy
  │
  └─ review and freeze taxonomy.json
                         │
                         ▼
                         RUNTIME
reviews + taxonomy.json
  │
  ├─ batch structured classification
  ├─ validate IDs, evidence, duplicates, and limits
  ├─ retry only invalid or uncertain items
  ├─ resolve parent paths deterministically
  ├─ write rich results.json
  └─ derive out/flat.json and run score.py
```

### Phase 1: ingestion and normalization

The loader would:

- parse the source JSON as UTF-8;
- verify required fields and unique IDs;
- normalize whitespace for prompts without modifying original text;
- retain exact original text for evidence checking;
- compute basic diagnostics such as review lengths, missing titles, and rating distribution.

No aggressive text cleaning is needed. Punctuation, casing, and clause boundaries can be meaningful, and the English text is already provided.

### Phase 2: candidate subject extraction

Reviews would be processed in small batches. For each review, the discovery model would return zero or more atomic candidate subjects with:

- a short provisional label;
- a one-sentence subject description;
- an exact evidence substring;
- a flag indicating whether the text contains no identifiable subject.

The prompt would explicitly reject sentiment-only labels and review summaries. Longer reviews may produce several candidates.

The extraction output is deliberately provisional. Candidate labels are not yet the final taxonomy.

### Phase 3: consolidation and hierarchy construction

Candidate subjects would be consolidated through:

1. exact normalization for casing and punctuation;
2. embedding similarity to propose likely synonym pairs;
3. model-assisted merging using candidate descriptions and representative evidence;
4. recurrence analysis to identify singleton or overly narrow labels;
5. hierarchy construction from specific to midlevel to strategic;
6. deterministic validation of the one-parent rule.

Embedding similarity would propose merges, not decide them. Two labels can be linguistically similar but operationally different, while differently worded labels can represent the same subject.

I would retain a small amount of human review at this stage because taxonomy construction is explicitly a design decision. That review would adjust definitions and boundaries, not hand-write per-review outputs or create an ID-based lookup table.

### Phase 4: taxonomy freeze

The final taxonomy would be checked into the repository as a standalone readable file. Each theme would have:

- a stable machine ID;
- a display label;
- a one-line definition;
- its parent ID, except at the strategic level;
- optional inclusion and exclusion guidance;
- optional representative examples used only for prompting and documentation.

The taxonomy receives a version and content hash. Any classification cache must include that hash so assignments made under an older taxonomy cannot be reused accidentally.

### Phase 5: fixed-taxonomy classification

The runtime classifier would receive:

- the review ID;
- optional title;
- English review text;
- the list of allowed specific themes with concise definitions and parent context.

It would return:

- zero or more `specific_theme_id` values;
- an exact evidence substring for each assignment;
- a short match explanation;
- an uncertainty flag when the review is ambiguous.

The model is not asked to emit strategic or midlevel labels. Those are joined from the taxonomy registry.

### Phase 6: validation and targeted correction

Every response is validated before it enters the final result:

- JSON conforms to the schema;
- returned review IDs match the batch;
- theme IDs exist;
- evidence is a substring of the review after conservative whitespace normalization;
- no duplicate specific theme is assigned to a review;
- the maximum assignment limit is not exceeded;
- no-theme reasons use an allowed value;
- an item cannot simultaneously contain assignments and a no-theme reason.

Only failed or uncertain items are retried. The correction prompt receives the validation error and the same taxonomy, which is cheaper and safer than rerunning successful reviews.

### Phase 7: output generation

The rich result is written first. The flat projection is generated deterministically from it by looking up each specific theme's parent path. The supplied checker then measures the resulting rows, tree, shape, and spread.

---

## 7. Taxonomy design rules

I would use the following rules when reviewing the taxonomy:

1. **Subject, not sentiment:** "application speed" is valid; "frustrating application" is not.
2. **Reusable:** a specific theme should normally apply to multiple reviews.
3. **Single parent:** a theme's placement cannot vary by review.
4. **Consistent granularity:** sibling themes should be comparable in specificity.
5. **Clear boundaries:** definitions should explain what belongs and what does not.
6. **Customer language:** labels should be understandable without internal banking jargon.
7. **No duplicated dimensions:** product area, process stage, and outcome should not be mixed arbitrarily at the same level.
8. **Limited catch-alls:** broad "other" themes hide gaps and should be avoided unless they have a defensible definition.
9. **Stable IDs:** wording can be refined without breaking references.
10. **Evidence coverage:** every specific theme should retain representative review evidence for auditability.

A likely tension is whether a lower-level theme should encode only the subject or also a problem type. I would keep polarity out of the taxonomy. "Advisor availability" should cover both easy and difficult access, while evidence or optional assignment metadata can capture the customer's stance.

---

## 8. Primary output model

I would use a normalized assignment model referencing a readable nested taxonomy:

```json
{
  "schema_version": "1.0",
  "taxonomy": {
    "version": "1.0",
    "strategic_themes": [
      {
        "id": "service",
        "label": "Service experience",
        "definition": "Customer feedback about receiving help and interacting with bank staff.",
        "midlevel_themes": [
          {
            "id": "advisor_relationship",
            "label": "Advisor relationship",
            "definition": "Feedback about the customer's ongoing relationship with an advisor.",
            "specific_themes": [
              {
                "id": "advisor_continuity",
                "label": "Advisor continuity",
                "definition": "Feedback about keeping or changing the advisor responsible for the customer."
              }
            ]
          }
        ]
      }
    ]
  },
  "reviews": [
    {
      "review_id": "rev-example",
      "assignments": [
        {
          "specific_theme_id": "advisor_continuity",
          "evidence": "the advisor we had spoken to left",
          "match_explanation": "The review explicitly describes an advisor change.",
          "uncertain": false
        }
      ],
      "no_theme_reason": null
    }
  ],
  "run": {
    "started_at": "...",
    "finished_at": "...",
    "models": [],
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0,
    "wall_seconds": 0,
    "taxonomy_hash": "..."
  }
}
```

This structure has several advantages:

- The hierarchy is readable and self-contained.
- Review assignments do not duplicate labels and definitions.
- Stable IDs allow labels to be edited safely.
- Parent paths are derived rather than regenerated.
- Evidence makes the classification auditable.
- The run block captures the numbers required by the brief.
- The flat output can be created with a small deterministic traversal.

The main tradeoff is that consumers must join an assignment with the taxonomy. I consider that worthwhile because it prevents denormalized copies of a label or parent path from disagreeing.

---

## 9. Prompt and model strategy

I would keep prompts versioned as files rather than embedding long strings throughout the code.

### Discovery prompt

The discovery prompt should:

- define a theme as a recurring subject;
- distinguish themes from sentiment and summaries;
- request atomic subjects rather than combined labels;
- allow multiple subjects and no subject;
- require exact evidence;
- include a few boundary-focused examples.

### Classification prompt

The classification prompt should:

- state that only supplied theme IDs are allowed;
- include concise definitions and relevant parent context;
- require evidence for every assignment;
- allow abstention;
- prohibit using rating as evidence;
- return strict schema-constrained JSON.

### Model selection

I would run a small quality-and-cost probe before committing to a model:

- a strong model for taxonomy consolidation, where judgement quality matters most;
- a smaller inexpensive model for repeated fixed-label classification;
- the strong model only for targeted adjudication of uncertain cases.

If a single inexpensive model performs adequately in the probe, using one model would simplify reproducibility. The provider and model names would remain configuration values so the pipeline is not tied to one vendor.

I would use deterministic settings where supported, but I would not claim that temperature zero makes a remote model perfectly deterministic. Stability comes primarily from the frozen taxonomy, strict schema, evidence validation, and caching.

---

## 10. Cost and latency controls

The dataset contains only 223 short reviews, so the budget should be comfortable if calls are batched and retries are targeted.

I would implement:

- batches of roughly 15–25 reviews, tuned during the probe;
- bounded concurrency to reduce wall time without triggering rate limits;
- a token estimate before each call;
- a hard configurable cost ceiling below $6 to leave retry headroom;
- response usage accounting from the provider;
- checkpoint files after every successful batch;
- cache keys based on model, prompt version, review content, and taxonomy hash;
- exponential backoff for transient provider errors;
- a low retry limit for malformed output;
- no model calls during flat projection or structural validation.

The run should stop with a clear error if the projected next call would cross the configured budget. I would report measured wall time and API usage from the clean full run. I would not invent numbers in advance.

---

## 11. Evaluation plan

The supplied checker is necessary but cannot measure whether an assignment is actually supported by a review. I would supplement it with the following measurements.

### Taxonomy measurements

- number of strategic, midlevel, and specific themes;
- children per parent;
- themes used once or twice;
- largest branch share;
- duplicate or highly similar labels;
- themes with overlapping definitions;
- themes with no representative evidence.

These are diagnostics, not automatic pass/fail scores. For example, a singleton may be a taxonomy mistake, or it may be a rare but important recurring-capable subject.

### Assignment measurements

- reviews with at least one theme;
- assignments per review;
- evidence-substring success rate;
- uncertain assignment count;
- retry count and cause;
- no-subject count;
- classification-failure count;
- theme frequency distribution.

### Holdout coverage check

During taxonomy development, I would hold out a stratified subset of reviews from initial discovery. After constructing the preliminary taxonomy, I would classify the holdout and inspect:

- reviews that need a subject absent from the taxonomy;
- specific themes with unclear boundaries;
- reviews receiving an overly broad fallback;
- false abstentions.

The holdout can inform one final taxonomy revision. After that revision, the taxonomy is frozen and the complete dataset is processed from scratch.

### Manual audit

I would manually inspect a stratified sample containing:

- short and long reviews;
- every rating;
- blank and non-blank titles;
- single-subject and multi-subject reviews;
- positive and negative references to the same subject;
- vague or sentiment-only reviews;
- low-frequency themes.

This audit evaluates the pipeline; it does not replace the pipeline with hand-written per-review answers.

### Stability check

I would rerun classification for a small sample with the same taxonomy and compare assignment sets. High disagreement would indicate unclear definitions or a weak classifier. Evidence differences alone are less concerning if the selected theme remains stable and both spans are valid.

---

## 12. Testing strategy

### Unit tests

- loading and required-field validation;
- duplicate review ID detection;
- taxonomy parent validation;
- unique theme ID validation;
- evidence normalization and substring matching;
- duplicate assignment removal;
- no-theme invariants;
- parent-path resolution;
- flat projection generation;
- token and cost aggregation;
- cache-key invalidation when prompts or taxonomy change.

### Provider contract tests

A fake provider would return:

- valid structured output;
- malformed JSON;
- unknown theme IDs;
- missing review results;
- duplicated assignments;
- evidence not present in the review;
- transient rate-limit errors.

These tests ensure retries and failure reporting work without spending API money.

### Integration test

A small fixture dataset would exercise discovery output parsing, fixed-taxonomy classification, rich output generation, flat projection, and the supplied checker.

### End-to-end verification

The final clean run would:

1. install dependencies;
2. run the complete pipeline;
3. run the supplied checker;
4. run the project tests;
5. record outputs and measured usage;
6. render and inspect the self-contained notes document.

---

## 13. Proposed repository structure

```text
.
├── data/
├── prompts/
│   ├── discover.md
│   ├── consolidate.md
│   ├── classify.md
│   └── correct.md
├── src/
│   └── feedback_themes/
│       ├── cli.py
│       ├── config.py
│       ├── ingest.py
│       ├── models.py
│       ├── provider.py
│       ├── discovery.py
│       ├── taxonomy.py
│       ├── classify.py
│       ├── validate.py
│       ├── projection.py
│       └── telemetry.py
├── tests/
├── taxonomy.json
├── out/
│   ├── results.json
│   └── flat.json
├── RUN.md
├── NOTES.html
└── pyproject.toml
```

I would avoid introducing an orchestration framework for a pipeline of this size. Plain async Python, typed data models, and explicit stages should be easier to understand, test, and defend.

---

## 14. Implementation sequence

### Milestone 1: contracts and deterministic core

- Define taxonomy and result schemas.
- Implement loading, validation, path resolution, and flat projection.
- Add unit tests.
- Verify a hand-constructed tiny fixture with the supplied checker.

### Milestone 2: provider and classification

- Add the provider abstraction.
- Implement structured batch classification.
- Capture token and cost metadata.
- Add checkpointing, retries, and fake-provider tests.

### Milestone 3: taxonomy discovery

- Implement candidate extraction.
- Add candidate consolidation and recurrence reports.
- Produce and review the initial three-level taxonomy.
- Run the holdout coverage check and freeze the taxonomy.

### Milestone 4: complete run and error analysis

- Process all reviews from scratch.
- Inspect shape and spread.
- Audit representative and uncertain cases.
- Identify at least five incorrect outputs and explain their causes.
- Refine code or prompts only through general rules, not per-review overrides.

### Milestone 5: submission packaging

- Finalize `RUN.md`.
- Generate self-contained `NOTES.html`.
- Record actual model, token, cost, and wall-time numbers.
- Run the complete clean-checkout verification.

---

## 15. Key tradeoffs and risks

### Stable taxonomy versus novel-topic recall

A frozen taxonomy improves consistency but can miss new subjects. For this fixed dataset, a discovery and holdout cycle is appropriate. In production, I would add a monitored `unsupported_subject` path that proposes taxonomy changes without silently changing live labels.

### Fine-grained themes versus recurrence

Very specific themes are actionable but risk becoming single-review descriptions. I would use recurrence reports and boundary definitions to merge overly narrow labels while preserving genuinely different customer subjects.

### Evidence versus output size

Evidence increases output tokens and storage slightly, but it materially improves auditability and the quality of error analysis. With 223 short reviews, the tradeoff is clearly favorable.

### Manual review versus automation

Reviewing the taxonomy is legitimate design work; hand-labeling all review outputs is not. I would automate candidate extraction and final classification, while using human judgement only to define and validate the shared theme system.

### Model flexibility versus provider lock-in

A provider abstraction makes experiments easier, but too much abstraction can obscure useful provider features. I would keep the interface narrow: structured completion, token usage, and model metadata.

### Batching versus failure isolation

Larger batches reduce overhead but make partial failures more expensive. Moderate batches plus item-level validation provide a practical balance.

### Self-reported confidence

Model confidence is not calibrated probability. I would treat it as a diagnostic flag and rely more heavily on observable signals: valid evidence, taxonomy membership, prompt agreement, and stability across a small repeated sample.

---

## 16. Expected failure modes

I expect the hardest cases to include:

- sentiment-only reviews with no subject;
- long reviews containing several clauses and contrasting opinions;
- implicit references such as "it took forever" where the process must be inferred from context;
- overlapping boundaries between communication, responsiveness, and advisor availability;
- comments about price where the customer may mean rate, fee transparency, or value for money;
- operational events that could fit both a product and a service theme;
- sarcastic or reputation-focused comments;
- translation choices that soften or alter domain terminology;
- rare subjects that are real but do not recur enough to justify a specific theme.

The required five-error analysis should connect mistakes to these architectural causes. For example, a missed implicit theme might be caused by overly literal evidence requirements, while an overly broad label might indicate that sibling definitions are not discriminative enough.

---

## 17. What I would do with another week

With additional time, I would focus on evaluation rather than adding more orchestration:

- create a small double-reviewed gold set and measure inter-annotator disagreement;
- calibrate taxonomy boundaries using confusion pairs;
- compare two classifier models on the same frozen taxonomy;
- add an evidence-grounded adjudication pass for ambiguous reviews;
- evaluate Norwegian and English classification agreement;
- build a lightweight taxonomy browser showing counts and representative evidence;
- measure stability across repeated runs and prompt versions;
- add a production-style novel-subject queue;
- investigate whether a small local classifier can replace repeated API assignment after bootstrapping labels.

These improvements would make the system more measurable and maintainable without changing the core design.

---

## Final rationale

The controlled hybrid approach is the one I would defend because it turns an underspecified language task into two inspectable problems:

1. define a coherent and useful theme system;
2. apply that system faithfully to each review.

The first problem benefits from open-ended model reasoning and deliberate taxonomy review. The second benefits from strict IDs, structured output, exact evidence, deterministic parent lookup, and targeted validation. Keeping those concerns separate produces a pipeline that is modest in size, economical to run, easy to explain, and honest about its remaining uncertainty.
