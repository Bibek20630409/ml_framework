# Model Selection

How this framework chooses **which model family to ship** and **which
hyperparameters it runs with** — jointly, on cross-validation, against five
production criteria rather than a single score.

- [The problem](#the-problem)
- [What the framework did before P11](#what-the-framework-did-before-p11)
- [The pipeline](#the-pipeline)
- [1. Gating: discarding incompatible families](#1-gating-discarding-incompatible-families)
- [2. Joint selection and tuning](#2-joint-selection-and-tuning)
- [3. Cross-validation strategy per data domain](#3-cross-validation-strategy-per-data-domain)
- [4. The five evaluation criteria](#4-the-five-evaluation-criteria)
- [5. The decision rule](#5-the-decision-rule)
- [6. Running candidates in parallel](#6-running-candidates-in-parallel)
- [Configuration reference](#configuration-reference)
- [CLI reference](#cli-reference)
- [Artifacts](#artifacts)
- [Worked examples](#worked-examples)
- [Cost](#cost)
- [Limitations](#limitations)

---

## The problem

A trained model arrives with exactly one number attached to it: its score.
Shipping on that number alone is how a 40 ms ensemble ends up behind a 20 ms SLA,
how a 2 GB model ends up in a serving image with a 500 MB budget, and how an
uninterpretable model ends up making credit decisions in a regulated market.

Model selection therefore has to answer five questions at once:

| Criterion | Why it matters in production | Effect on the choice |
|---|---|---|
| **Predictive performance** | High ROC-AUC / F1 / low RMSE on unseen data. | The model has to actually work. |
| **Inference latency** | Must return a prediction in < 20 ms for ad bidding or fraud scoring. | Eliminates heavy models that are too slow for real-time APIs. |
| **Memory & compute cost** | GPU/CPU cost of serving millions of requests a day. | Favours efficient models when budgets are tight. |
| **Explainability** | Credit scoring and healthcare need auditable attributions. | Rules out black boxes where regulators require feature importance. |
| **Maintainability** | How fast and reliably it retrains on fresh daily data without breaking. | Prefers architectures that train predictably. |

And it has to answer them for the *best-tuned instance of each family*, not for
each family's defaults — comparing a tuned XGBoost against an untuned MLP
measures the tuning, not the models.

---

## What the framework did before P11

Not nothing, but not this. Worth being precise about, because the pieces that
already existed are the ones the new code reuses rather than replaces.

| Capability | Before P11 | Where |
|---|---|---|
| Refuse an incompatible (task, kind, model) triple | **Yes** | `core/registry.py::validate_combination` |
| Enumerate compatible installed models, ranked | **Yes** | `core/registry.py::models_for` |
| Pick one model by data kind and row count | **Yes** | `config/defaults.py::select_model` |
| Tune hyperparameters within one model | **Yes** (Optuna TPE) | `pipeline/tune.py` |
| Stratified k-fold / plain k-fold | **Yes** | `data/splitters.py::CrossValidationSplitter` |
| Walk-forward (expanding + rolling) | **Yes** | `data/splitters.py::RollingOriginSplitter` |
| Holdout (random / temporal / grouped) | **Yes** | `data/splitters.py` |
| ROC-AUC, F1, RMSE, MAE, MASE, sMAPE | **Yes** | `core/metrics.py` |
| **Compare model families against each other** | **No** | — |
| **Tune and select jointly** | **No** — tuning ran inside one named model | — |
| **Score a trial on CV rather than one holdout** | **No** | — |
| **Purged / embargoed CV, CPCV** | **No** — `gap` was the only buffer | — |
| **Measure inference latency** | **No** — only Prometheus histograms at serving time | — |
| **Measure artifact size** | **Partly** — parameter/tree *counts*, no bytes | — |
| **Feature attribution** | **No** | — |
| **Maintainability signals** | **Partly** — `cv_*_std` existed, nothing read it | — |
| **Evaluate candidates in parallel** | **No** — everything sequential, DAG linear | — |

So: the ingredients existed; the driver that combines them did not. P11 adds the
driver and the four missing measurements, and reuses every row marked *Yes*.

---

## The pipeline

```
                    ┌─────────────────────────────────────────┐
   data + task ───► │ GATE                                    │
                    │  models_for(task, data_kind)            │  compatible
                    │  ∩ installed extras                     │  + installed
                    │  ∩ MODEL_RULES.fits(n_rows)             │  + right size
                    │  ∩ declared capability constraints      │  + eligible
                    └────────────────┬────────────────────────┘
                                     │  candidate pool
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        ┌───────────┐          ┌───────────┐          ┌───────────┐
        │ xgboost   │          │ lightgbm  │          │ mlp       │   ← concurrent
        │  tune     │          │  tune     │          │  tune     │     (processes,
        │  CV score │          │  CV score │          │  CV score │      or Airflow
        │  profile  │          │  profile  │          │  profile  │      mapped tasks)
        └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
              └──────────────────────┼──────────────────────┘
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │ DISQUALIFY on measured hard constraints │
                    │   p95 latency · artifact MB · explain   │
                    ├─────────────────────────────────────────┤
                    │ DECIDE  tolerance (default) | weighted   │
                    └────────────────┬────────────────────────┘
                                     ▼
                        winning config (model + params)
                                     │
                                     ▼
                        train() refits at full budget
                                     ▼
                          bundle + manifest.selection
```

Each stage returns a **config**, never a fitted model. That is what keeps
`train()` at a single bundle-writing path whether a bake-off ran, only tuning
ran, or neither did:

```python
select(config)  →  SelectionResult.config   # the winning family + its params
  tune(config)  →  TuneResult.config        # the tuned params
 train(config)  →  metrics                  # fits it once, writes the bundle
```

`select()` is a pass-through to `tune()` when `select.enabled` is false, which is
the default — so a run that does not ask for a bake-off behaves exactly as it did
before P11.

---

## 1. Gating: discarding incompatible families

Gating happens in **two phases**, because it has to: some constraints are
answerable before training and save the work, others cannot be known until a
model exists.

### A priori — before any fitting

| Filter | Source | Example |
|---|---|---|
| Task and data-kind compatibility | `validate_combination` | `xgboost` cannot consume an image folder |
| Extra installed | `MODELS.is_available` | `catboost` skipped, with the `pip install` line |
| Row-count rules | `MODEL_RULES` / `ModelRule.fits` | `ts.lstm` needs ≥ 200 rows |
| Declared explainability | `Capabilities.native_feature_importance` | `min_explainability: 0.9` drops `mlp` before fitting it |
| Pool cap | `select.max_candidates` | keeps a plugin-rich install from becoming 40 fits |

An explicit `select.candidates` list overrides discovery but still passes through
`validate_combination`, so a typo or a missing extra is an **error naming the
model** rather than a silently shorter bake-off.

### A posteriori — after each candidate is trained

Latency and size cannot be known before a model exists, so they disqualify after
the fact. A disqualified candidate is **reported with the number it missed by**,
not omitted:

```
lightgbm   0.8750  0.0179   2.4038  0.1980  1.00  rejected: p95 latency 2.40 ms exceeds the 1.4 ms budget
```

Telling someone their model is 1 ms too slow is actionable. Dropping it from the
table is not.

---

## 2. Joint selection and tuning

Each candidate is tuned **on its own search space** before being scored, so the
comparison is between best-tuned instances:

```
effective space = backend.search_space() | model.search_space | tune.overrides
```

`xgboost` is tuned on `max_depth`/`min_child_weight`/`reg_lambda` plus the GBDT
backend's `learning_rate`/`n_estimators`/`subsample`/`colsample_bytree`; the
`mlp` on `dropout` and a conditional layer stack plus the Lightning backend's
`lr`/`weight_decay`/`batch_size`. Neither is tuned on knobs it does not have —
the failure that made the pre-P4 `mlf hpo` meaningless.

### Scoring a trial on cross-validation

By default a trial is scored on the validation split (`tune.objective: holdout`).
With 30 trials against a small validation set, part of what wins is *luck on
those rows*, and the tuned model then underperforms its own reported number.

```yaml
tune:
  objective: cv      # average the objective across inner folds
  cv_folds: 3        # inner folds, independent of data.split.folds
```

`tune.cv_folds` is deliberately separate from `data.split.folds`. Sharing one
number would make the honest nested arrangement — 5 outer folds for the estimate,
3 inner folds per trial — impossible to express.

Pruning is **not** wired into the inner loop: a pruner comparing partial fold
means across trials would prune on a quantity that means different things at
different fold counts. The outer study still prunes on the returned value.

---

## 3. Cross-validation strategy per data domain

`data.split.cv_strategy` chooses how folds are cut once `data.split.folds >= 2`.
It is orthogonal to `data.split.strategy`, which cuts the single holdout — a run
can hold out temporally and cross-validate with rolling origins, and those are two
separate statements.

| `cv_strategy` | Class | Use when | Guards against |
|---|---|---|---|
| `auto` *(default)* | — | You have not thought about it yet | Resolves per the table below |
| `stratified` | `CrossValidationSplitter` | Tabular classification | Imbalanced folds, high evaluation variance |
| `kfold` | `CrossValidationSplitter` | Regression, or classification you do not want stratified | Stratifying a continuous target (which raises) |
| `rolling_origin` | `RollingOriginSplitter` | Time series, forecasting | Training on the future |
| `purged` | `PurgedKFoldSplitter` | Overlapping labels, serially correlated features | Training on the test set's own observations |
| `cpcv` | `CombinatorialPurgedSplitter` | You need the score's *distribution*, not one path | Judging a model on one backtest arrangement |

### How `auto` resolves

```
time_col set, or data.kind == timeseries   →  rolling_origin
label_horizon / label_end_col / embargo set →  purged
task is binary or multiclass                →  stratified
otherwise                                   →  kfold
```

The order encodes which mistake is worse. Time ordering wins first, because a
shuffled fold on temporal data reports a *better* score. Purging settings win
next: declaring a label horizon and then getting plain k-fold would ignore the
one thing the user said about their data.

### Walk-forward (`rolling_origin`)

```yaml
data:
  split:
    folds: 4
    cv_strategy: rolling_origin
    horizon: 12       # steps ahead each fold forecasts
    expanding: true   # grow the window (a production retrain); false slides it
    gap: 2            # rows dropped between train and test
```

```
fold 0:  [train........]  gap  [val]  gap  [test]
fold 1:  [train...........]    gap  [val]  gap  [test]
fold 2:  [train..............]      gap  [val]  gap  [test]
```

`expanding: true` mirrors what a production retrain does. `expanding: false`
slides a fixed window, which is what you want when old data is actively
misleading — a regime change, a changed measurement process.

### Purged k-fold with embargo

For data where rows are **not independent**. Plain k-fold assumes they are and
quietly reports a score that assumes it too.

```yaml
data:
  split:
    folds: 5
    cv_strategy: purged
    label_horizon: 10   # a row's label is computed from the next 10 rows
    embargo: 0.01       # drop a further 1% of rows after each test block
```

Two mechanisms, both required, neither sufficient alone:

- **Purge** drops training rows whose *label span* reaches into the test window.
  A 10-step-ahead return at row `i` is not known until row `i + 10`, so training
  on row `i` while testing row `i + 10` fits a target the model is about to be
  graded against. Note it is the span that is checked, not the row index — a
  training row far before the test set still leaks if its horizon reaches inside.
- **Embargo** drops training rows in a window immediately *after* the test set.
  Purging looks forward; the embargo handles leakage backward, through serial
  correlation. A feature computed at `test_end + 1` correlates with observations
  inside the test window even though no label spans it.

```
                    purged            test            embargo
    [ train ....... XXXXXX ] [ ============== ] [ XXXX ] [ train ... ]
                    ▲                                ▲
        labels reach into test              serially correlated
```

Folds are **contiguous, not shuffled**: a shuffled fold interleaves test rows
through the training set, leaving every training row adjacent to a test row and
making the purge remove almost everything.

**Exact label spans.** `label_horizon` states the span in rows. For
variable-length labels (a triple-barrier target), `label_end_col` names a column
of per-row label end *times*. Mapping a time onto a row position needs the rows'
own observation times, so it requires `time_col` alongside it:

```yaml
data:
  split:
    cv_strategy: purged
    time_col: timestamp     # required with label_end_col
    label_end_col: t1       # per-row label end time
    embargo: 0.01
```

`embargo` below 1.0 is a fraction of the dataset; 1.0 or above is a literal row
count. `embargo: 1.0` means one row, not 100% of the data.

### Combinatorial purged CV (CPCV)

Purged k-fold gives **one** backtest path — each row is tested exactly once, in
one particular arrangement. That single path is itself a sample. CPCV splits the
data into `cpcv_groups` contiguous blocks and tests every combination of
`cpcv_test_groups` of them:

```yaml
data:
  split:
    folds: 2                # ignored; the count is C(groups, test_groups)
    cv_strategy: cpcv
    cpcv_groups: 6
    cpcv_test_groups: 2     # → C(6,2) = 15 folds
    cpcv_max_folds: 20      # the cost ceiling
    label_horizon: 10
```

Test groups are generally **not contiguous** — a fold that tests blocks 0 and 4
trains on the blocks between them, and the purge is applied around each test
block separately. One purge across the union would span the gap and delete the
training data inside it.

`cpcv_max_folds` keeps the first N combinations in lexicographic order — a
deterministic prefix, not a random sample, so a capped run is reproducible.

### Holdout

`folds: 0` (the default) is a single holdout split, cut by `data.split.strategy`
(`random` / `temporal` / `group` / `auto`). Cheapest, and the right choice when
GPU time is the binding constraint.

---

## 4. The five evaluation criteria

Every one is **measured**, on this machine with this data, in the same run and
under identical conditions. A latency measured on a laptop is not a production
latency — but it is a *comparable* number across candidates, and the ranking
transfers even when the absolute value does not.

### Predictive performance

The cross-validated mean of the task's primary metric, with its spread.

- From folds, not a single holdout: two candidates measured on one split differ
  partly by which one suited that split.
- The spread is kept, not just the mean: a mean of 0.85 across folds of
  0.84/0.86 and across 0.70/1.00 are the same number and completely different
  results. It is also the default tolerance and the stability term.

`ModelProfile.score`, `score_std`, `score_std_error`, `metrics` (all `cv_*_mean`
/ `cv_*_std`).

### Inference latency

`core/profile.py::measure_latency`. Warms up 5 calls (discarding lazy kernel
selection, cuDNN autotuning, JIT warmup, page faults), then times at least 20
single-row `predict` calls with `time.perf_counter`.

- `p95_ms` is per call at **batch size 1** — the number an online SLA is written
  against, and what `max_latency_p95_ms` compares to. The p95 rather than the
  mean because a tail that misses the budget is a timeout, and the mean hides it.
- `batch_ms_per_row` is throughput — what an offline scoring job is costed
  against. A GPU model is often terrible at the first and excellent at the second.
- `time.perf_counter` specifically: `time.time()` on Windows has ~16 ms
  granularity, which is most of a 20 ms budget.

Measured on the **estimator**, not through HTTP: request parsing, validation and
network time belong to the API, not the model, and including them would make
every candidate look the same. The serving-side Prometheus histograms in
`serving/metrics.py` answer a different question and neither replaces the other.

### Memory & compute cost

`core/profile.py::measure_cost`. Serializes the model to a scratch directory and
weighs what came out.

- `artifact_bytes` — the exact number. Saving is the only honest way to get a
  size comparable across a neural net, a tree ensemble and an HF model directory:
  a parameter count does not know about dtype, a tree count does not know about
  node payloads, and neither knows about the tokenizer.
- `native` — the backend's own counts (`trainable_parameters`, `trees`, `nodes`,
  `series`). Informational and per-backend; never compared across backends.

The scratch directory is cleaned up — a five-candidate bake-off must not leave
five model copies behind.

### Explainability

`core/explain.py`. A **tiered** capability, because "is this model explainable?"
has no useful boolean answer:

| Tier | Score | How | Cost |
|---|---|---|---|
| `native` | **1.0** | `feature_importances_` / `coef_` | Free |
| `shap` | **0.8** | `shap.TreeExplainer` (trees only) | Seconds |
| `permutation` | **0.5** | `sklearn.inspection.permutation_importance` | One predict pass per feature per repeat |
| `none` | **0.0** | Nothing attributable | — |

The score is what `min_explainability` compares against; the *values* go to
`feature_importance.json`, because a regulator asking "why was this application
declined?" wants the numbers, not the tier.

The ordering is not arbitrary. Native importances are exact statements about the
fitted model and cost nothing. SHAP is more informative (per-prediction, signed,
additive) but is an approximation for anything that is not a tree — and a
sampling explainer on a neural net takes minutes, inside a routine that is
*also* measuring latency. Permutation is model-agnostic and honest but measures
*this dataset's* dependence rather than the model's structure.

Reaching `native` and `permutation` needs **no extra** — the tree libraries carry
importances and sklearn is a base dependency. `pip install 'ml-framework[explain]'`
buys the middle tier only.

### Maintainability

The image's framing — "how fast and reliably the model can be retrained on fresh
daily data without breaking" — decomposes into three observable things:

| Signal | Meaning |
|---|---|
| `fit_seconds` | The retrain window this model needs, every day, forever |
| `fold_stability` | `1 - |std / mean|`, clamped to [0, 1]. A score that swings between folds will swing between retrains |
| `fold_failures` | Folds that raised. Nonzero means the training path is fragile against ordinary data variation |

`MaintainabilityProfile.score` combines them, with any fold failure dominating: a
pipeline that fails one fold in five is not 80% maintainable, it is a pager at
3am.

`fold_stability` uses the coefficient of variation because it is scale-free, which
is what makes an accuracy of 0.9 and an RMSE of 340 comparable on this axis.

---

## 5. The decision rule

### `objective: tolerance` (default)

**Hard constraints disqualify. Among survivors, take the best score, then the
cheapest model statistically tied with it.**

```python
survivors = [c for c in candidates if c.meets(constraints)]
best      = max(survivors, key=cv_score)
tol       = select.tolerance or best.score_std_error
ties      = [c for c in survivors if best.score - c.score <= tol]
winner    = min(ties, key=lambda c: (c.latency_p95,      # 1st
                                     c.artifact_mb,      # 2nd
                                     -c.explainability,  # 3rd
                                     -c.stability))      # 4th
```

The rule practitioners actually use, made explicit: *take the simplest model that
is not measurably worse than the best one.* A 0.9012 model answering in 3 ms and
a 0.9019 model answering in 40 ms are the same model as far as the data can tell,
and the 3 ms one is obviously the one to deploy.

**"Not measurably worse" defaults to one standard error of the best candidate's
CV mean**, which adapts to how variable the data actually is — a noisy dataset
admits a wider tie, a clean one a narrower. `select.tolerance` replaces it with a
fixed band in metric units when a domain has a view.

The critical half: a *meaningfully* better score is never traded for speed. With
a tight CV, a 5-point gap is real and no amount of latency buys it.

**The tie-break order is fixed**: latency leads because it is the constraint that
turns into a user-visible failure; explainability sits below size because it is a
tiered approximation while the first two are measurements. An **unmeasured**
latency sorts *last*, never first — a measurement failure must not look like a
measurement of zero.

The reason string names the axis that actually decided:

```
winner: xgboost — xgboost scores 0.8724 against lightgbm's 0.8750 — within the
0.0103 tolerance (std error of the CV mean) — and wins the tie-break on
p95 latency (1.59 ms vs 5.78 ms)
```

### `objective: weighted`

For a regulated context that needs an auditable weight table.

```yaml
select:
  objective: weighted
  weights:
    performance: 0.50
    latency: 0.20
    cost: 0.15
    explainability: 0.10
    maintainability: 0.05
```

Each criterion is min-max normalized to [0, 1] **across the candidates in this
bake-off**, then combined. Normalizing within the run is what makes 40 ms and
0.91 addable at all — the cost is that the composite has **no meaning outside the
run that produced it**, and comparing two bake-offs' composites is not valid. The
per-criterion normalized values are reported alongside so the arithmetic is
auditable:

```
winner: lightgbm — highest weighted score (0.6000) over 2 eligible candidates:
performance=1.00x0.6, latency=0.00x0.4
```

Weights need not sum to 1; they are normalized. A criterion where every candidate
is identical normalizes to 1.0 for all of them — it carries no information in
this run and must not break the tie on float noise.

**Why `tolerance` is the default.** The weighted rule makes accuracy and
milliseconds commensurable, which they are not, and the winner can shift on a
weight nobody remembers setting. The tolerance rule never trades a real accuracy
difference for speed, and its output is a sentence a reviewer can check.

---

## 6. Running candidates in parallel

Candidates are fully independent, so they parallelize cleanly. Two mechanisms,
for two situations.

### In-process — one machine

```bash
mlf select --config configs/mine.yaml --max-workers 4
```

`concurrent.futures.ProcessPoolExecutor`, one candidate per worker. **Processes,
not threads**: a fit is CPU-bound and holds the GIL for most of its life, and two
Lightning trainers in one interpreter share global state (the seed, the logger,
the accelerator registry) in ways that make results depend on interleaving.

Results are reordered into candidate order after collection, so the table and the
tie-breaks are deterministic regardless of which worker finished first. If a pool
cannot be created (restricted sandbox, frozen executable, notebook without a
`__main__` guard) it **falls back to sequential and says so** — better than
failing a run over a scheduling detail.

Default is `1`. A bake-off on one GPU is not made faster by running four fits on
it at once.

### Optuna trial concurrency

```yaml
tune:
  n_jobs: 4     # threads within one study
```

Threads, not processes — real speedup for GBDT and sklearn (which release the GIL
in `fit`), close to none for a Python-bound loop. Leave at 1 on a single GPU.

### Airflow — across a cluster

`orchestration/airflow/dags/ml_pipeline.py` uses **dynamic task mapping** to fan
out one task per family, then reduces:

```
dvc_pull → spark_preprocess → validate_data
    → tune_candidate.expand([xgboost, lightgbm, catboost, mlp])   ← concurrent
    → collect_winner
    → announce_winner
    → train → evaluate_gate → promote_model → trigger_deploy
```

```python
tune_candidate = BashOperator.partial(task_id="tune_candidate", retries=1).expand(
    bash_command=[
        f"mlf select --config {CONFIG} --candidate {c} --report-dir {REPORT_DIR} {flags}"
        for c in CANDIDATES
    ]
)
collect_winner = BashOperator(
    task_id="collect_winner",
    bash_command=f"mlf select --config {CONFIG} --collect {REPORT_DIR} {flags}",
)
```

Each mapped task evaluates **one** family and writes `<report-dir>/<model>.json`;
it decides nothing. The reduce task reads every report, applies the measured
constraints and runs the decision rule. Reports are plain JSON rather than
pickles: writer and reader are separate processes on separate machines, and a
pickle would couple them to one interpreter and one framework version.

Independent retries and independent failures are the reason this is N tasks
rather than one task with a loop: a family whose extra is missing on one worker
should be one red square, not a dead pipeline.

Configure it with environment variables:

```bash
SELECT_CANDIDATES=xgboost,lightgbm,catboost,mlp
SELECT_REPORT_DIR=outputs/reports
SELECT_MAX_LATENCY_MS=20
SELECT_MAX_MODEL_MB=100
SELECT_MIN_EXPLAINABILITY=0.5
```

`SELECT_CANDIDATES=""` (the default) skips the bake-off entirely and trains the
configured model — the original linear DAG. Re-running a bake-off nightly for a
decision nobody will revisit is just a way to spend GPU hours.

Requires Airflow ≥ 2.3 for `.expand()`.

---

## Configuration reference

### `select`

```yaml
select:
  enabled: false            # off by default: a bake-off costs one budget per candidate
  candidates: []            # [] → every compatible installed family, by auto_priority
  max_candidates: 8
  objective: tolerance      # tolerance | weighted
  tolerance: null           # null → one std error of the CV mean
  max_workers: 1            # candidates evaluated concurrently, as processes
  profile: true             # measure latency / size / explainability
  profile_samples: 128      # rows sampled for the latency benchmark

  constraints:              # all null → unconstrained
    max_latency_p95_ms: null
    max_model_mb: null
    min_explainability: null   # 1.0 native · 0.8 shap · 0.5 permutation
    min_performance: null

  weights:                  # objective: weighted only
    performance: 0.5
    latency: 0.2
    cost: 0.15
    explainability: 0.1
    maintainability: 0.05
```

### `data.split` (P11 additions)

```yaml
data:
  split:
    folds: 5
    cv_strategy: auto       # auto | stratified | kfold | rolling_origin | purged | cpcv

    # purged / cpcv
    label_horizon: 0        # rows a label spans forward
    label_end_col: null     # exact per-row label end times (needs time_col)
    embargo: 0.0            # < 1.0 → fraction of rows; >= 1.0 → row count

    # cpcv
    cpcv_groups: 6
    cpcv_test_groups: 2
    cpcv_max_folds: 20
```

### `tune` (P11 additions)

```yaml
tune:
  objective: holdout        # holdout | cv
  cv_folds: 3               # inner folds for objective: cv
  n_jobs: 1                 # concurrent Optuna trials (threads)
```

---

## CLI reference

```bash
# Compare every compatible family, print the table, decide nothing else
mlf select --config configs/mine.yaml

# With production constraints
mlf select --config configs/mine.yaml \
    --max-latency-ms 20 --max-model-mb 100 --min-explainability 0.5

# A named shortlist, four at a time, saving the winner
mlf select --config configs/mine.yaml \
    --candidates xgboost,lightgbm,catboost,mlp \
    --max-workers 4 --emit-config configs/winner.yaml

# Weighted objective
mlf select --config configs/mine.yaml --objective weighted \
    --set select.weights.performance=0.6 --set select.weights.latency=0.4

# Compare and then train the winner, in one command
mlf train --config configs/mine.yaml --select --max-latency-ms 20

# Orchestrated: fan out, then fan in
mlf select --config c.yaml --candidate xgboost --report-dir reports/
mlf select --config c.yaml --candidate lightgbm --report-dir reports/
mlf select --config c.yaml --collect reports/
```

Every flag maps to exactly one `select.*` config key, so the CLI and the YAML
cannot describe different runs. Flags default to `None`, so "the user asked" is
distinguishable from "nobody said".

Sample output:

```
model                  score     +/-   p95 ms      MB  expl  status
-------------------------------------------------------------------
catboost              0.9025  0.0326   1.2235  0.1019  1.00  WINNER
lightgbm              0.8750  0.0179   2.5612  0.1980  1.00  ok
xgboost               0.8724  0.0187   1.3515  0.1747  1.00  ok
mlp                   0.8700  0.0144   1.0520  1.5621  0.50  rejected: explainability 0.50 (permutation) is below the 0.80 floor
```

ASCII only — a Windows console in cp1252 cannot render a box character, and a
crash in the *reporting* of a successful bake-off is an absurd way to lose one.

---

## Artifacts

| File | Written by | Contains |
|---|---|---|
| `selection.json` | `mlf select`, `mlf train --select` | Every candidate's score, latency, size, explainability, stability; the winner and the reason |
| `<output>/candidates/<model>/feature_importance.json` | per candidate | Method, tier score, per-feature shares, top 10 |
| `<report-dir>/<model>.json` | `mlf select --candidate` | One candidate's report, for the reduce step |
| `manifest.selection` | `train()` | The whole bake-off, inside the served bundle |
| `hpo.json` / `manifest.hpo` | `tune()` | The winner's search space and best params |

`selection.json` records **every** candidate, not just the winner. "We chose
XGBoost" is not an answer to "why not the neural net?" — the table of what each
one scored and cost is, and it is what an architecture review actually wants.

`manifest.selection` puts it inside the bundle, so a *served* model can answer
"why this family?" without the training directory. It is `null` when the model
was named rather than chosen, which is the common case.

---

## Worked examples

### Real-time fraud scoring — a hard latency budget

```yaml
task: binary
data: { kind: tabular, path: data/txn.csv, target: is_fraud, split: { folds: 5 } }
select:
  enabled: true
  max_workers: 4
  constraints:
    max_latency_p95_ms: 20     # the SLA
    min_performance: 0.85      # nothing below this ships
tune:
  objective: cv                # trials scored across folds, not one split
```

Anything that cannot answer in 20 ms is disqualified with its measured p95,
however good its ROC-AUC. Among the rest, the best score wins unless something
statistically tied with it is faster.

### Credit scoring — regulator needs attributions

```yaml
task: binary
select:
  enabled: true
  constraints:
    min_explainability: 0.9    # native importances only
```

Above the permutation tier, the a-priori gate drops models with no
`native_feature_importance` *before* fitting them — a bake-off of trees, and
`feature_importance.json` in the bundle for the audit.

### Financial time series — overlapping labels

```yaml
task: regression
data:
  kind: tabular
  path: data/returns.csv
  target: fwd_return_20d
  split:
    strategy: temporal
    time_col: date
    folds: 6
    cv_strategy: cpcv          # the distribution, not one path
    cpcv_groups: 6
    cpcv_test_groups: 2        # C(6,2) = 15 folds
    label_horizon: 20          # the 20-day forward return
    embargo: 0.01
select: { enabled: true }
```

A 20-day forward return means rows 20 apart share information. Purging removes
the training rows whose labels reach into each test block; the embargo removes
the serially correlated rows just after it. CPCV runs 15 arrangements so the
fold spread is a real distribution rather than one draw — which then feeds the
stability term and the default tolerance.

### Cost-constrained batch scoring

```yaml
select:
  enabled: true
  objective: weighted
  weights: { performance: 0.4, cost: 0.4, maintainability: 0.2 }
  constraints: { max_model_mb: 50 }
```

---

## Cost

A bake-off is **`n_candidates × tuning_budget`**, plus `folds` fits per candidate
for the CV estimate, plus a few hundred forward passes for the profile.

| Setting | Multiplier |
|---|---|
| `select.enabled: true` with N candidates | × N |
| `data.split.folds: k` | + k fits per candidate |
| `tune.objective: cv` with `cv_folds: j` | × j per trial |
| `cv_strategy: cpcv` | fits = C(groups, test_groups), capped by `cpcv_max_folds` |
| `select.max_workers: w` | wall-clock ÷ ~w (CPU-bound, one candidate per core) |

Defaults chosen to make the cost a decision rather than an inheritance:
`select.enabled: false`, `max_candidates: 8`, `tune.objective: holdout`,
`max_workers: 1`.

Rough reference, 400 rows × 6 features, 3 tree families, 3 trials each, on a
laptop CPU: **~15 s** with `--max-workers 3`, ~40 s sequential.

---

## Limitations

Stated rather than papered over.

1. **Latency and size are measured on this machine.** A laptop number is not a
   production number. What transfers is the *ranking* under identical conditions,
   not the absolute value. Set `max_latency_p95_ms` from a measurement on
   hardware resembling production, or treat it as relative.
2. **Forecasting candidates are not latency-profiled.** A forecaster is called
   with a horizon, not with rows, so "milliseconds per row" is not a quantity it
   has. The profile reports "not measured" and the tie-break sorts it last —
   which means a latency constraint disqualifies every forecaster. That is
   correct-but-blunt, and a horizon-based latency measure is the obvious follow-up.
3. **The weighted composite has no cross-run meaning.** Normalization is within
   the bake-off. Comparing two runs' composite scores is not valid.
4. **The profiled estimator is the last fold's**, not a full-data refit. Latency
   and size are properties of the architecture and its hyperparameters, so
   refitting to measure them would double the cost to change the fourth decimal.
5. **CPCV validation folds come off the end of the surviving training block**,
   which is simple and slightly conservative rather than optimal.
6. **`label_end_col` requires `time_col`** and is only readable for
   tabular/timeseries sources. Image and text folds use `label_horizon` or
   nothing.
7. **SHAP is tree-only here.** `shap.Explainer` falls back to a sampling
   explainer for other models, which takes minutes inside a routine that is also
   timing inference. Non-tree models get permutation importance.
8. **No warm-starting between candidates.** Each family's Optuna study starts
   cold. Sharing information across families (a meta-learner over past runs) is
   real AutoML and out of scope here.
9. **Process-pool parallelism is CPU-oriented.** On one GPU, concurrent
   candidates contend and the wall-clock gets worse. Keep `max_workers: 1` there
   and fan out across machines with Airflow instead.

---

## See also

- [`docs/MLOPS.md`](MLOPS.md) — the surrounding stack: MLflow, DVC, Spark,
  Airflow, Kubernetes, Prometheus
- [`docs/PHASE_STATUS.md`](PHASE_STATUS.md) — P11's landing record and open items
- `src/ml_framework/pipeline/select.py` — the driver
- `src/ml_framework/core/profile.py` — the four measurements
- `src/ml_framework/core/explain.py` — the attribution tiers
- `src/ml_framework/data/splitters.py` — every CV strategy
