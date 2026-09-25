# HFI estimation from acoustic signatures

This pipeline selects an acoustic representation, aggregation, dimensionality reduction,
model family, and hyperparameters **inside grouped inner cross-validation**. Outer folds
estimate the performance of that complete selection procedure on unseen Points.

Run commands from the repository root. Install `requirements.txt` in a Python 3.11+
environment; the checked environment uses the existing `venv/bin/python`.
The historical notebooks and their outputs remain exploratory records; their old
configuration and internal APIs are not the entry point for this experiment.

## Data and training contract

`data/schema/schema.json` maps the dataset fields:

- `Point` groups nearby installations. A Point never crosses a training/validation or
  training/test boundary within a fold.
- `CapturePointId` identifies an installation, with one fixed `meanHFI`. Nearby
  installations in the same Point may have different labels.
- Segment feature vectors are averaged within `Audio_Name`. This aggregate is one
  operational one-minute sample, following the project convention, regardless of
  whether one, two, or three segments survived quality control.
- Only recordings within `[RECORDING_START, RECORDING_END)` in the timestamp's local
  time enter the experiment (defaults: 04:00 inclusive, 06:00 exclusive).
- Every training installation contributes **one signature formed from all its
  available recordings**. No sampled submissions enter training. All families use
  equal-installation fitting weights, including PLS and Gaussian processes.
- Validation/test installations contribute repeated random submissions at each
  requested effort. Sampling is uniform over available Audio_Name samples, without
  replacement within a submission and without day quotas. More heavily recorded
  days consequently have greater sampling probability. Across efforts, each
  installation/repetition uses nested prefixes of the same random permutation.
- Validation/test use a common cohort: installations with at least the largest
  selection effort. Installations with fewer recordings still enter training when
  their Point is in a training fold. `data_audit.csv` records eligibility. Performance
  claims therefore apply to that eligible evaluation cohort.

Input features must be finite, embedding lengths consistent, and installation labels
and parent-recording timestamps unambiguous. Invalid data fail explicitly. Feature
extraction and saturation screening are upstream of this repository.

## Protocol and commands

`.env.example` lists every supported parameter. The local `.env` has been migrated to
this format; copy the example for a new installation. Unknown or legacy file settings
raise an error instead of silently changing the protocol. Process environment values
override the dotenv file; CLI arguments override both. Alternative files can be selected
with `--env-file path/to/experiment.env`.

### 1. Freeze and validate the selection experiment

Choose the representations, aggregations, reduction candidates, model families,
selection efforts, trial budget, repetitions and seeds **before inspecting outer
scores**. Then run:

```bash
venv/bin/python main.py --check
venv/bin/python main.py --run-mode nested_selection
```

`--check` validates the data and all grouped split designs, reports cohort size and the
maximum inner-fit budget, and creates no experiment artifacts. At the current default
settings there are 15 outer folds and up to 39,600 inner estimator fits
(`5 × 3 × 11 × 60 × 4`), plus 15 outer refits. This is a substantial study.
Reduce budgets in a separate smoke configuration when checking infrastructure; a smoke
run is not a scientific result. `MODEL_JOBS` limits model/native worker counts. Optuna
trials execute serially, with cached fold preprocessing shared across compatible trials.

For every outer fold:

1. Reserve its Points completely from selection.
2. Create fixed grouped inner folds and random validation submissions. Every candidate
   sees the same folds and sample identities.
3. Run one seeded Optuna TPE study per model family with the same number of trial
   attempts. Each trial jointly searches representation, aggregation, feature scaling,
   reduction and family-specific hyperparameters. Each inner training fold uses all
   its recordings; scaling, PCA and target scaling are fitted on training signatures only.
4. Select the complete configuration with the smallest inner selection score across
   families. Deterministic ties use family name, then earliest trial within that family.
5. Refit that configuration on every outer-training installation's full signature,
   save it, and predict the outer-test submissions. Also evaluate constant mean/median
   baselines fitted from outer-training labels.

The primary selection score is **equal-effort mean of Point-balanced MAE**, measured in
original HFI units. At each effort, absolute errors are averaged over sampled submissions
within each installation, then installations within each Point, then Points. Scores
pool the inner out-of-fold predictions, rather than giving arbitrarily sized folds equal
weight. Each configured effort has equal weight in the final objective. This is a
prespecified application tradeoff, not a claim that every effort is equally frequent
among future users.

Outer out-of-fold metrics use the same weighting. MAE, RMSE, R², bias, median and 90th
percentile absolute error are saved by effort and outer repetition. Repeated submissions
are not treated as independent locations. Outer results describe the **selection
procedure**; do not pick a winning outer fold or retune the search from its test scores.
Adaptive trial rankings and selection frequency are descriptive, not controlled tests
that one representation is intrinsically superior.

### 2. Generate the detailed effort curve without retraining

After completing selection, set `EFFORT_ANALYSIS_COUNTS` and
`EFFORT_ANALYSIS_REPEATS`, then supply the printed selection run directory:

```bash
venv/bin/python main.py --run-mode effort_analysis --source-run output_experiments/<selection-run>
```

This loads the selected fitted pipeline from each outer fold and evaluates new random
submissions from its original held-out installations. It never reruns Optuna or fits
an estimator. No curves are generated for every candidate family. The source run's data,
search settings, seeds and eligibility cohort are authoritative. Current selection
settings in `.env` do not replace that frozen protocol.

Counts can include every integer up to the source run's maximum selection effort.
Extending that maximum requires a new prespecified selection experiment, to avoid
changing the evaluation cohort along the curve. Curves are not forced to decrease.

Outputs include `effort_curve.png`, `effort_curve.pdf`, `effort_summary.csv`, raw
predictions, sample manifests and `incremental_improvement.csv`. Confidence intervals
resample whole Points, keeping paired counts, installations and outer repetitions
together. These are **conditional intervals for average error**, based on fixed fitted
models and submissions; they do not include retraining uncertainty and are not
individual prediction intervals.

`PLATEAU_TOLERANCE=0` disables declaring a plateau. To explore practical saturation,
choose a meaningful tolerance in HFI units beforehand. The optional diagnostic finds
the first count whose upper bootstrap improvement bound is below that tolerance for
each of the next `PLATEAU_WINDOW` evaluated counts. It is exploratory, has no simultaneous
coverage guarantee, and does not establish a permanent plateau beyond observed counts.
Use the raw curve and paired improvements when reporting the result.

For the app, describe the curve as the observed average error on held-out Points under
this sampling protocol. A count alone cannot guarantee the error of an individual place.
Generalization to other microphones, regions, seasons or recording behavior requires
appropriate independent validation.

### 3. Select and fit the deployment artifact using all data

```bash
venv/bin/python main.py --run-mode final_fit --source-run output_experiments/<selection-run>
```

This repeats the same inner selection procedure on the entire development dataset,
then fits its selected configuration on all installations using all eligible-time-window
recordings, including installations below the evaluation effort threshold. It does not
choose a model from outer test scores. The final search adds
`11 × 60 × 4 = 2,640` inner fits at defaults, plus one final fit.

`deployment.joblib` contains the estimator, fitted feature and target transforms,
schema, aggregation configuration and recording window. `deployment_manifest.json`
records its selection and preprocessing. With a trusted locally generated artifact:

```python
import joblib
import pandas as pd

model = joblib.load("output_experiments/<final-run>/deployment.joblib")
segment_features = pd.read_parquet("new_segment_features.parquet")
predictions = model.predict_raw(segment_features)  # no meanHFI column needed
```

The input is a schema-compatible feature table with installation, Point, Audio_Name and
timestamp metadata, not waveform bytes. Use the same upstream feature extraction as
training. `predict_recordings` also accepts already prepared Audio_Name-level features.
Neither inference method fits any transforms. Joblib artifacts require the matching code
and package versions and must only be loaded from trusted sources.

The nested effort curve estimates the selection procedure trained on smaller outer
training sets. It is not direct validation of this particular all-data final artifact;
that requires an independent dataset.

## Search options

`MODELS=all` includes all eleven families. Explicit names are:

```ini
MODELS=RIDGE_REGRESSION,ELASTIC_NET,HUBER,PLS,SVR,KERNEL_RIDGE,GAUSSIAN_PROCESS,RANDOM_FOREST,EXTRA_TREES,XGBOOST,CATBOOST
FEATURE_SETS=indices,embeddings,both
AGGREGATION_STRATEGIES=mean,mean_std,hierarchical
REDUCTION_METHODS=none,pca,pca_variance
PCA_COMPONENT_CANDIDATES=5,10,20,40,60
PCA_VARIANCE_CANDIDATES=0.90,0.95,0.99
FEATURE_SCALINGS=standard,robust,none
```

The example defaults use `none,pca` and standard scaling; add the other choices before
freezing the protocol if desired. `mean` pools recordings; `mean_std` concatenates mean
and population standard deviation. `hierarchical` concatenates the equally weighted
mean of daily means, RMS daily population standard deviations, and population standard
deviation of daily means. Singleton standard deviations are zero; availability indicators
identify unsupported variability components. All statistics for validation/test are
computed only from the sampled recordings.

Indices and embeddings are scaled/reduced separately before concatenation. PCA counts
are filtered against the smallest inner training size and each block's dimensions;
infeasible configurations are pruned, never silently clipped. PLS performs its own
supervised reduction and is always evaluated without preceding PCA. Kernel bandwidths
for SVR/kernel ridge are relative to training feature variance. Optuna directly tunes
the Gaussian-process kernel without a second hidden kernel optimizer.

The versioned ranges are in `src/core/models/search_space.py`. Equal trial budgets
protect each family from being ignored by one global categorical sampler, but are not
equal wall-clock budgets or exhaustive configuration coverage. Failed and pruned trials
consume budget and are retained. Recognized numerical/convergence failures are recorded;
unexpected failures stop the run. A family with no successful trials stops selection
rather than silently disappearing.

## Artifacts and recovery

Each invocation writes a unique directory under `OUTPUT_DIR`. Important selection files:

| Artifact                                                                                   | Purpose                                                                           |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `run_manifest.json`, `resolved_config.json`, `request_config.json`                   | Status, source provenance, input/code hashes, package versions and exact settings |
| `schema.json`, `data_audit.csv`, `outer_splits.parquet`                              | Schema, cohort coverage and grouped split assignments                             |
| `repeat_*/search/studies.sqlite3`, `*_trials.json`, `*_sampler.joblib`               | Persisted Optuna studies, complete trial diagnostics and sampler states           |
| `repeat_*/search/inner_splits.parquet`, `inner_*_submissions.parquet`                  | Reconstructible inner evaluation design                                           |
| `repeat_*/pipeline.joblib`, `submissions.parquet`, `selection.json`                  | Fitted outer pipeline, exact test sample IDs and selected configuration           |
| `oof_predictions.parquet`, `metrics_by_effort.csv`, `selection_procedure_scores.csv` | Held-out results and primary scores                                               |
| `outer_selections.json`, `selection_frequency.csv`                                     | Selection stability across outer folds                                            |

Interrupted selection runs can resume with the original configuration:

```bash
venv/bin/python main.py --resume-run output_experiments/<interrupted-run>
```

For interrupted `final_fit` or `effort_analysis`, supply the same mode and source run as
well. Completed outer folds are reused; remaining search trials continue from persisted
studies. An interrupted running trial is marked failed and consumes an attempt; a crash
between sampler/database writes need not reproduce an uninterrupted random trajectory.
Completed runs are returned without rerunning. Do not run concurrent writers against the
same run directory. Source/resume operations reject changed input data, schema,
implementation, scientific configuration or relevant package versions. Keep the original
checkout/environment available for reproducing an experiment.

## Verification

```bash
venv/bin/python -B -m pytest -q
```

Tests cover aggregation and sampling identities, Point isolation, training-only
transforms, hierarchical metric weighting, all model families, joint Optuna configuration
search, persistence/resume, final inference without labels, and effort analysis without
refitting. Small test runs establish implementation behavior, not predictive performance.

## Start with the readable results report

For an existing completed run, generate the report **without fitting any models**:

```bash
venv/bin/python results.py output_experiments/<run>
```

Open `<run>/reports/report.html` in a browser, or preview
`<run>/reports/summary.md` in VS Code. These are the two main results views. They explain
accuracy, improvement over constant baselines, deployment-selection status, the model
search ranking, selected representations/aggregations, and gains from additional recordings.
The HTML embeds its plots and works offline. Detailed fold comparisons and search health
are expandable; CSV downloads and PNG/PDF figures are linked from the report.

To run the experiment configured in `.env` **and generate its report automatically**:

```bash
venv/bin/python results.py --run-experiment
```

Use `--env-file other.env` for another protocol. The existing `main.py` training entry
point continues to work. `results.py` wraps the same pipeline; it does not introduce a
second training implementation. A reporting failure can be retried against the completed
run directory without repeating the experiment. Reports also support completed
`effort_analysis` and `final_fit` runs. The final-fit report names the actual deployment
model, while nested-selection reports explicitly show that the final decision is pending.

The most useful report tables are:

| File under `reports/` | Question answered |
| --- | --- |
| `experiment_summary.csv` / `.json` | What are the main results and interpretations? |
| `performance_summary.csv` | How accurate is the selected procedure versus baselines? |
| `performance_by_effort.csv`, `recording_effort_gains.csv` | How much does each additional evaluated effort help? |
| `model_search_summary.csv` | Which families performed best during inner tuning, and how often were they selected? |
| `selected_pipelines.csv` | What exactly was selected in each fold, including PCA and hyperparameters? |
| `installation_diagnostics.csv`, `point_errors.csv` | Where are predictions inaccurate or biased? |

Report MAE intervals use paired Point-cluster bootstrap resampling of the fixed held-out
predictions; CV repetitions and random submissions do not count as new independent Points.
They exclude retraining uncertainty and are not per-user prediction intervals. Figures
showing per-installation average predictions are explicitly distinguished from the MAE,
which retains errors of individual submissions. Search rankings are always labeled
`inner_search`; they are not a held-out family leaderboard or controlled representation
ablation. The additional audit tables remain available without dominating the main report.

### Optional fair comparison of every model family

The original experiment predicts outer-test submissions using only each fold's selected
pipeline. To add outer-test scores for **every tuned family on the same submissions**:

```bash
venv/bin/python results.py output_experiments/<nested-selection-run> --compare-models
```

This refits each family's stored inner-selected configuration on the original outer
training data. It reuses the selected family's existing predictions and the exact saved
outer-test submission identities. There is **no new Optuna search**. With 11 families and
15 outer folds, it adds 150 fits, not another 39,600-fit search. It does not run the dense
effort experiment for every family. Completed comparison folds can be reused after an
interruption by rerunning the same command; an incomplete fold is refitted.

The report then includes `outer_model_comparison.csv` and its figure. Treat the ranking
as a descriptive comparison of tuned families. Choosing the best outer-test row would
introduce another selection step; use the prescribed all-data inner search for deployment.
This optional stage validates the original data, package versions and training-code
fingerprint, and refuses incompatible refits or unmatched cohorts. Failing families stop
the comparison rather than disappearing from the leaderboard.

Reporting lives in `reporting/` and `results.py`, outside the original training-code
fingerprint. Report-only changes therefore leave the completed experiment usable for
`effort_analysis` and `final_fit`. Each report records its own code version, source-artifact
hashes, bootstrap settings and source run. Original results and saved models are preserved;
new human-facing outputs are written under `reports/`. `--output-dir` can put them elsewhere.
