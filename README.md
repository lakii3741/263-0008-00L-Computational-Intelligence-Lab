# CIL Project 2026

Collaborative filtering Project Repository for Group "Spring Rolls" for the ETH CIL 2026 paper recommendation task.

## Quick Setup

This project uses `uv` for dependency management. 

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already.
2. Clone this repository and run the following command to install dependencies:
```bash
uv sync
```
3. Ensure your dataset files (`sample_submission.csv`, `train_ratings.csv`, `train_tbr.csv`) are placed in the `data/` directory.

## Reproducing Report Results

To reproduce the exact experiments, benchmarks, and ablations from our report, run the following scripts. Note that outputs will be saved to the `results/` directory, which our plotting notebooks consume.

**1. Baseline Benchmarks**
Run the cross-validation and holdout pipeline for all baseline methods (or specify models via `--models`):
```bash
uv run python scripts/benchmark_holdout.py
```

**2. Main Model (Ensemble Pipeline)**
Train the main 47-base-member ensemble and evaluate on the holdout split:
```bash
uv run python scripts/benchmark_ensemble/benchmark_full_pipeline.py
```

**3. Ablation Studies**
Run the individual ablation scripts to analyze the contribution of individual components in our ensemble:
```bash
uv run python scripts/ablations/without_isotonic.py
# Run other scripts in scripts/ablations/ similarly...
```

**4. Generate Figures and Tables**
Start Jupyter and run the notebooks in the `notebooks/` directory (e.g., `Fig2_ablation_metalearners.ipynb`, `Tab3_ablation_base_predictors.ipynb`) to dynamically generate the plots and tables used in the report. Figures are saved to `notebooks/figures/`:
```bash
uv run jupyter notebook
```
*(Alternatively, simply open and execute them directly inside VS Code.)*

## Directory Overview

* **`config/`**: YAML configuration files for defining hyperparameter search spaces during baseline grid search (e.g., `als.yaml`, `svd.yaml`).
* **`notebooks/`**: Jupyter notebooks for visualizing data, plotting figures, and laying out ablation tables for the report.
* **`scripts/`**: Executable scripts for running different parts of the pipeline:
  * `benchmark_holdout.py`: Baseline grid-search and evaluation.
  * `benchmark_ensemble/`: Scripts for evaluating the full ensembled model pipeline.
  * `ablations/`: Scripts that test model design choices by disabling specific components.
  * `kaggle/`: Scripts dedicated to generating predictions on the blind test set for Kaggle submission.
* **`results/`**: Model prediction outputs, scored JSONs, Pickles, and ablation logging used by the notebooks.
* **`cil/` & `cf/`**: Core library modules containing data loaders, normalizers, baseline predictors, and ensembling logic.

---

## Extended Documentation & Development
*Note: `Evaluator.cross_validate` was our first CV procedure, now only kept for demos. For benchmarking and reported results, we used the cross-validation + held-out pipeline described below.*
### Usage Example

```python
from cil import DataLoader, Evaluator, IdentityNormalizer
from cil.predictors.baselines import SVDPredictor

loader = DataLoader("data")

# Cross-validate
Evaluator.cross_validate(
    loader,
    normalizer_factory=IdentityNormalizer,
    predictor_factory=lambda: SVDPredictor(k=2),
    n_folds=5,
)

# Fit on full data and generate submission
normalizer = IdentityNormalizer()
train_norm = normalizer.fit_transform(loader.ratings)

predictor = SVDPredictor(k=2)
predictor.fit(train_norm, loader.num_scientists, loader.num_papers, wishlist=loader.wishlist)

Evaluator(normalizer).make_submission(predictor, loader, "submission.csv")
```

Run the demo: `uv run python scripts/e2e_demo.py`

### Data flow

```
Evaluator.cross_validate(loader, normalizer_factory, predictor_factory, n_folds)
  For each fold:
    DataLoader.kfold_split() -> train_df, val_df  (coverage-preserving)
            |
    Normalizer.fit_transform(train_df) -> train_norm
            |
    Predictor.fit(train_norm, ...)
            |
    Evaluator.evaluate(predictor, val_df)
      1. predictor.predict(sids, pids)    -> normalized predictions
      2. normalizer.inverse_transform()   -> original-scale predictions
      3. clip to [1, 5]
      4. RMSE vs val_df["rating"]
  -> CVResult (per-fold RMSEs, mean RMSE, std RMSE)
```

`val_df` is never normalized. The evaluator inverse-transforms predictions before comparing.

`DataLoader.split()` is still available for one-off train/val splits.

### Available predictors

| Predictor | Module | Description |
|-----------|--------|-------------|
| `SVDPredictor` | `cil.predictors.baselines` | Truncated SVD (rank-k approximation) |
| `IterativeSVDPredictor` | `cil.predictors.baselines` | Iterative SVD imputation |
| `BiasedSVDPredictor` | `cil.predictors.baselines` | Matrix factorization with user and item biases |
| `ALSPredictor` | `cil.predictors.baselines` | Alternating Least Squares (Explicit Feedback) |
| `SVDppPredictor` | `cil.predictors.baselines` | Numba-accelerated SVD++ with implicit feedback |
| `NeuMFPredictor` | `cil.predictors.baselines` | Neural Matrix Factorization |
| `DMFPredictor` | `cil.predictors.baselines` | Deep Matrix Factorization |
| `SlopeOnePredictor` | `cil.predictors.baselines` | Slope One item-based collaborative filtering |
| `KNNBasicPredictor` | `cil.predictors.baselines` | Basic k-NN collaborative filtering |
| `KNNWithMeansPredictor` | `cil.predictors.baselines` | k-NN CF with mean-centering |
| `KNNWithZScorePredictor` | `cil.predictors.baselines` | k-NN CF with Z-Score normalization |
| `GlobalMeanPredictor` | `cil.predictors.baselines` | Predicts the global mean rating across all items |
| `UserMeanPredictor` | `cil.predictors.baselines` | Predicts the historical mean rating of the user |
| `ItemMeanPredictor` | `cil.predictors.baselines` | Predicts the historical mean rating of the item |
| `ItemKNNPredictor` | `cil.predictors` | Item-based KNN with shrunk cosine similarity |
| `EnsemblePredictor` | `cil.predictors` | Uniform average over (predictor, normalizer) pairs |
| `IALSPredictor` | `cil.predictors.implicit` | Implicit Alternating Least Squares |
| `BPRPredictor` | `cil.predictors.implicit` | Bayesian Personalized Ranking |
| `EASEBinaryPredictor` | `cil.predictors.implicit` | Embarrassingly Shallow Autoencoders for Sparse Data |
| `ItemPopPredictor` | `cil.predictors.implicit` | Popularity-based implicit baseline |

All are re-exported from `cil` or `cil.predictors`.

### Adding a new predictor

Create a file in `cil/predictors/` (or a subfolder like `baselines/`):

```python
# cil/predictors/my_predictor.py
from cil.predictors.base import Predictor

class MyPredictor(Predictor):
    def fit(self, train_df, num_scientists, num_papers, wishlist=None):
        ...

    def predict(self, sids, pids):
        ...  # return predictions in normalized scale
```

Re-export in the relevant `__init__.py`.

### Ensembling predictors

`EnsemblePredictor` averages predictions from multiple (predictor, normalizer)
pairs. Each member is fitted and inverse-transformed independently, so members
can use different normalizers. Pair the ensemble with `IdentityNormalizer` in
the `Evaluator` since de-normalization is handled internally.

```python
from cil import DataLoader, EnsemblePredictor, Evaluator, IdentityNormalizer
from cil.normalizers.baselines import ScientistNormalizer
from cil.predictors.baselines import ALSPredictor, SVDppPredictor

loader = DataLoader("data")

ensemble = EnsemblePredictor([
    (SVDppPredictor(k=25, n_epochs=25, seed=42), IdentityNormalizer()),
    (SVDppPredictor(k=25, n_epochs=25, seed=123), ScientistNormalizer()),
    (ALSPredictor(k=20, reg=0.1, n_iters=15), IdentityNormalizer()),
])
ensemble.fit(loader.ratings, loader.num_scientists, loader.num_papers)

Evaluator(IdentityNormalizer()).make_submission(ensemble, loader, "submission.csv")
```

### Benchmarking and parameter sweeps

`benchmark` cross-validates named (predictor, normalizer) configs and prints a
sorted table with mean and std RMSE:

```python
from cil import DataLoader, IdentityNormalizer, benchmark
from cil.predictors.baselines import SVDPredictor, ALSPredictor

loader = DataLoader("data")

benchmark(loader, {
    "SVD k=10": (lambda: SVDPredictor(k=10), IdentityNormalizer),
    "ALS k=10": (lambda: ALSPredictor(k=10, reg=0.1, n_iters=10), IdentityNormalizer),
})
```

`param_sweep` does a grid search over hyperparameters for a single predictor class:

```python
from cil import DataLoader, IdentityNormalizer, param_sweep
from cil.predictors.baselines import SVDppPredictor

param_sweep(
    DataLoader("data"),
    SVDppPredictor,
    param_grid={"k": [10, 20, 30], "reg": [0.01, 0.05]},
    normalizer_factory=IdentityNormalizer,
    fixed_params={"n_epochs": 10},
)
```

#### Cross-validation + held-out benchmark

Fair benchmarking with hyperparameter tuning:

1. Split the data once into train (80%) and holdout (20%) set.
2. Run a grid search with 5-fold CV *on the train set only* to pick the best config per model (i.e. with lowest mean RMSE across folds).
3. Refit the best config on the full train set.
4. Score it once on the untouched holdout set.

Report the holdout RMSE and MAE. Out-of-fold predictions from the winning config are saved alongside the holdout predictions.

`scripts/benchmark_holdout.py` runs this pipeline for one or more models in a
single invocation, pickles each model's result, and prints a comparison table
sorted by holdout score on the chosen metric.

Each model reads `config/<model>.yaml`:

```yaml
normalizer: identity           # identity | paper | scientist (default: identity).
                               # Pass a list (e.g. [identity, paper, scientist]) to
                               # search over normalizers as an extra grid axis.
search_space:                  # cartesian product; may be empty or omitted
  k: [10, 20, 40]
  reg: [0.02, 0.05]
fixed_params:                  # held constant across every grid point
  n_epochs: 20
  seed: 42
```

### Adding a new normalizer

```python
# cil/normalizers/my_normalizer.py
from cil.normalizers.base import Normalizer

class MyNormalizer(Normalizer):
    def fit(self, train_df):
        ...  # compute statistics from training data

    def transform(self, df):
        ...  # return new DataFrame with normalized ratings (don't mutate input)

    def inverse_transform(self, sids, pids, predictions):
        ...  # reverse the normalization
```

### Tests

```bash
uv run python tests/test_normalizers.py
```

### Slurm
To run something on the student-cluster, execute the following
```bash
sbatch scripts/submit_job.sh scripts/\<YOUR_SCRIPT_NAME\>.py
```

To check node GPU usage:
```bash
srun --jobid=\<JOB_ID\> --overlap --pty bash
watch -n 1 nvidia-smi
```
---

## AI Usage Declaration
The use of generative AI tools is documented below:

**Tool used:** Claude Sonnet 4.6\
**Files affected:** Plotting notebooks in `notebooks/`\
**Purpose:** Generating plotting code for the figures in the report.

**Tool used:** Claude Sonnet 4.6\
**Files affected:** All files in `cil/predictors/implicit/`\
**Purpose:** Helped write wrapper code around ranking models from the `LensKit` library (used for wishlist ablation study).

**Tool used:** Claude Sonnet 4.6\
**Files affected:** All files in `cf/`\
**Purpose:** Helped with implementation support, refactoring, documentation, and code-quality improvements.

**Tool used:** Gemini 3.1 Pro\
**Files affected:** `cil/predictors/baselines/neumf` and `cil/predictors/baselines/dmf`\
**Purpose:** Helped to correctly implement the baselines of the DL papers we compare to.

**Tool used:** Claude Sonnet 4.6\
**Files affected:** Infrastructure code in `cil/`\
**Purpose:** Reviewed code for Python best practices and recommend improvements
