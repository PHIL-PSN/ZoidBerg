"""Build Jupyter/zoidberg2.ipynb from readable Python source.

Run once after editing. Idempotent: rewrites the .ipynb in place.

    python Jupyter/_build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent / "zoidberg2.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "name": "zoidberg2",
        "display_name": "Python (zoidberg2)",
        "language": "python",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
# 0. Title
# ---------------------------------------------------------------------------
md(
    """
# ZoidBerg2.0 — Computer-Aided Diagnosis (Pneumonia from Chest X-rays)

> Given some X-ray images, use machine learning to help doctors detect pneumonia.

This notebook is the technical deliverable for the **ZoidBerg2.0** project.
It walks through the full classical-ML pipeline — load → resize → normalize →
**PCA** → train classical models → evaluate — and applies the **three
evaluation strategies** required by the brief:

1. **Simple train/test split** (baseline reference)
2. **Train / validation / test split** with hyperparameter tuning on the validation set
3. **Stratified K-fold cross-validation**

It also discusses why **accuracy alone is the wrong metric** for medical
imaging and reports **precision, recall, F1 and ROC-AUC**, with confusion
matrices and ROC curves. Finally, the best model is **persisted with
`joblib`** so the test results can be reproduced without retraining.
"""
)

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
md(
    """
## 1. Setup
"""
)

code(
    """
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(context="notebook", style="whitegrid")

print("numpy", np.__version__)
print("pandas", pd.__version__)
import sklearn
print("sklearn", sklearn.__version__)
"""
)

code(
    """
RNG_SEED = 42
np.random.seed(RNG_SEED)

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "Jupyter" else NOTEBOOK_DIR
DATA_ROOT = PROJECT_ROOT / "chest_Xray" / "chest_Xray"
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

IMG_SIZE = 64           # resize to 64x64 grayscale
N_FEATURES = IMG_SIZE * IMG_SIZE
CLASSES = ("NORMAL", "PNEUMONIA")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

assert DATA_ROOT.exists(), f"Dataset not found at {DATA_ROOT}"
print("Project root :", PROJECT_ROOT)
print("Data root    :", DATA_ROOT)
print("Artifacts    :", ARTIFACTS)
print("Image size   :", f"{IMG_SIZE}x{IMG_SIZE} -> {N_FEATURES} features")
"""
)

# ---------------------------------------------------------------------------
# 2. Dataset exploration
# ---------------------------------------------------------------------------
md(
    """
## 2. Dataset exploration

The dataset is the Kaggle *Chest X-Ray Images (Pneumonia)* set, organised as

```
chest_Xray/chest_Xray/
├── train/{NORMAL,PNEUMONIA}/   # large set, used to fit models
├── val/  {NORMAL,PNEUMONIA}/   # tiny held-out set, used to tune hyperparameters
└── test/ {NORMAL,PNEUMONIA}/   # held out until the very end
```
"""
)

code(
    """
def list_images(split: str, cls: str) -> list[Path]:
    folder = DATA_ROOT / split / cls
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})


index = {
    split: {cls: list_images(split, cls) for cls in CLASSES}
    for split in ("train", "val", "test")
}

summary = pd.DataFrame(
    {
        split: {cls: len(paths) for cls, paths in by_class.items()}
        for split, by_class in index.items()
    }
).T
summary["TOTAL"] = summary.sum(axis=1)
summary.loc["TOTAL"] = summary.sum(axis=0)
summary
"""
)

code(
    """
balance = (
    summary.drop(index="TOTAL", columns="TOTAL")
    .reset_index()
    .melt(id_vars="index", var_name="class", value_name="count")
    .rename(columns={"index": "split"})
)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(balance, x="split", y="count", hue="class", ax=ax)
ax.set_title("Class balance per split")
ax.set_xlabel("")
ax.set_ylabel("# images")
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=2, fontsize=9)
plt.tight_layout()
plt.show()
"""
)

code(
    """
def show_grid(paths: Iterable[Path], title: str, cols: int = 4) -> None:
    paths = list(paths)
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.array(axes).reshape(-1)
    for ax, p in zip(axes, paths):
        ax.imshow(Image.open(p).convert("L"), cmap="gray")
        ax.set_title(p.parent.name, fontsize=9)
        ax.axis("off")
    for ax in axes[len(paths):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


rng = np.random.default_rng(RNG_SEED)
sample_normal = rng.choice(index["train"]["NORMAL"], size=4, replace=False)
sample_pneumo = rng.choice(index["train"]["PNEUMONIA"], size=4, replace=False)
show_grid(list(sample_normal) + list(sample_pneumo), "Random samples from train/")
"""
)

# ---------------------------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------------------------
md(
    """
## 3. Preprocessing pipeline

Each X-ray is read in grayscale, resized to `IMG_SIZE × IMG_SIZE`, normalised
to `[0, 1]` and flattened to a 1-D feature vector of length `N_FEATURES`.

The resulting NumPy arrays are cached to `artifacts/features_<split>.npz` so
the notebook does not have to re-decode every JPEG on each run.
"""
)

code(
    """
def load_image(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0


def load_split(split: str, size: int = IMG_SIZE, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    cache = ARTIFACTS / f"features_{split}_{size}.npz"
    if cache.exists() and not force:
        data = np.load(cache)
        return data["X"], data["y"]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for cls in CLASSES:
        for path in tqdm(index[split][cls], desc=f"{split}/{cls}", leave=False):
            xs.append(load_image(path, size))
            ys.append(CLASS_TO_IDX[cls])
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    np.savez_compressed(cache, X=X, y=y)
    return X, y
"""
)

code(
    """
t0 = time.perf_counter()
X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")
X_test,  y_test  = load_split("test")
print(f"Loaded everything in {time.perf_counter() - t0:.1f}s")

shapes = pd.DataFrame(
    {
        "X.shape": [X_train.shape, X_val.shape, X_test.shape],
        "NORMAL": [(y == 0).sum() for y in (y_train, y_val, y_test)],
        "PNEUMONIA": [(y == 1).sum() for y in (y_train, y_val, y_test)],
    },
    index=["train", "val", "test"],
)
shapes
"""
)

# ---------------------------------------------------------------------------
# 4. Class balance & why accuracy is wrong
# ---------------------------------------------------------------------------
md(
    """
## 4. Class balance & why **accuracy alone is wrong**

The training set is heavily imbalanced — there are roughly **3× more pneumonia
cases than normal**. A trivial classifier that always answers *"PNEUMONIA"*
would already score ≈ 74 % accuracy on train and ≈ 62 % on test, while being
medically useless.

In medical diagnosis the asymmetry of errors matters:

- A **false negative** (missing a sick patient) is dangerous.
- A **false positive** (flagging a healthy patient) is annoying but recoverable.

Therefore we will optimise for **recall on the PNEUMONIA class** and report
**ROC-AUC**, which is threshold-independent and robust to class imbalance.
"""
)

code(
    """
dummy = DummyClassifier(strategy="most_frequent", random_state=RNG_SEED)
dummy.fit(X_train, y_train)
print(f"Dummy 'always-PNEUMONIA' accuracy on test : {dummy.score(X_test, y_test):.3f}")
print(f"...but recall on NORMAL                    : {recall_score(y_test, dummy.predict(X_test), pos_label=0):.3f}")
print(f"...and  ROC-AUC                            : {roc_auc_score(y_test, dummy.predict(X_test)):.3f}")
"""
)

# ---------------------------------------------------------------------------
# 5. PCA
# ---------------------------------------------------------------------------
md(
    """
## 5. Feature engineering — PCA

A 64×64 image is a vector in ℝ^4096, but neighbouring pixels are extremely
correlated. **Principal Component Analysis** (PCA) projects the data onto the
directions of maximum variance, dramatically reducing the dimensionality
without losing much signal — which speeds up *every* downstream estimator.

> **Critical rule:** PCA is fitted on the **training data only**. Applying
> `fit_transform` to the test set would leak information about the test
> distribution into the training procedure.
"""
)

code(
    """
pca_probe = PCA(n_components=200, random_state=RNG_SEED).fit(X_train)
cum = np.cumsum(pca_probe.explained_variance_ratio_)

target = 0.95
n95 = int(np.argmax(cum >= target)) + 1

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(cum) + 1), cum, lw=2)
ax.axhline(target, ls="--", color="grey")
ax.axvline(n95, ls="--", color="crimson")
ax.set_xlabel("# components")
ax.set_ylabel("cumulative explained variance")
ax.set_title(f"PCA — {n95} components explain {target:.0%} of the variance")
plt.tight_layout()
plt.show()

N_PCA = max(50, n95)
print(f"Using N_PCA = {N_PCA}")
"""
)

code(
    """
def make_pipeline(estimator, n_components: int = N_PCA) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=n_components, random_state=RNG_SEED)),
            ("clf", estimator),
        ]
    )
"""
)

# ---------------------------------------------------------------------------
# 6. Strategy A — simple train/test split
# ---------------------------------------------------------------------------
md(
    """
## 6. Strategy A — Simple train/test split

We pool every image, do a single stratified 80/20 split, fit a baseline
**Logistic Regression** on it, and report the metrics. This is the *fastest
but least reliable* protocol — the score depends on which samples ended up in
the test fold.
"""
)

code(
    """
X_all = np.concatenate([X_train, X_val, X_test], axis=0)
y_all = np.concatenate([y_train, y_val, y_test], axis=0)

X_a_tr, X_a_te, y_a_tr, y_a_te = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=RNG_SEED
)

pipe_simple = make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG_SEED))
pipe_simple.fit(X_a_tr, y_a_tr)

proba = pipe_simple.predict_proba(X_a_te)[:, 1]
pred = (proba >= 0.5).astype(int)

simple_metrics = {
    "accuracy":  float((pred == y_a_te).mean()),
    "precision": float(precision_score(y_a_te, pred)),
    "recall":    float(recall_score(y_a_te, pred)),
    "f1":        float(f1_score(y_a_te, pred)),
    "roc_auc":   float(roc_auc_score(y_a_te, proba)),
}
pd.Series(simple_metrics, name="A — simple split").round(3).to_frame()
"""
)

# ---------------------------------------------------------------------------
# 7. Strategy B — train / val / test
# ---------------------------------------------------------------------------
md(
    """
## 7. Strategy B — Train / Validation / Test split

The dataset already ships with the canonical split. We:

1. Fit several estimators on **train**.
2. For each estimator, search over a small hyperparameter grid and pick the
   configuration that maximises **ROC-AUC on the validation set**.
3. The winner is then retrained on **train ∪ val** and evaluated **once** on
   **test** (cell 9).

The validation set in this dataset is tiny (16 images) — we keep it that way
on purpose to mirror the project brief, but we are aware the validation
metric will be noisy.
"""
)

code(
    """
@dataclass
class Candidate:
    name: str
    estimator: object
    param_name: str   # name of the hyperparameter that varies, e.g. "clf__C"
    param_grid: list

CANDIDATES = [
    Candidate(
        name="LogReg",
        estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG_SEED),
        param_name="clf__C",
        param_grid=[0.1, 1.0, 10.0],
    ),
    Candidate(
        name="SVM-RBF",
        estimator=SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG_SEED),
        param_name="clf__C",
        param_grid=[1.0, 5.0, 10.0],
    ),
    Candidate(
        name="RandomForest",
        estimator=RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RNG_SEED, n_jobs=-1),
        param_name="clf__max_depth",
        param_grid=[None, 8, 16],
    ),
    Candidate(
        name="KNN",
        estimator=KNeighborsClassifier(n_jobs=-1),
        param_name="clf__n_neighbors",
        param_grid=[3, 5, 11],
    ),
]


def evaluate(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    else:
        proba = pipe.decision_function(X)
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy":  float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y, proba)),
    }
"""
)

code(
    """
results: list[dict] = []
trained: dict[str, Pipeline] = {}

for cand in CANDIDATES:
    best_auc = -np.inf
    best_pipe = None
    best_param = None
    for value in cand.param_grid:
        pipe = make_pipeline(cand.estimator)
        pipe.set_params(**{cand.param_name: value})
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        m = evaluate(pipe, X_val, y_val)
        results.append({"model": cand.name, cand.param_name: value, "fit_s": round(fit_s, 1), **m, "split": "val"})
        if m["roc_auc"] > best_auc:
            best_auc, best_pipe, best_param = m["roc_auc"], pipe, value
    print(f"  best {cand.name:>12s}  {cand.param_name}={best_param}  val AUC={best_auc:.3f}")
    trained[cand.name] = best_pipe

results_df = pd.DataFrame(results).round(3)
results_df
"""
)

# ---------------------------------------------------------------------------
# 8. Strategy C — Stratified K-fold CV
# ---------------------------------------------------------------------------
md(
    """
## 8. Strategy C — Stratified K-fold cross-validation

K-fold CV gives a more stable estimate of generalisation than a single
train/val split. We run **stratified 5-fold CV on the training set only** and
score with **ROC-AUC**.
"""
)

code(
    """
K = 5
cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=RNG_SEED)

cv_rows: list[dict] = []
for cand in CANDIDATES:
    pipe = make_pipeline(cand.estimator)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_rows.append(
        {"model": cand.name, "mean_auc": scores.mean(), "std_auc": scores.std(), "folds": list(np.round(scores, 3))}
    )

cv_df = pd.DataFrame(cv_rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
cv_df.round(3)
"""
)

code(
    """
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(cv_df["model"], cv_df["mean_auc"], yerr=cv_df["std_auc"], fmt="o", capsize=5, lw=2)
ax.set_ylabel(f"ROC-AUC (mean ± std over {K} folds)")
ax.set_title(f"Strategy C — Stratified {K}-fold CV on train/")
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.show()
"""
)

# ---------------------------------------------------------------------------
# 9. Final evaluation
# ---------------------------------------------------------------------------
md(
    """
## 9. Final evaluation on the held-out test set

We pick the model with the best **validation ROC-AUC** (Strategy B), retrain
it on `train ∪ val`, and run it **once** on the test set. This is the score
that goes into the synthesis report.
"""
)

code(
    """
val_metrics = (
    pd.DataFrame([
        {"model": cand.name, **evaluate(trained[cand.name], X_val, y_val)}
        for cand in CANDIDATES
    ])
    .sort_values("roc_auc", ascending=False)
    .reset_index(drop=True)
)
val_metrics.round(3)
"""
)

code(
    """
best_name = val_metrics.iloc[0]["model"]
best_pipe = trained[best_name]
print(f"Best model on validation: {best_name}")

X_trval = np.concatenate([X_train, X_val], axis=0)
y_trval = np.concatenate([y_train, y_val], axis=0)
final_pipe = Pipeline(best_pipe.steps).set_params(**{
    k: v for k, v in best_pipe.get_params().items() if k.startswith("clf__") or k.startswith("pca__")
})
final_pipe.fit(X_trval, y_trval)

test_proba = final_pipe.predict_proba(X_test)[:, 1]
test_pred  = (test_proba >= 0.5).astype(int)
final_metrics = {
    "accuracy":  float((test_pred == y_test).mean()),
    "precision": float(precision_score(y_test, test_pred)),
    "recall":    float(recall_score(y_test, test_pred)),
    "f1":        float(f1_score(y_test, test_pred)),
    "roc_auc":   float(roc_auc_score(y_test, test_proba)),
}
pd.Series(final_metrics, name=f"{best_name} — TEST").round(3).to_frame()
"""
)

code(
    """
print(classification_report(y_test, test_pred, target_names=list(CLASSES), digits=3))
"""
)

code(
    """
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, test_pred),
    display_labels=list(CLASSES),
).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion matrix (counts)")

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, test_pred, normalize="true"),
    display_labels=list(CLASSES),
).plot(ax=axes[1], colorbar=False, cmap="Blues", values_format=".2f")
axes[1].set_title("Confusion matrix (row-normalised)")
plt.tight_layout()
plt.show()
"""
)

code(
    """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, test_proba, name=best_name, ax=axes[0])
axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
axes[0].set_title("ROC curve — test set")

PrecisionRecallDisplay.from_predictions(y_test, test_proba, name=best_name, ax=axes[1])
axes[1].set_title("Precision–Recall curve — test set")
plt.tight_layout()
plt.show()
"""
)

# ---------------------------------------------------------------------------
# 10. Cross-strategy summary
# ---------------------------------------------------------------------------
md(
    """
## 10. Cross-strategy summary

The same pipeline evaluated three different ways. The K-fold mean is the most
trustworthy estimate; the simple split is the noisiest.
"""
)

code(
    """
summary_df = pd.DataFrame(
    {
        "A — simple train/test split":       simple_metrics,
        f"B — train/val/test ({best_name})": final_metrics,
        f"C — {K}-fold CV (mean)":           {"roc_auc": cv_df.iloc[0]["mean_auc"]},
    }
).T
summary_df.round(3)
"""
)

# ---------------------------------------------------------------------------
# 11. Persisting the model
# ---------------------------------------------------------------------------
md(
    """
## 11. Persisting the model with `joblib`

Saving the fitted `Pipeline` keeps every preprocessing step (`StandardScaler`,
`PCA`) bundled with the classifier, so we can reproduce the test results
without retraining.
"""
)

code(
    """
model_path = ARTIFACTS / f"zoidberg2_{best_name.lower()}.joblib"
joblib.dump(
    {
        "model": final_pipe,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "n_pca": N_PCA,
        "metrics_test": final_metrics,
    },
    model_path,
)
print("Saved", model_path, f"({model_path.stat().st_size / 1024:.0f} KB)")

reloaded = joblib.load(model_path)
proba_check = reloaded["model"].predict_proba(X_test[:5])[:, 1]
print("Reloaded probabilities (first 5):", np.round(proba_check, 3))
"""
)

# ---------------------------------------------------------------------------
# 12. Conclusion
# ---------------------------------------------------------------------------
md(
    """
## 12. Conclusion & next steps

- **Why these metrics?** In medical diagnosis a missed pneumonia (false
  negative) is far worse than a false alarm. We therefore tracked **recall**
  on the `PNEUMONIA` class as the primary safety metric and **ROC-AUC** as a
  threshold-independent ranking metric.
- **Why three evaluation strategies?** The simple split is fast but noisy.
  Train/val/test mirrors a realistic deployment (tune on val, freeze test).
  K-fold CV averages out fold-specific luck and is the most honest single
  number to put in the report.
- **Why PCA?** It compresses 4096-dimensional pixel vectors to a few dozen
  components that already capture > 95 % of the variance, which makes every
  classifier faster and reduces overfitting.

### Bonus ideas (per the brief)

- 3-class extension: `NORMAL` vs `BACTERIAL` vs `VIRAL` — the filenames in
  `train/PNEUMONIA/` already encode this.
- Self-organising map for visualising the PCA-reduced space.
- A small CNN (PyTorch / Keras) — typically pushes ROC-AUC > 0.97 on this
  dataset.
"""
)


nb["cells"] = cells
NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote {NB_PATH} with {len(cells)} cells")
