"""
train_svm.py -- trains the model used by the Streamlit app (HOG + linear SVM).

Changes vs. the original version:
  * balanced RANDOM sample via common.load_samples (not "first N files")
  * held-out test set is split by GROUP (frame/date), not by individual crop,
    so near-identical crops of the same scene cannot appear in both sets
  * reports precision / recall / F1 / confusion matrix
  * after evaluation the model is refit on ALL sampled data (standard practice
    for the deployed model) and saved to hog_svm_parking.pkl
  * writes results/model_card.json -- the app reads its displayed metrics from
    this file instead of hard-coded numbers

For the full model comparison and cross-validation use evaluate.py.

Usage:
  python train_svm.py --data-dir processed_data --max-per-class 2000
"""
import argparse
import json
import os
import platform
from datetime import datetime

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC

from common import CLASS_NAMES, HOG_PARAMS, SEED, compute_features, load_samples, resolve_groups

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./processed_data")
ap.add_argument("--out", default="./results")
ap.add_argument("--model-path", default="hog_svm_parking.pkl")
ap.add_argument("--max-per-class", type=int, default=2000)
ap.add_argument("--group-by", default="auto", choices=["auto", "frame", "date", "lot"])
ap.add_argument("--seed", type=int, default=SEED)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

print("Loading crops and extracting HOG features. Please wait...")
df = load_samples(a.data_dir, None, a.max_per_class, a.seed)
X = compute_features(df, a.data_dir, cache_path=os.path.join(a.out, "cache", "features.npz"))
y = df["label"].values
groups, group_desc = resolve_groups(df, a.group_by)
print(f"Extracted features from {len(X)} parking spots {df['label_name'].value_counts().to_dict()}")
print(f"Grouping for the held-out split: {group_desc}")

# 1. Group-based 80/20 split: a group never appears on both sides.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed)
train_idx, test_idx = next(gss.split(X, y, groups))
assert not set(groups[train_idx]) & set(groups[test_idx]), "group leakage!"
print(f"Train: {len(train_idx)} crops | Test: {len(test_idx)} crops "
      f"(test class counts: {dict(zip(*np.unique(y[test_idx], return_counts=True)))})")

# 2. Evaluate on the held-out groups
print("\nTraining the SVM model (evaluation fit)...")
eval_model = SVC(kernel="linear", probability=True, random_state=a.seed)
eval_model.fit(X[train_idx], y[train_idx])
pred = eval_model.predict(X[test_idx])
acc = accuracy_score(y[test_idx], pred)
print(f"\nGrouped hold-out accuracy: {acc * 100:.2f}%\n")
print(classification_report(y[test_idx], pred, target_names=CLASS_NAMES, digits=4))
cm = confusion_matrix(y[test_idx], pred, labels=[0, 1])
print("Confusion matrix (rows = true, cols = predicted; order = empty, occupied):\n", cm)

# 3. Refit on all sampled data for deployment and save
print("\nRefitting on all sampled data for the deployed model...")
final = SVC(kernel="linear", probability=True, random_state=a.seed)
final.fit(X, y)
joblib.dump(final, a.model_path)
print(f"Saved model to '{a.model_path}'")

rep = classification_report(y[test_idx], pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
card = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "evaluation": "single grouped hold-out (20% of groups); for cross-validated numbers see evaluate.py",
    "grouping": group_desc, "seed": a.seed,
    "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
    "n_samples_total": int(len(X)), "class_counts": df["label_name"].value_counts().to_dict(),
    "holdout_accuracy": float(acc),
    "holdout_report": {k: v for k, v in rep.items() if k in CLASS_NAMES + ["macro avg"]},
    "confusion_matrix_true_rows_pred_cols": cm.tolist(),
    "hog_params": {k: list(v) if isinstance(v, tuple) else v for k, v in HOG_PARAMS.items()},
    "feature_dim": int(X.shape[1]),
    "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
}
with open(os.path.join(a.out, "model_card.json"), "w") as f:
    json.dump(card, f, indent=2)
print("Wrote results/model_card.json")
