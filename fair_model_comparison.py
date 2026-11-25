
"""Fair comparison of HOG classical models and the saved CNN.

All models use the same five outer date-grouped test folds.
Classical models train on the exact inner training partitions used by the CNN.
The CNN predictions are loaded from its existing OOF prediction file.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.svm import SVC

from common import CLASS_NAMES, compute_features


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed_data"
METADATA_PATH = DATA_DIR / "metadata.csv"

CNN_OOF_PATH = (
    ROOT / "outputs" / "cnn_date_cv" / "oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "fair_model_comparison"

CACHE_PATH = OUTPUT_DIR / "cache" / "hog_features.npz"

N_SPLITS = 5
RANDOM_SEED = 42
INNER_SPLIT_ATTEMPTS = 40
INNER_VALIDATION_SIZE = 0.20

CLASS_LABELS = [0, 1]


# ============================================================
# LOAD DATA - MATCH train_cnn.py
# ============================================================

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = [
        "crop_path",
        "label",
        "date",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing metadata column: {column}"
            )

    df = df.copy()

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["label", "date", "crop_path"]
    ).reset_index(drop=True)

    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Expected labels 0 and 1 only."
        )

    df["date"] = df["date"].astype(str)

    df["image_path"] = df["crop_path"].apply(
        lambda p: DATA_DIR / str(p)
    )

    missing = [
        str(path)
        for path in df["image_path"]
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing. "
            f"First missing image: {missing[0]}"
        )

    y = df["label"].to_numpy(dtype=np.int64)
    groups = df["date"].to_numpy()

    if len(np.unique(groups)) < N_SPLITS:
        raise ValueError(
            "Not enough unique dates for 5-fold CV."
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Dataset must contain both classes."
        )

    if df["crop_path"].astype(str).duplicated().any():
        raise ValueError(
            "Duplicate crop_path values found. "
            "Cannot safely match CNN OOF predictions."
        )

    print("\nDATASET SUMMARY")
    print("=" * 55)
    print("Samples:", len(df))
    print("Empty:", int(np.sum(y == 0)))
    print("Occupied:", int(np.sum(y == 1)))
    print("Unique dates:", len(np.unique(groups)))
    print("Missing images:", len(missing))

    return df, y, groups


# ============================================================
# INNER SPLIT - EXACTLY MATCH train_cnn.py
# ============================================================

def create_inner_split(indices, y, groups, seed):

    X_dummy = np.zeros(len(indices))

    splitter = GroupShuffleSplit(
        n_splits=INNER_SPLIT_ATTEMPTS,
        test_size=INNER_VALIDATION_SIZE,
        random_state=seed
    )

    for train_local, val_local in splitter.split(
        X_dummy,
        y[indices],
        groups[indices]
    ):

        train_indices = indices[train_local]
        val_indices = indices[val_local]

        train_groups = set(groups[train_indices])
        val_groups = set(groups[val_indices])

        if train_groups.intersection(val_groups):
            continue

        if len(np.unique(y[train_indices])) != 2:
            continue

        if len(np.unique(y[val_indices])) != 2:
            continue

        return train_indices, val_indices

    raise RuntimeError(
        "Could not reproduce a valid inner split."
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def make_models():

    return {
        "HOG + Linear SVM": SVC(
            kernel="linear",
            C=1.0,
            random_state=RANDOM_SEED
        ),

        "HOG + Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "HOG + Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            zero_division=0
        )
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "macro_precision": float(
            np.mean(precision)
        ),

        "macro_recall": float(
            np.mean(recall)
        ),

        "macro_f1": float(
            np.mean(f1)
        ),

        "empty_precision": float(precision[0]),
        "empty_recall": float(recall[0]),
        "empty_f1": float(f1[0]),

        "occupied_precision": float(precision[1]),
        "occupied_recall": float(recall[1]),
        "occupied_f1": float(f1[1]),
    }


# ============================================================
# VERIFY CNN PREDICTIONS AGAINST RECREATED FOLDS
# ============================================================

def load_and_validate_cnn_predictions(
    df,
    y,
    outer_splits
):

    if not CNN_OOF_PATH.exists():
        raise FileNotFoundError(
            f"CNN predictions not found: {CNN_OOF_PATH}"
        )

    cnn = pd.read_csv(CNN_OOF_PATH)

    required = {
        "crop_path",
        "date",
        "fold",
        "true_label",
        "predicted_label",
    }

    missing = required - set(cnn.columns)

    if missing:
        raise ValueError(
            f"CNN OOF file is missing columns: {sorted(missing)}"
        )

    cnn["crop_path"] = cnn["crop_path"].astype(str)
    cnn["date"] = cnn["date"].astype(str)
    cnn["fold"] = cnn["fold"].astype(int)
    cnn["true_label"] = cnn["true_label"].astype(int)
    cnn["predicted_label"] = cnn["predicted_label"].astype(int)

    if cnn["crop_path"].duplicated().any():
        raise ValueError(
            "CNN OOF file contains duplicate crop paths."
        )

    if len(cnn) != len(df):
        raise ValueError(
            f"CNN OOF sample count ({len(cnn)}) "
            f"does not match metadata ({len(df)})."
        )

    expected_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_splits,
        start=1
    ):

        for idx in test_idx:

            expected_rows.append({
                "crop_path": str(df.iloc[idx]["crop_path"]),
                "date": str(df.iloc[idx]["date"]),
                "fold": fold,
                "true_label": int(y[idx]),
            })

    expected = pd.DataFrame(expected_rows)

    check = expected.merge(
        cnn[
            [
                "crop_path",
                "date",
                "fold",
                "true_label",
            ]
        ],
        on="crop_path",
        how="outer",
        suffixes=("_expected", "_cnn"),
        indicator=True,
        validate="one_to_one"
    )

    if not (check["_merge"] == "both").all():
        raise RuntimeError(
            "CNN predictions do not contain exactly "
            "the expected dataset samples."
        )

    for column in ["date", "fold", "true_label"]:

        if not (
            check[f"{column}_expected"]
            == check[f"{column}_cnn"]
        ).all():

            raise RuntimeError(
                f"CNN {column} values do not match "
                "the recreated folds."
            )

    print("\nCNN FOLD VALIDATION")
    print("=" * 55)
    print("CNN OOF samples:", len(cnn))
    print("Recreated samples:", len(expected))
    print("Fold assignments: MATCH")
    print("Dates: MATCH")
    print("True labels: MATCH")

    return cnn


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def confusion_rows_for_model(
    model_name,
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS
    )

    rows = []

    for i, true_name in enumerate(CLASS_NAMES):

        for j, predicted_name in enumerate(CLASS_NAMES):

            rows.append({
                "model": model_name,
                "true_class": true_name,
                "predicted_class": predicted_name,
                "count": int(cm[i, j]),
            })

    return rows


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load the exact dataset used by the CNN.
    df, y, groups = load_metadata()

    # Recreate the exact outer CNN folds.
    outer_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    outer_splits = list(
        outer_splitter.split(
            np.zeros(len(y)),
            y,
            groups
        )
    )

    # Recreate the exact inner CNN partitions.
    inner_splits = []

    manifest_rows = []

    for fold, (outer_train, outer_test) in enumerate(
        outer_splits,
        start=1
    ):

        inner_train, inner_val = create_inner_split(
            outer_train,
            y,
            groups,
            RANDOM_SEED + fold
        )

        # Explicitly verify no date leakage.
        train_dates = set(groups[inner_train])
        val_dates = set(groups[inner_val])
        test_dates = set(groups[outer_test])

        assert train_dates.isdisjoint(val_dates)
        assert train_dates.isdisjoint(test_dates)
        assert val_dates.isdisjoint(test_dates)

        inner_splits.append(
            (inner_train, inner_val)
        )

        manifest_rows.append({
            "fold": fold,
            "inner_train_samples": len(inner_train),
            "inner_validation_samples": len(inner_val),
            "outer_test_samples": len(outer_test),
            "inner_train_dates": len(train_dates),
            "inner_validation_dates": len(val_dates),
            "outer_test_dates": len(test_dates),
        })

    pd.DataFrame(manifest_rows).to_csv(
        OUTPUT_DIR / "split_manifest.csv",
        index=False
    )

    # Verify the recreated splits against CNN OOF predictions.
    cnn = load_and_validate_cnn_predictions(
        df,
        y,
        outer_splits
    )

    # Extract deterministic HOG features.
    print("\nEXTRACTING HOG FEATURES")
    print("=" * 55)

    X = compute_features(
        df,
        DATA_DIR,
        CACHE_PATH
    )

    print("Feature matrix:", X.shape)

    # Output storage.
    fold_rows = []
    prediction_rows = []
    confusion_rows = []
    per_class_rows = []

    # Add existing CNN predictions to the common prediction table.
    for row in cnn.itertuples(index=False):

        prediction_rows.append({
            "model": "Lightweight CNN",
            "crop_path": row.crop_path,
            "date": row.date,
            "fold": int(row.fold),
            "true_label": int(row.true_label),
            "predicted_label": int(row.predicted_label),
            "correct": bool(
                row.true_label == row.predicted_label
            ),
        })

    models = make_models()

    # --------------------------------------------------------
    # Evaluate CNN using its existing outer-test predictions.
    # --------------------------------------------------------

    for fold in range(1, N_SPLITS + 1):

        fold_cnn = cnn[cnn["fold"] == fold]

        fold_metrics = calculate_metrics(
            fold_cnn["true_label"].to_numpy(),
            fold_cnn["predicted_label"].to_numpy()
        )

        fold_rows.append({
            "model": "Lightweight CNN",
            "fold": fold,
            "n_train": manifest_rows[fold - 1][
                "inner_train_samples"
            ],
            "n_test": len(fold_cnn),
            "training_seconds": np.nan,
            "inference_ms_per_sample": np.nan,
            **fold_metrics,
        })

    # --------------------------------------------------------
    # Train and evaluate classical models.
    # --------------------------------------------------------

    for model_name, base_model in models.items():

        print("\n" + "=" * 65)
        print("MODEL:", model_name)
        print("=" * 65)

        for fold, (
            outer_split,
            inner_split
        ) in enumerate(
            zip(outer_splits, inner_splits),
            start=1
        ):

            outer_train, outer_test = outer_split
            inner_train, inner_val = inner_split

            # Use only the same inner training samples as CNN.
            X_train = X[inner_train]
            y_train = y[inner_train]

            # Evaluate on the exact same outer test samples.
            X_test = X[outer_test]
            y_test = y[outer_test]

            model = make_models()[model_name]

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train
            )

            training_seconds = (
                time.perf_counter() - start
            )

            start = time.perf_counter()

            predictions = model.predict(X_test)

            inference_seconds = (
                time.perf_counter() - start
            )

            inference_ms = (
                inference_seconds / len(outer_test)
            ) * 1000

            fold_metrics = calculate_metrics(
                y_test,
                predictions
            )

            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "n_train": len(inner_train),
                "n_test": len(outer_test),
                "training_seconds": training_seconds,
                "inference_ms_per_sample": inference_ms,
                **fold_metrics,
            })

            # Store out-of-fold predictions.
            for idx, predicted in zip(
                outer_test,
                predictions
            ):

                prediction_rows.append({
                    "model": model_name,
                    "crop_path": str(
                        df.iloc[idx]["crop_path"]
                    ),
                    "date": str(
                        df.iloc[idx]["date"]
                    ),
                    "fold": fold,
                    "true_label": int(y[idx]),
                    "predicted_label": int(predicted),
                    "correct": bool(
                        y[idx] == predicted
                    ),
                })

            print(
                f"Fold {fold}: "
                f"Accuracy={fold_metrics['accuracy']:.4f} | "
                f"Macro-F1={fold_metrics['macro_f1']:.4f} | "
                f"Training={training_seconds:.2f}s"
            )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        index=False
    )

    # ========================================================
    # FOLD METRICS
    # ========================================================

    fold_df = pd.DataFrame(fold_rows)

    fold_df.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False
    )

    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    summary_rows = []

    for model_name in fold_df["model"].unique():

        subset = fold_df[
            fold_df["model"] == model_name
        ]

        summary = {
            "model": model_name,
            "accuracy_mean": subset["accuracy"].mean(),
            "accuracy_std": subset["accuracy"].std(ddof=1),
            "macro_f1_mean": subset["macro_f1"].mean(),
            "macro_f1_std": subset["macro_f1"].std(ddof=1),
            "macro_precision_mean": subset[
                "macro_precision"
            ].mean(),
            "macro_recall_mean": subset[
                "macro_recall"
            ].mean(),
            "mean_training_seconds": subset[
                "training_seconds"
            ].mean(),
            "mean_inference_ms_per_sample": subset[
                "inference_ms_per_sample"
            ].mean(),
        }

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "summary_metrics.csv",
        index=False
    )

    # ========================================================
    # POOLED METRICS AND CONFUSION MATRICES
    # ========================================================

    pooled_rows = []

    for model_name in predictions_df["model"].unique():

        subset = predictions_df[
            predictions_df["model"] == model_name
        ]

        y_true = subset["true_label"].to_numpy()
        y_pred = subset["predicted_label"].to_numpy()

        pooled_metrics = calculate_metrics(
            y_true,
            y_pred
        )

        pooled_rows.append({
            "model": model_name,
            "n_samples": len(subset),
            "errors": int(
                (y_true != y_pred).sum()
            ),
            **pooled_metrics,
        })

        confusion_rows.extend(
            confusion_rows_for_model(
                model_name,
                y_true,
                y_pred
            )
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=CLASS_LABELS,
                zero_division=0
            )
        )

        for index, class_name in enumerate(CLASS_NAMES):

            per_class_rows.append({
                "model": model_name,
                "class": class_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            })

    pd.DataFrame(pooled_rows).to_csv(
        OUTPUT_DIR / "pooled_metrics.csv",
        index=False
    )

    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT_DIR / "confusion_matrices.csv",
        index=False
    )

    pd.DataFrame(per_class_rows).to_csv(
        OUTPUT_DIR / "per_class_metrics.csv",
        index=False
    )

    # ========================================================
    # ERROR COMPARISON
    # ========================================================

    errors_df = predictions_df[
        ~predictions_df["correct"]
    ].copy()

    errors_df.to_csv(
        OUTPUT_DIR / "errors.csv",
        index=False
    )

    # ========================================================
    # EXPERIMENT CONFIGURATION
    # ========================================================

    run_info = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pytorch": torch.__version__,
        "samples": int(len(df)),
        "feature_dimensions": int(X.shape[1]),
        "unique_dates": int(len(np.unique(groups))),
        "outer_folds": N_SPLITS,
        "inner_validation_size": INNER_VALIDATION_SIZE,
        "seed": RANDOM_SEED,
        "grouping": "date",
        "training_protocol": (
            "Classical models trained on the exact inner "
            "training partitions used by the CNN. "
            "All models evaluated on identical outer "
            "date-grouped test folds."
        ),
        "classical_hyperparameters": (
            "Fixed configurations inherited from evaluate.py. "
            "No outer-test hyperparameter tuning."
        ),
        "cnn_predictions_source": str(CNN_OOF_PATH),
        "cnn_fold_assignments_verified": True,
        "limitations": [
            "CNN training time is not included in this comparison.",
            "Classical models use fixed hyperparameters.",
            "CNN uses validation-based early stopping.",
            "Inference timing is a single-run diagnostic.",
            "Results are based on one dataset."
        ]
    }

    with open(
        OUTPUT_DIR / "experiment_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            run_info,
            file,
            indent=2
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 75)
    print("FAIR MODEL COMPARISON - POOLED RESULTS")
    print("=" * 75)

    print(
        pd.DataFrame(pooled_rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nFOLD MEAN AND STANDARD DEVIATION")
    print("=" * 75)

    print(
        summary_df[
            [
                "model",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nTOTAL ERRORS BY MODEL")
    print("=" * 75)

    print(
        errors_df.groupby("model").size().to_string()
    )

    print("\nSaved all comparison outputs to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
