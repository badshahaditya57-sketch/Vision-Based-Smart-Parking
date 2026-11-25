
"""Controlled preprocessing experiment for date-grouped smart parking evaluation.

Compares:
1. Original grayscale + HOG
2. CLAHE + HOG
3. Percentile contrast normalization + HOG

All methods use identical date-grouped StratifiedGroupKFold splits and
the same Random Forest configuration.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

from common import (
    CLASS_NAMES,
    HOG_PARAMS,
    SEED,
    TARGET_H,
    TARGET_W,
    load_samples,
    resolve_groups,
)


def standardize_image(image):
    """Convert an image to grayscale and resize to the project's HOG size."""
    if image is None:
        raise ValueError("Image could not be decoded.")

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.resize(
        gray,
        (TARGET_W, TARGET_H),
        interpolation=cv2.INTER_AREA
    )


def apply_clahe(gray):
    """Apply CLAHE to a standardized grayscale image."""
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    return clahe.apply(gray)


def apply_contrast_normalization(gray):
    """Normalize intensity using the 2nd and 98th percentiles."""
    low, high = np.percentile(gray, [2, 98])

    if high <= low:
        return gray.copy()

    normalized = (gray.astype(np.float32) - low) * (
        255.0 / (high - low)
    )

    return np.clip(normalized, 0, 255).astype(np.uint8)


def extract_hog_features(image):
    """Extract HOG using exactly the project's existing HOG parameters."""
    return hog(
        image,
        **HOG_PARAMS
    ).astype(np.float32)


def load_preprocessed_features(df, data_dir, method):
    """Load crops and extract features using one preprocessing method."""
    features = []

    for index, row in df.iterrows():
        image_path = data_dir / str(row["crop_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(
                f"Could not read crop at row {index}: {image_path}"
            )

        gray = standardize_image(image)

        if method == "original":
            processed = gray

        elif method == "clahe":
            processed = apply_clahe(gray)

        elif method == "contrast_normalized":
            processed = apply_contrast_normalization(gray)

        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

        features.append(extract_hog_features(processed))

        if (index + 1) % 500 == 0:
            print(
                f"  Processed {index + 1}/{len(df)} crops",
                flush=True
            )

    X = np.asarray(features, dtype=np.float32)

    if X.ndim != 2 or X.shape[1] != 3780:
        raise ValueError(
            f"Unexpected HOG feature shape: {X.shape}. "
            "Expected 3780 features per crop."
        )

    return X


def build_grouped_splits(y, groups, n_splits, seed):
    """Build and validate the same type of grouped splits as evaluate.py."""
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)

    if k < 2:
        raise ValueError(
            "Grouped evaluation requires at least two distinct groups."
        )

    splitter = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=seed
    )

    splits = list(
        splitter.split(np.zeros(len(y)), y, groups)
    )

    valid_splits = []

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        if len(np.unique(y[train_idx])) < 2:
            raise ValueError(
                f"Fold {fold} training partition contains only one class."
            )

        if set(groups[train_idx]) & set(groups[test_idx]):
            raise AssertionError(
                f"Group leakage detected in fold {fold}."
            )

        if len(np.unique(y[test_idx])) < 2:
            print(
                f"WARNING: Fold {fold} test partition contains one class."
            )

        valid_splits.append((train_idx, test_idx))

    return valid_splits


def make_model(seed):
    """Use the same Random Forest settings as evaluate.py."""
    return RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )


def evaluate_method(
    method,
    X,
    df,
    y,
    splits,
    seed
):
    """Evaluate one preprocessing method on the fixed folds."""
    fold_rows = []
    prediction_rows = []
    confusion_rows = []

    oof_predictions = np.full(len(y), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"  Fold {fold}/{len(splits)}")

        model = make_model(seed)

        model.fit(
            X[train_idx],
            y[train_idx]
        )

        predictions = model.predict(X[test_idx])
        oof_predictions[test_idx] = predictions

        accuracy = accuracy_score(
            y[test_idx],
            predictions
        )

        macro_f1 = f1_score(
            y[test_idx],
            predictions,
            labels=[0, 1],
            average="macro",
            zero_division=0
        )

        precision, recall, class_f1, _ = (
            precision_recall_fscore_support(
                y[test_idx],
                predictions,
                labels=[0, 1],
                zero_division=0
            )
        )

        fold_rows.append({
            "method": method,
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "empty_precision": precision[0],
            "empty_recall": recall[0],
            "empty_f1": class_f1[0],
            "occupied_precision": precision[1],
            "occupied_recall": recall[1],
            "occupied_f1": class_f1[1],
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
        })

        matrix = confusion_matrix(
            y[test_idx],
            predictions,
            labels=[0, 1]
        )

        for true_class in range(2):
            for predicted_class in range(2):
                confusion_rows.append({
                    "method": method,
                    "fold": fold,
                    "true_class": CLASS_NAMES[true_class],
                    "predicted_class": CLASS_NAMES[predicted_class],
                    "count": int(matrix[true_class, predicted_class])
                })

        test_metadata = df.iloc[test_idx]

        for position, sample_index in enumerate(test_idx):
            row = test_metadata.iloc[position]

            prediction_rows.append({
                "method": method,
                "fold": fold,
                "sample_index": int(sample_index),
                "crop_path": row["crop_path"],
                "source_image": row.get("source_image", "unknown"),
                "date": row.get("date", "unknown"),
                "frame_id": row.get("frame_id", "unknown"),
                "true_label": int(y[sample_index]),
                "true_class": CLASS_NAMES[int(y[sample_index])],
                "predicted_label": int(predictions[position]),
                "predicted_class": CLASS_NAMES[int(predictions[position])],
                "correct": bool(
                    y[sample_index] == predictions[position]
                )
            })

    if np.any(oof_predictions == -1):
        raise AssertionError(
            "Some samples did not receive an out-of-fold prediction."
        )

    pooled_matrix = confusion_matrix(
        y,
        oof_predictions,
        labels=[0, 1]
    )

    pooled_accuracy = accuracy_score(y, oof_predictions)

    pooled_macro_f1 = f1_score(
        y,
        oof_predictions,
        labels=[0, 1],
        average="macro",
        zero_division=0
    )

    return (
        fold_rows,
        prediction_rows,
        confusion_rows,
        pooled_matrix,
        pooled_accuracy,
        pooled_macro_f1
    )


def build_summary(fold_df, predictions_df, method_names):
    """Build mean/std fold metrics and pooled out-of-fold metrics."""
    summary_rows = []

    for method in method_names:
        method_folds = fold_df[
            fold_df["method"] == method
        ]

        method_predictions = predictions_df[
            predictions_df["method"] == method
        ]

        row = {
            "method": method,
            "accuracy_mean": method_folds["accuracy"].mean(),
            "accuracy_std": method_folds["accuracy"].std(ddof=1),
            "macro_f1_mean": method_folds["macro_f1"].mean(),
            "macro_f1_std": method_folds["macro_f1"].std(ddof=1),
            "pooled_oof_accuracy": accuracy_score(
                method_predictions["true_label"],
                method_predictions["predicted_label"]
            ),
            "pooled_oof_macro_f1": f1_score(
                method_predictions["true_label"],
                method_predictions["predicted_label"],
                labels=[0, 1],
                average="macro",
                zero_division=0
            ),
            "total_samples": len(method_predictions),
            "total_errors": int((~method_predictions["correct"]).sum())
        }

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def build_date_metrics(predictions_df):
    """Calculate out-of-fold performance separately for each date."""
    rows = []

    for (method, date), group in predictions_df.groupby(
        ["method", "date"],
        dropna=False
    ):
        y_true = group["true_label"]
        y_pred = group["predicted_label"]

        rows.append({
            "method": method,
            "date": date,
            "samples": len(group),
            "errors": int((y_true != y_pred).sum()),
            "error_rate_pct": (
                100.0 * (y_true != y_pred).mean()
            ),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare preprocessing methods using date-grouped CV."
    )

    parser.add_argument(
        "--data-dir",
        default="./data/processed_data"
    )

    parser.add_argument(
        "--out",
        default="./outputs/preprocessing_experiment"
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading dataset...")

    df = load_samples(
        args.data_dir,
        None,
        args.max_per_class,
        args.seed
    )

    y = df["label"].to_numpy(dtype=int)

    groups, group_description = resolve_groups(
        df,
        "date"
    )

    print(f"Samples: {len(df)}")
    print(f"Classes: {np.bincount(y)}")
    print(f"Grouping: {group_description}")

    # Generate folds only once. Every preprocessing method uses these folds.
    splits = build_grouped_splits(
        y,
        groups,
        args.n_splits,
        args.seed
    )

    print(f"Number of folds: {len(splits)}")

    methods = [
        "original",
        "clahe",
        "contrast_normalized"
    ]

    all_fold_rows = []
    all_prediction_rows = []
    all_confusion_rows = []
    pooled_confusion_rows = []

    for method in methods:
        print("\n" + "=" * 65)
        print(f"PREPROCESSING METHOD: {method}")
        print("=" * 65)

        X = load_preprocessed_features(
            df,
            data_dir,
            method
        )

        print(f"Feature matrix shape: {X.shape}")

        (
            fold_rows,
            prediction_rows,
            confusion_rows,
            pooled_matrix,
            pooled_accuracy,
            pooled_macro_f1
        ) = evaluate_method(
            method,
            X,
            df,
            y,
            splits,
            args.seed
        )

        all_fold_rows.extend(fold_rows)
        all_prediction_rows.extend(prediction_rows)
        all_confusion_rows.extend(confusion_rows)

        for true_class in range(2):
            for predicted_class in range(2):
                pooled_confusion_rows.append({
                    "method": method,
                    "true_class": CLASS_NAMES[true_class],
                    "predicted_class": CLASS_NAMES[predicted_class],
                    "count": int(
                        pooled_matrix[true_class, predicted_class]
                    )
                })

        print(f"Pooled OOF accuracy: {pooled_accuracy:.4f}")
        print(f"Pooled OOF macro-F1: {pooled_macro_f1:.4f}")

        del X

    fold_df = pd.DataFrame(all_fold_rows)
    predictions_df = pd.DataFrame(all_prediction_rows)
    confusion_df = pd.DataFrame(all_confusion_rows)
    pooled_confusion_df = pd.DataFrame(pooled_confusion_rows)

    summary_df = build_summary(
        fold_df,
        predictions_df,
        methods
    )

    date_metrics_df = build_date_metrics(
        predictions_df
    )

    # Save all results separately from the original evaluation.
    fold_df.to_csv(
        out_dir / "fold_metrics.csv",
        index=False
    )

    predictions_df.to_csv(
        out_dir / "predictions.csv",
        index=False
    )

    confusion_df.to_csv(
        out_dir / "fold_confusion_matrices.csv",
        index=False
    )

    pooled_confusion_df.to_csv(
        out_dir / "pooled_confusion_matrices.csv",
        index=False
    )

    summary_df.to_csv(
        out_dir / "summary.csv",
        index=False
    )

    date_metrics_df.to_csv(
        out_dir / "date_metrics.csv",
        index=False
    )

    experiment_config = {
        "dataset_samples": len(df),
        "grouping": group_description,
        "n_splits": len(splits),
        "seed": args.seed,
        "methods": methods,
        "hog_parameters": HOG_PARAMS,
        "image_size": [TARGET_W, TARGET_H],
        "random_forest": {
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": args.seed
        },
        "clahe": {
            "clip_limit": 2.0,
            "tile_grid_size": [8, 8]
        },
        "contrast_normalization": {
            "lower_percentile": 2,
            "upper_percentile": 98
        }
    }

    with open(
        out_dir / "experiment_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            experiment_config,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("PREPROCESSING EXPERIMENT SUMMARY")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nResults saved in:")
    print(out_dir.resolve())

    print("\nGenerated files:")
    for filename in [
        "summary.csv",
        "fold_metrics.csv",
        "predictions.csv",
        "date_metrics.csv",
        "fold_confusion_matrices.csv",
        "pooled_confusion_matrices.csv",
        "experiment_config.json"
    ]:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
