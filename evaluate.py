
"""Compare HOG + Linear SVM, Random Forest and Logistic Regression.

Use grouped results as the primary estimate. Random crop-level results are a
diagnostic baseline and may be optimistic when related crops share a frame.

Additional outputs:
  grouped_predictions.csv - all grouped out-of-fold predictions
  grouped_errors.csv      - incorrectly classified grouped samples
"""

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import SVC

from common import (
    CLASS_NAMES,
    SEED,
    compute_features,
    load_samples,
    resolve_groups,
)


def make_models(seed):
    return {
        "HOG + Linear SVM": SVC(
            kernel="linear",
            C=1.0,
            random_state=seed
        ),
        "HOG + Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "HOG + Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=seed
        ),
    }


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "empty_precision": float(p[0]),
        "empty_recall": float(r[0]),
        "empty_f1": float(f[0]),
        "occupied_precision": float(p[1]),
        "occupied_recall": float(r[1]),
        "occupied_f1": float(f[1]),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        ),
    }


def build_splits(y, groups, protocol, n_splits, seed):

    if protocol == "random":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        return list(splitter.split(np.zeros(len(y)), y))

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

    valid = []

    for fold, (tr, te) in enumerate(splits, 1):

        if len(np.unique(y[tr])) < 2:
            raise ValueError(
                f"Fold {fold} training partition contains only one class. "
                "Reduce folds, increase group coverage, or use another grouping."
            )

        if len(np.unique(y[te])) < 2:
            print(
                f"WARNING: fold {fold} test partition contains one class; "
                "per-class metrics may be zero."
            )

        if set(groups[tr]) & set(groups[te]):
            raise AssertionError(
                f"Group leakage detected in fold {fold}."
            )

        valid.append((tr, te))

    return valid


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data-dir", default="./processed_data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument(
        "--group-by",
        choices=["auto", "frame", "date", "lot"],
        default="auto"
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset and compute HOG features.
    df = load_samples(
        args.data_dir,
        None,
        args.max_per_class,
        args.seed
    )

    X = compute_features(
        df,
        args.data_dir,
        out / "cache" / "features.npz"
    )

    y = df["label"].to_numpy()

    groups, group_description = resolve_groups(
        df,
        args.group_by
    )

    print(
        f"Samples: {len(df)} | features: {X.shape[1]} | "
        f"{group_description}"
    )

    all_rows = []
    confusion_rows = []
    weather_rows = []
    prediction_rows = []
    summary_rows = []

    # Evaluate random and grouped protocols.
    for protocol in ("random", "grouped"):

        splits = build_splits(
            y,
            groups,
            protocol,
            args.n_splits,
            args.seed
        )

        for model_name, base_model in make_models(args.seed).items():

            y_true_all = []
            y_pred_all = []
            fold_rows = []

            for fold, (train_idx, test_idx) in enumerate(splits, 1):

                model = clone(base_model)

                model.fit(
                    X[train_idx],
                    y[train_idx]
                )

                pred = model.predict(X[test_idx])

                m = metrics(
                    y[test_idx],
                    pred
                )

                # Save grouped out-of-fold predictions for error analysis.
                if protocol == "grouped":

                    for idx, predicted in zip(test_idx, pred):

                        sample = df.iloc[idx]

                        prediction_rows.append({
                            "model": model_name,
                            "fold": fold,
                            "sample_index": int(idx),
                            "crop_path": sample["crop_path"],
                            "source_image": sample["source_image"],
                            "date": sample["date"],
                            "frame_id": sample["frame_id"],
                            "true_label": int(y[idx]),
                            "true_class": CLASS_NAMES[int(y[idx])],
                            "predicted_label": int(predicted),
                            "predicted_class": CLASS_NAMES[int(predicted)],
                            "correct": bool(y[idx] == predicted)
                        })

                row = {
                    "split": protocol,
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_train_groups": len(
                        np.unique(groups[train_idx])
                    ),
                    "n_test_groups": len(
                        np.unique(groups[test_idx])
                    ),
                    **m
                }

                fold_rows.append(row)
                all_rows.append(row)

                y_true_all.extend(y[test_idx].tolist())
                y_pred_all.extend(pred.tolist())

            # Calculate fold mean and standard deviation.
            for key in ("accuracy", "macro_f1"):

                vals = [r[key] for r in fold_rows]

                summary_rows.append({
                    "split": protocol,
                    "model": model_name,
                    f"{key}_mean": float(np.mean(vals)),
                    f"{key}_std": (
                        float(np.std(vals, ddof=1))
                        if len(vals) > 1 else 0.0
                    )
                })

            # Confusion matrix.
            cm = confusion_matrix(
                y_true_all,
                y_pred_all,
                labels=[0, 1]
            )

            for i, true_name in enumerate(CLASS_NAMES):

                for j, pred_name in enumerate(CLASS_NAMES):

                    confusion_rows.append({
                        "split": protocol,
                        "model": model_name,
                        "true_label": true_name,
                        "predicted_label": pred_name,
                        "count": int(cm[i, j])
                    })

            # Weather-sliced grouped out-of-fold predictions.
            if "weather" in df.columns:

                weather = (
                    df["weather"]
                    .fillna("unknown")
                    .astype(str)
                    .str.lower()
                    .to_numpy()
                )

                if protocol == "grouped":

                    preds = np.full(
                        len(y),
                        -1,
                        dtype=int
                    )

                    for train_idx, test_idx in splits:

                        model = clone(base_model).fit(
                            X[train_idx],
                            y[train_idx]
                        )

                        preds[test_idx] = model.predict(
                            X[test_idx]
                        )

                    for condition in ("sunny", "cloudy", "rainy"):

                        mask = (
                            (weather == condition)
                            & (preds >= 0)
                        )

                        if mask.sum():

                            wm = metrics(
                                y[mask],
                                preds[mask]
                            )

                            weather_rows.append({
                                "weather": condition,
                                "model": model_name,
                                "n_samples": int(mask.sum()),
                                "n_groups": int(
                                    len(np.unique(groups[mask]))
                                ),
                                **wm
                            })

    # Save evaluation outputs.
    pd.DataFrame(all_rows).to_csv(
        out / "fold_metrics.csv",
        index=False
    )

    pd.DataFrame(summary_rows).to_csv(
        out / "summary_metrics.csv",
        index=False
    )

    pd.DataFrame(confusion_rows).to_csv(
        out / "confusion_matrices.csv",
        index=False
    )

    # Save all grouped predictions and incorrect predictions.
    predictions_df = pd.DataFrame(prediction_rows)

    predictions_df.to_csv(
        out / "grouped_predictions.csv",
        index=False
    )

    errors_df = predictions_df[
        ~predictions_df["correct"]
    ].copy()

    errors_df.to_csv(
        out / "grouped_errors.csv",
        index=False
    )

    if weather_rows:

        pd.DataFrame(weather_rows).to_csv(
            out / "weather_metrics.csv",
            index=False
        )

    # Model comparison plot: grouped mean metrics only.
    summary = pd.DataFrame(summary_rows)

    grouped = summary[
        summary.split == "grouped"
    ]

    if not grouped.empty:

        ax = grouped.plot(
            x="model",
            y=["accuracy_mean", "macro_f1_mean"],
            kind="bar",
            ylim=(0, 1),
            rot=15
        )

        ax.set_ylabel("Score")
        ax.set_title(
            "Grouped cross-validation performance"
        )

        plt.tight_layout()

        plt.savefig(
            out / "fig_model_comparison.png",
            dpi=180
        )

        plt.close()

    # Save experiment information.
    run_info = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "n_samples": int(len(df)),
        "feature_dim": int(X.shape[1]),
        "class_counts": {
            str(k): int(v)
            for k, v in df["label"].value_counts().items()
        },
        "grouping": group_description,
        "n_splits_requested": args.n_splits,
        "primary_protocol": "grouped",
        "seed": args.seed,
        "note": (
            "Fold mean/std are descriptive; "
            "no nested hyperparameter tuning was performed."
        )
    }

    with open(out / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\nGrouped results (primary):")
    print(grouped.to_string(index=False))

    print("\nPrediction analysis:")
    print(f"Total grouped predictions: {len(predictions_df)}")
    print(f"Total grouped errors: {len(errors_df)}")

    print(
        f"\nSaved evaluation artifacts to: {out.resolve()}"
    )


if __name__ == "__main__":
    main()
