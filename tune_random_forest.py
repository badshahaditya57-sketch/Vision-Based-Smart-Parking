
"""Nested date-grouped hyperparameter tuning for Random Forest.

Outer CV estimates generalization to unseen dates.
Inner CV selects hyperparameters using macro-F1.

Existing baseline files are not modified.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedGroupKFold,
)

from common import (
    CLASS_NAMES,
    SEED,
    compute_features,
    load_samples,
    resolve_groups,
)


PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "class_weight": ["balanced"],
}


def calculate_metrics(y_true, y_pred):

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "empty_precision": float(precision[0]),
        "empty_recall": float(recall[0]),
        "empty_f1": float(f1[0]),
        "occupied_precision": float(precision[1]),
        "occupied_recall": float(recall[1]),
        "occupied_f1": float(f1[1]),
    }


def validate_split(y, groups, train_idx, test_idx, description):

    if len(np.unique(y[train_idx])) != 2:
        raise ValueError(
            f"{description}: training partition does not contain both classes."
        )

    if len(np.unique(y[test_idx])) != 2:
        raise ValueError(
            f"{description}: validation/test partition does not contain "
            "both classes. This configuration cannot provide reliable "
            "two-class macro-F1 evaluation."
        )

    if set(groups[train_idx]) & set(groups[test_idx]):
        raise AssertionError(
            f"{description}: group leakage detected."
        )


def make_group_splits(
    y,
    groups,
    n_splits,
    seed,
    description,
):

    unique_groups = np.unique(groups)

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"{description}: only {len(unique_groups)} groups available "
            f"for {n_splits} folds."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits = list(
        splitter.split(
            np.zeros(len(y)),
            y,
            groups,
        )
    )

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        validate_split(
            y,
            groups,
            train_idx,
            test_idx,
            f"{description}, fold {fold}",
        )

    return splits


def tune_inner(
    X,
    y,
    groups,
    inner_splits,
    seed,
):

    grid = list(ParameterGrid(PARAM_GRID))
    tuning_rows = []

    best_score = -np.inf
    best_params = None

    for candidate_number, params in enumerate(grid, 1):

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            inner_splits,
            1,
        ):

            model = RandomForestClassifier(
                **params,
                random_state=seed,
                n_jobs=-1,
            )

            model.fit(
                X[train_idx],
                y[train_idx],
            )

            predictions = model.predict(X[val_idx])

            score = f1_score(
                y[val_idx],
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )

            fold_scores.append(float(score))

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores, ddof=1))

        tuning_rows.append({
            "candidate": candidate_number,
            "params": json.dumps(params, sort_keys=True),
            "inner_macro_f1_mean": mean_score,
            "inner_macro_f1_std": std_score,
        })

        print(
            f"Candidate {candidate_number}/{len(grid)}: "
            f"macro-F1={mean_score:.4f} +/- {std_score:.4f}"
        )

        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()

    return best_params, best_score, tuning_rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="./data/processed_data",
    )

    parser.add_argument(
        "--out",
        default="./outputs/random_forest_nested_tuning",
    )

    parser.add_argument(
        "--outer-splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--inner-splits",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--group-by",
        choices=["date", "frame", "lot"],
        default="date",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    if args.group_by != "date":
        raise ValueError(
            "This experiment is designed for date-grouped evaluation. "
            "Use --group-by date."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    df = load_samples(
        args.data_dir,
        None,
        args.max_per_class,
        args.seed,
    )

    X = compute_features(
        df,
        args.data_dir,
        out / "cache" / "features.npz",
    )

    y = df["label"].to_numpy()

    groups, group_description = resolve_groups(
        df,
        args.group_by,
    )

    print(
        f"Samples: {len(df)} | "
        f"Features: {X.shape[1]} | "
        f"{group_description}"
    )

    print("\nPreparing outer date-grouped folds...")

    outer_splits = make_group_splits(
        y,
        groups,
        args.outer_splits,
        args.seed,
        "Outer CV",
    )

    all_predictions = []
    outer_metrics = []
    selected_parameters = []
    tuning_history = []

    for outer_fold, (outer_train, outer_test) in enumerate(
        outer_splits,
        1,
    ):

        print("\n" + "=" * 60)
        print(f"OUTER FOLD {outer_fold}/{len(outer_splits)}")
        print("=" * 60)

        X_train = X[outer_train]
        y_train = y[outer_train]
        groups_train = groups[outer_train]

        X_test = X[outer_test]
        y_test = y[outer_test]

        print("Preparing inner folds...")

        inner_splits_local = make_group_splits(
            y_train,
            groups_train,
            args.inner_splits,
            args.seed + outer_fold,
            f"Outer fold {outer_fold} inner CV",
        )

        # Convert local inner indices to global dataset indices.
        inner_splits = [
            (
                outer_train[train_idx],
                outer_train[val_idx],
            )
            for train_idx, val_idx in inner_splits_local
        ]

        print("\nTuning hyperparameters...")

        best_params, best_inner_score, history = tune_inner(
            X,
            y,
            groups,
            inner_splits,
            args.seed + outer_fold,
        )

        for row in history:
            tuning_history.append({
                "outer_fold": outer_fold,
                **row,
            })

        print("\nSelected parameters:")
        print(json.dumps(best_params, indent=2))
        print(f"Best inner macro-F1: {best_inner_score:.4f}")

        selected_parameters.append({
            "outer_fold": outer_fold,
            "best_inner_macro_f1": best_inner_score,
            "best_params": json.dumps(
                best_params,
                sort_keys=True,
            ),
        })

        # Fit tuned model on the complete outer training partition.
        tuned_model = RandomForestClassifier(
            **best_params,
            random_state=args.seed,
            n_jobs=-1,
        )

        tuned_model.fit(X_train, y_train)
        tuned_predictions = tuned_model.predict(X_test)

        tuned_metrics = calculate_metrics(
            y_test,
            tuned_predictions,
        )

        # Fit original baseline on exactly the same outer training data.
        baseline_model = RandomForestClassifier(
            n_estimators=200,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced",
        )

        baseline_model.fit(X_train, y_train)
        baseline_predictions = baseline_model.predict(X_test)

        baseline_metrics = calculate_metrics(
            y_test,
            baseline_predictions,
        )

        for model_name, predictions, model_metrics in [
            (
                "Random Forest Baseline",
                baseline_predictions,
                baseline_metrics,
            ),
            (
                "Random Forest Tuned",
                tuned_predictions,
                tuned_metrics,
            ),
        ]:

            outer_metrics.append({
                "outer_fold": outer_fold,
                "model": model_name,
                "n_train": len(outer_train),
                "n_test": len(outer_test),
                "n_train_dates": len(
                    np.unique(groups[outer_train])
                ),
                "n_test_dates": len(
                    np.unique(groups[outer_test])
                ),
                **model_metrics,
            })

            for idx, prediction in zip(
                outer_test,
                predictions,
            ):

                sample = df.iloc[idx]

                all_predictions.append({
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "sample_index": int(idx),
                    "date": sample["date"],
                    "frame_id": sample["frame_id"],
                    "crop_path": sample["crop_path"],
                    "true_label": int(y[idx]),
                    "true_class": CLASS_NAMES[int(y[idx])],
                    "predicted_label": int(prediction),
                    "predicted_class": CLASS_NAMES[int(prediction)],
                    "correct": bool(y[idx] == prediction),
                })

        print("\nOuter test results:")
        print("Baseline:", baseline_metrics)
        print("Tuned:", tuned_metrics)

    # Save detailed outputs.
    metrics_df = pd.DataFrame(outer_metrics)
    predictions_df = pd.DataFrame(all_predictions)

    metrics_df.to_csv(
        out / "outer_fold_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        out / "outer_predictions.csv",
        index=False,
    )

    pd.DataFrame(selected_parameters).to_csv(
        out / "selected_parameters.csv",
        index=False,
    )

    pd.DataFrame(tuning_history).to_csv(
        out / "inner_tuning_history.csv",
        index=False,
    )

    # Pooled out-of-fold confusion matrices.
    confusion_rows = []

    for model_name in metrics_df["model"].unique():

        subset = predictions_df[
            predictions_df["model"] == model_name
        ]

        cm = confusion_matrix(
            subset["true_label"],
            subset["predicted_label"],
            labels=[0, 1],
        )

        for i, true_name in enumerate(CLASS_NAMES):
            for j, predicted_name in enumerate(CLASS_NAMES):

                confusion_rows.append({
                    "model": model_name,
                    "true_class": true_name,
                    "predicted_class": predicted_name,
                    "count": int(cm[i, j]),
                })

    pd.DataFrame(confusion_rows).to_csv(
        out / "pooled_confusion_matrices.csv",
        index=False,
    )

    # Fold summary.
    summary = (
        metrics_df.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
    )

    summary.to_csv(
        out / "summary_metrics.csv",
        index=False,
    )

    run_info = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "samples": int(len(df)),
        "features": int(X.shape[1]),
        "grouping": group_description,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "optimization_metric": "macro_f1",
        "parameter_grid": PARAM_GRID,
        "seed": args.seed,
        "protocol": "nested_stratified_group_cross_validation",
        "note": (
            "Each outer test fold is excluded from inner "
            "hyperparameter selection. Outer-fold scores estimate "
            "the complete tuning procedure."
        ),
    }

    with open(out / "run_info.json", "w") as file:
        json.dump(run_info, file, indent=2)

    print("\n" + "=" * 60)
    print("FINAL NESTED CV SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    print(
        f"\nSaved all outputs to: {out.resolve()}"
    )


if __name__ == "__main__":
    main()
