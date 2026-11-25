
"""Date-level cluster bootstrap confidence intervals for parking models.

Uses existing out-of-fold predictions. No model retraining is performed.

Outputs:
    outputs/bootstrap_analysis/bootstrap_summary.csv
    outputs/bootstrap_analysis/paired_comparisons.csv
    outputs/bootstrap_analysis/bootstrap_replicates.csv
    outputs/bootstrap_analysis/experiment_config.json
"""

from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    ROOT / "outputs" / "fair_model_comparison" / "oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "bootstrap_analysis"

SEED = 42
N_BOOTSTRAP = 5000
CONFIDENCE_LEVEL = 0.95

MODELS = [
    "Lightweight CNN",
    "HOG + Random Forest",
    "HOG + Logistic Regression",
    "HOG + Linear SVM",
]

CNN_MODEL = "Lightweight CNN"
RF_MODEL = "HOG + Random Forest"

KEY_COLUMNS = [
    "crop_path",
    "date",
    "fold",
    "true_label",
]


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy and binary macro-F1."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    accuracy = np.mean(y_true == y_pred)

    f1_scores = []

    for label in [0, 1]:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        denominator = 2 * tp + fp + fn

        if denominator == 0:
            f1 = 0.0
        else:
            f1 = (2 * tp) / denominator

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
    }


def load_and_validate_predictions():
    """Load predictions and ensure every model uses identical samples."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "model",
        "crop_path",
        "date",
        "fold",
        "true_label",
        "predicted_label",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()

    if df[required].isnull().any().any():
        raise ValueError("Input contains missing values.")

    df["true_label"] = pd.to_numeric(
        df["true_label"], errors="raise"
    ).astype(int)

    df["predicted_label"] = pd.to_numeric(
        df["predicted_label"], errors="raise"
    ).astype(int)

    if not set(df["true_label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected true labels. Expected 0 and 1.")

    if not set(df["predicted_label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected predicted labels. Expected 0 and 1.")

    found_models = set(df["model"].unique())
    expected_models = set(MODELS)

    if found_models != expected_models:
        raise ValueError(
            f"Model mismatch.\nExpected: {expected_models}\n"
            f"Found: {found_models}"
        )

    if df.duplicated(KEY_COLUMNS + ["model"]).any():
        raise ValueError(
            "Duplicate sample/model rows found. "
            "Cannot safely pair predictions."
        )

    # Each model must have the same samples and true labels.
    wide = df.pivot(
        index=KEY_COLUMNS,
        columns="model",
        values="predicted_label",
    ).reset_index()

    if wide[MODELS].isnull().any().any():
        raise ValueError(
            "Some samples are missing predictions for one or more models."
        )

    wide["date"] = wide["date"].astype(str)

    if wide["date"].nunique() < 2:
        raise ValueError("At least two distinct dates are required.")

    return wide


def main():
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("DATE-LEVEL CLUSTER BOOTSTRAP")
    print("=" * 65)

    wide = load_and_validate_predictions()

    dates = sorted(wide["date"].unique())

    print(f"Samples: {len(wide)}")
    print(f"Distinct dates: {len(dates)}")
    print(f"Models: {len(MODELS)}")
    print(f"Bootstrap repetitions: {N_BOOTSTRAP}")

    # Keep all images from a date together.
    date_data = {}

    for date in dates:
        date_df = wide[wide["date"] == date]

        date_data[date] = {
            "y_true": date_df["true_label"].to_numpy(dtype=int),
            "predictions": {
                model: date_df[model].to_numpy(dtype=int)
                for model in MODELS
            },
        }

    # Original OOF metrics.
    observed = {}

    for model in MODELS:
        observed[model] = calculate_metrics(
            wide["true_label"].to_numpy(),
            wide[model].to_numpy(),
        )

    print("\nOBSERVED OOF METRICS")

    for model in MODELS:
        metrics = observed[model]

        print(
            f"{model}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )

    rng = np.random.default_rng(SEED)

    bootstrap_rows = []

    date_count = len(dates)

    print("\nRunning bootstrap...")

    for iteration in range(N_BOOTSTRAP):

        # Sample date clusters with replacement.
        sampled_dates = rng.choice(
            dates,
            size=date_count,
            replace=True,
        )

        # Every crop from a selected date stays together.
        y_parts = []
        prediction_parts = {model: [] for model in MODELS}

        for date in sampled_dates:
            item = date_data[date]

            y_parts.append(item["y_true"])

            for model in MODELS:
                prediction_parts[model].append(
                    item["predictions"][model]
                )

        y_true = np.concatenate(y_parts)

        row = {"iteration": iteration + 1}

        for model in MODELS:
            y_pred = np.concatenate(prediction_parts[model])

            metrics = calculate_metrics(y_true, y_pred)

            model_key = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")

            row[f"{model_key}_accuracy"] = metrics["accuracy"]
            row[f"{model_key}_macro_f1"] = metrics["macro_f1"]

        # Paired differences: same sampled dates for both models.
        for metric in ["accuracy", "macro_f1"]:
            cnn_key = (
                f"{re.sub(r'[^A-Za-z0-9]+', '_', CNN_MODEL).strip('_')}"
                f"_{metric}"
            )

            rf_key = (
                f"{re.sub(r'[^A-Za-z0-9]+', '_', RF_MODEL).strip('_')}"
                f"_{metric}"
            )

            row[f"CNN_minus_RF_{metric}"] = (
                row[cnn_key] - row[rf_key]
            )

        bootstrap_rows.append(row)

        if (iteration + 1) % 500 == 0:
            print(
                f"Completed {iteration + 1}/{N_BOOTSTRAP}"
            )

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    bootstrap_path = OUTPUT_DIR / "bootstrap_replicates.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    alpha = 1.0 - CONFIDENCE_LEVEL

    # Individual model confidence intervals.
    summary_rows = []

    for model in MODELS:
        model_key = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")

        for metric in ["accuracy", "macro_f1"]:
            column = f"{model_key}_{metric}"

            lower = bootstrap_df[column].quantile(alpha / 2)
            upper = bootstrap_df[column].quantile(1 - alpha / 2)

            summary_rows.append({
                "model": model,
                "metric": metric,
                "observed_value": observed[model][metric],
                "confidence_level": CONFIDENCE_LEVEL,
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_repetitions": N_BOOTSTRAP,
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "bootstrap_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Paired CNN versus Random Forest differences.
    comparison_rows = []

    for metric in ["accuracy", "macro_f1"]:
        column = f"CNN_minus_RF_{metric}"

        observed_difference = (
            observed[CNN_MODEL][metric]
            - observed[RF_MODEL][metric]
        )

        lower = bootstrap_df[column].quantile(alpha / 2)
        upper = bootstrap_df[column].quantile(1 - alpha / 2)

        comparison_rows.append({
            "comparison": "Lightweight CNN - HOG + Random Forest",
            "metric": metric,
            "observed_difference": observed_difference,
            "confidence_level": CONFIDENCE_LEVEL,
            "ci_lower": lower,
            "ci_upper": upper,
            "bootstrap_repetitions": N_BOOTSTRAP,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_path = OUTPUT_DIR / "paired_comparisons.csv"
    comparison_df.to_csv(comparison_path, index=False)

    config = {
        "input_file": str(INPUT_FILE),
        "bootstrap_method": "Date-level cluster bootstrap",
        "number_of_samples": int(len(wide)),
        "number_of_dates": int(date_count),
        "models": MODELS,
        "bootstrap_repetitions": N_BOOTSTRAP,
        "random_seed": SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "Percentile",
        "resampling_unit": "Date",
        "sampling": "Sample dates with replacement; retain all crops per date",
        "comparison": "Paired CNN versus Random Forest",
        "limitations": [
            "Uses existing out-of-fold predictions; models are not retrained.",
            "Confidence intervals describe variation across sampled dates.",
            "Only one dataset is evaluated.",
            "Date-level bootstrap does not establish generalization to unseen lots or cameras.",
        ],
    }

    config_path = OUTPUT_DIR / "experiment_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "observed_value": "{:.4f}".format,
                "ci_lower": "{:.4f}".format,
                "ci_upper": "{:.4f}".format,
            },
        )
    )

    print("\n" + "=" * 65)
    print("PAIRED CNN VS RANDOM FOREST")
    print("=" * 65)

    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "observed_difference": "{:.4f}".format,
                "ci_lower": "{:.4f}".format,
                "ci_upper": "{:.4f}".format,
            },
        )
    )

    print("\nFILES SAVED")

    print(summary_path)
    print(comparison_path)
    print(bootstrap_path)
    print(config_path)

    print(f"\nElapsed time: {elapsed:.2f} seconds")
    print("\nBOOTSTRAP ANALYSIS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
