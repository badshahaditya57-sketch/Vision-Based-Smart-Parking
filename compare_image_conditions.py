
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

CSV_PATH = Path("outputs/evaluation_date/grouped_predictions.csv")
DATA_DIR = Path("data/processed_data")

PROBLEM_DATES = [
    "2012-10-30",
    "2012-11-06",
    "2012-11-07",
]

MODEL = "HOG + Random Forest"

df = pd.read_csv(CSV_PATH)

df["correct"] = (
    df["correct"].astype(str).str.lower().eq("true")
)

# Keep only Random Forest predictions for empty spaces.
df = df[
    (df["model"] == MODEL)
    & (df["true_class"] == "empty")
].copy()

df["date_group"] = np.where(
    df["date"].astype(str).isin(PROBLEM_DATES),
    "Problem dates",
    "Other dates",
)

results = []

print("Analyzing empty-space crops...")

for _, row in df.iterrows():
    image_path = DATA_DIR / str(row["crop_path"])

    try:
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"))

            results.append({
                "date_group": row["date_group"],
                "date": row["date"],
                "correct": row["correct"],
                "brightness": float(gray.mean()),
                "contrast": float(gray.std()),
            })

    except Exception as error:
        print(f"Could not read {image_path}: {error}")

stats = pd.DataFrame(results)

if stats.empty:
    print("No images could be analyzed.")
    raise SystemExit(1)

stats["status"] = np.where(
    stats["correct"],
    "Correct",
    "Incorrect",
)

print("\n1. SAMPLE COUNTS")

print(
    stats.groupby(["date_group", "status"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\n2. BRIGHTNESS AND CONTRAST")

print(
    stats.groupby(["date_group", "status"])[
        ["brightness", "contrast"]
    ].agg(["mean", "median", "std"]).round(2).to_string()
)

print("\n3. ERROR RATE BY DATE GROUP")

error_rates = stats.groupby("date_group").agg(
    total=("correct", "size"),
    errors=("correct", lambda x: int((~x).sum())),
)

error_rates["error_rate_pct"] = (
    error_rates["errors"] / error_rates["total"] * 100
).round(2)

print(error_rates.to_string())

print("\n4. BRIGHTNESS RANGES")

print(
    stats.groupby(["date_group", "status"])["brightness"]
    .quantile([0.25, 0.50, 0.75])
    .round(2)
    .to_string()
)

print("\n5. CONTRAST RANGES")

print(
    stats.groupby(["date_group", "status"])["contrast"]
    .quantile([0.25, 0.50, 0.75])
    .round(2)
    .to_string()
)

print("\nAnalysis completed.")
