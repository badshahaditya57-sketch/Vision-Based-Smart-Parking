
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

CSV_PATH = Path("outputs/evaluation_date/grouped_predictions.csv")
DATA_DIR = Path("data/processed_data")

TARGET_DATES = [
    "2012-10-30",
    "2012-11-06",
    "2012-11-07",
]

df = pd.read_csv(CSV_PATH)

df["correct"] = (
    df["correct"].astype(str).str.lower().eq("true")
)

df = df[
    (df["model"] == "HOG + Random Forest")
    & (df["date"].astype(str).isin(TARGET_DATES))
].copy()

results = []

for _, row in df.iterrows():
    image_path = DATA_DIR / str(row["crop_path"])

    try:
        with Image.open(image_path) as image:
            gray = np.array(image.convert("L"))

            results.append({
                "date": row["date"],
                "correct": row["correct"],
                "true_class": row["true_class"],
                "predicted_class": row["predicted_class"],
                "brightness": round(float(gray.mean()), 2),
                "contrast": round(float(gray.std()), 2),
                "width": gray.shape[1],
                "height": gray.shape[0],
            })

    except Exception as error:
        print(f"Could not read {image_path}: {error}")

stats = pd.DataFrame(results)

if stats.empty:
    print("No images were successfully analyzed.")
    raise SystemExit(1)

print("\nTOTAL IMAGES ANALYZED:", len(stats))

print("\n1. IMAGE STATISTICS BY DATE AND CORRECTNESS")

summary = stats.groupby(
    ["date", "correct"]
)[["brightness", "contrast", "width", "height"]].mean()

print(summary.round(2).to_string())

print("\n2. BRIGHTNESS: ERROR VS CORRECT")

print(
    stats.groupby("correct")["brightness"]
    .describe()
    .round(2)
    .to_string()
)

print("\n3. CONTRAST: ERROR VS CORRECT")

print(
    stats.groupby("correct")["contrast"]
    .describe()
    .round(2)
    .to_string()
)

print("\n4. CROP DIMENSIONS")

print(
    stats.groupby("correct")[["width", "height"]]
    .agg(["min", "max", "mean"])
    .round(2)
    .to_string()
)

print("\n5. FIRST 15 MISCLASSIFIED IMAGES")

errors = stats[~stats["correct"]]

print(errors.head(15).to_string(index=False))

print("\nAnalysis completed.")
