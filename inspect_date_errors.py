
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed_data"
PREDICTIONS_FILE = (
    PROJECT_DIR / "outputs" / "cnn_date_cv" / "oof_predictions.csv"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "date_error_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DATES = [
    "2012-10-30",
    "2012-11-06",
    "2012-11-07",
]

# Contact sheet layout
COLS = 4
TILE_W = 210
TILE_H = 230
IMAGE_W = 170
IMAGE_H = 150


def parse_boolean(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def create_tile(row):
    image_path = DATA_DIR / str(row["crop_path"])

    if not image_path.exists():
        print(f"Missing image: {image_path}")
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Unreadable image: {image_path}")
        return None

    image = cv2.resize(
        image,
        (IMAGE_W, IMAGE_H),
        interpolation=cv2.INTER_AREA
    )

    tile = np.full((TILE_H, TILE_W, 3), 245, dtype=np.uint8)

    # Convert grayscale image to color
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    x = (TILE_W - IMAGE_W) // 2
    y = 8

    tile[y:y + IMAGE_H, x:x + IMAGE_W] = image_color

    correct = parse_boolean(row["correct"])

    # Red = incorrect, green = correct
    border_color = (0, 0, 255) if not correct else (0, 180, 0)

    cv2.rectangle(
        tile,
        (x - 2, y - 2),
        (x + IMAGE_W + 1, y + IMAGE_H + 1),
        border_color,
        3
    )

    true_label = str(row["true_class"])
    predicted_label = str(row["predicted_class"])
    fold = str(row["fold"])

    cv2.putText(
        tile,
        f"Actual: {true_label}",
        (8, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1
    )

    cv2.putText(
        tile,
        f"Predicted: {predicted_label}",
        (8, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 0, 0),
        1
    )

    cv2.putText(
        tile,
        f"Fold: {fold} | {'CORRECT' if correct else 'ERROR'}",
        (8, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        border_color,
        1
    )

    return tile


def create_date_sheet(df, date):
    date_df = df[df["date"].astype(str) == date].copy()

    if date_df.empty:
        print(f"No samples found for {date}")
        return

    date_df["_correct"] = date_df["correct"].apply(parse_boolean)

    # Show errors first, followed by correct predictions
    date_df = date_df.sort_values(
        by="_correct",
        ascending=True
    )

    tiles = []

    for _, row in date_df.iterrows():
        tile = create_tile(row)

        if tile is not None:
            tiles.append(tile)

    if not tiles:
        return

    rows = (len(tiles) + COLS - 1) // COLS

    sheet = np.full(
        (rows * TILE_H, COLS * TILE_W, 3),
        225,
        dtype=np.uint8
    )

    for index, tile in enumerate(tiles):
        r = index // COLS
        c = index % COLS

        y = r * TILE_H
        x = c * TILE_W

        sheet[
            y:y + TILE_H,
            x:x + TILE_W
        ] = tile

    output_file = OUTPUT_DIR / f"{date}_contact_sheet.jpg"

    cv2.imwrite(
        str(output_file),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    errors = sum(
        not parse_boolean(value)
        for value in date_df["correct"]
    )

    print(f"\nDate: {date}")
    print(f"Images displayed: {len(tiles)}")
    print(f"Errors: {errors}")
    print(f"Correct: {len(tiles) - errors}")
    print(f"Saved: {output_file}")


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required = [
        "date",
        "crop_path",
        "true_class",
        "predicted_class",
        "correct",
        "fold",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for date in TARGET_DATES:
        create_date_sheet(df, date)

    print("\nVisual inspection sheets created.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
