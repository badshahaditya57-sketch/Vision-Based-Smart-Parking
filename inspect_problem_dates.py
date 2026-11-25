
"""Create visual inspection sheets for date-specific false positives."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DATA_DIR = Path("./data/processed_data")
PREDICTIONS_FILE = Path(
    "./outputs/preprocessing_experiment/predictions.csv"
)
OUTPUT_DIR = Path(
    "./outputs/preprocessing_experiment/visual_audit"
)

PROBLEM_DATES = [
    "2012-10-30",
    "2012-11-06",
    "2012-11-07",
]

MAX_FALSE_POSITIVES = 12
MAX_CORRECT = 6

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_thumbnail(image_path, title, width=220, height=300):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.full(
        (height + 45, width, 3),
        245,
        dtype=np.uint8
    )

    canvas[:height] = image

    cv2.putText(
        canvas,
        title[:30],
        (5, height + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (20, 20, 20),
        1,
        cv2.LINE_AA
    )

    return canvas


def create_sheet(date, false_positives, correct_samples):
    tiles = []

    for _, row in false_positives.iterrows():
        path = DATA_DIR / str(row["crop_path"])

        tile = make_thumbnail(
            path,
            f"FALSE POS | {row['crop_path']}"
        )

        if tile is not None:
            tiles.append(tile)

    for _, row in correct_samples.iterrows():
        path = DATA_DIR / str(row["crop_path"])

        tile = make_thumbnail(
            path,
            f"CORRECT | {row['crop_path']}"
        )

        if tile is not None:
            tiles.append(tile)

    if not tiles:
        print(f"No readable images for {date}")
        return

    columns = 3
    rows = int(np.ceil(len(tiles) / columns))

    tile_h, tile_w = tiles[0].shape[:2]

    sheet = np.full(
        (rows * tile_h, columns * tile_w, 3),
        230,
        dtype=np.uint8
    )

    for index, tile in enumerate(tiles):
        r = index // columns
        c = index % columns

        sheet[
            r * tile_h:(r + 1) * tile_h,
            c * tile_w:(c + 1) * tile_w
        ] = tile

    output_path = OUTPUT_DIR / f"{date}_audit.jpg"

    cv2.imwrite(str(output_path), sheet)

    print(f"Saved: {output_path}")


def main():
    predictions = pd.read_csv(PREDICTIONS_FILE)

    original = predictions[
        predictions["method"] == "original"
    ].copy()

    for date in PROBLEM_DATES:
        subset = original[
            original["date"].astype(str) == date
        ]

        false_positives = subset[
            (subset["true_class"] == "empty")
            & (subset["predicted_class"] == "occupied")
        ]

        correct_samples = subset[
            (subset["true_class"] == "empty")
            & (subset["predicted_class"] == "empty")
        ]

        false_positives = false_positives.sample(
            n=min(MAX_FALSE_POSITIVES, len(false_positives)),
            random_state=42
        )

        correct_samples = correct_samples.sample(
            n=min(MAX_CORRECT, len(correct_samples)),
            random_state=42
        )

        print(
            f"\n{date}: "
            f"{len(false_positives)} false positives, "
            f"{len(correct_samples)} correct empty samples"
        )

        create_sheet(
            date,
            false_positives,
            correct_samples
        )


if __name__ == "__main__":
    main()
