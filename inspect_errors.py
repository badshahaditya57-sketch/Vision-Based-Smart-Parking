
"""Create contact sheets for Random Forest date-specific errors."""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

CSV_PATH = Path("outputs/evaluation_date/grouped_predictions.csv")
DATA_DIR = Path("data/processed_data")
OUTPUT_DIR = Path("outputs/error_analysis")

DATES = ["2012-10-30", "2012-11-06", "2012-11-07"]

MODEL = "HOG + Random Forest"

CELL_WIDTH = 260
CELL_HEIGHT = 230
IMAGE_HEIGHT = 180
COLUMNS = 4


def create_contact_sheet(date, rows):
    if rows.empty:
        print(f"No images available for {date}")
        return

    count = len(rows)
    rows_count = (count + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGB",
        (COLUMNS * CELL_WIDTH, 65 + rows_count * CELL_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    draw.text(
        (15, 15),
        f"{date} | Random Forest | {count} samples",
        fill="black",
    )

    missing = 0

    for index, (_, row) in enumerate(rows.iterrows()):
        image_path = DATA_DIR / str(row["crop_path"])

        if not image_path.exists():
            missing += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            missing += 1
            continue

        image = ImageOps.contain(
            image,
            (CELL_WIDTH - 12, IMAGE_HEIGHT - 10),
        )

        column = index % COLUMNS
        row_number = index // COLUMNS

        x = column * CELL_WIDTH
        y = 65 + row_number * CELL_HEIGHT

        image_x = x + (CELL_WIDTH - image.width) // 2
        image_y = y + 5

        sheet.paste(image, (image_x, image_y))

        status = "ERROR" if not row["correct"] else "CORRECT"

        caption = (
            f"{status} | True: {row['true_class']}\n"
            f"Predicted: {row['predicted_class']}"
        )

        draw.text(
            (x + 7, y + IMAGE_HEIGHT + 5),
            caption,
            fill="red" if status == "ERROR" else "green",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"errors_{date}.jpg"
    sheet.save(output_path, quality=95)

    print(f"Saved: {output_path}")
    print(f"Missing/unreadable images: {missing}")


def main():
    data = pd.read_csv(CSV_PATH)

    data["correct"] = (
        data["correct"].astype(str).str.lower().eq("true")
    )

    data = data[
        (data["model"] == MODEL)
        & (data["date"].astype(str).isin(DATES))
    ].copy()

    for date in DATES:
        date_data = data[data["date"].astype(str) == date]

        errors = date_data[~date_data["correct"]].copy()
        correct = date_data[date_data["correct"]].sample(
            n=min(6, int(date_data["correct"].sum())),
            random_state=42,
        )

        # Show every error and up to six correctly classified examples.
        selected = pd.concat([errors, correct])

        print(f"\n{date}:")
        print(f"Errors: {len(errors)}")
        print(f"Correct examples included: {len(correct)}")

        create_contact_sheet(date, selected)


if __name__ == "__main__":
    main()
