"""
preprocess.py -- COCO annotations -> labelled 64x128 grayscale crops + metadata.csv

Changes vs. the original version (see README, 'Methodology'):
  * writes metadata.csv so every crop can be traced back to its source frame,
    date, lot and weather (needed for leakage-free, group-based evaluation)
  * explicit label mapping; unknown categories are SKIPPED and reported
    (before: anything without 'empty' in its name silently became 'occupied')
  * crops saved as PNG (lossless) instead of JPEG (lossy, adds compression noise)
  * grayscale -> resize via common.standardize (same code path as the app)
  * optional --meta-csv to supply weather/lot per recording date
  * runs only when executed as a script (the original ran on import)

Usage:
  python preprocess.py --train-folder raw_dataset/train --output-dir processed_data
  python preprocess.py ... --meta-csv date_meta.csv     # optional, see README
"""
import argparse
import json
import os
from collections import Counter

import cv2
import pandas as pd

from common import CLASS_NAMES, frame_id_from_name, parse_image_name, standardize


def label_from_category(name):
    """Explicit mapping. Returns 0 (empty), 1 (occupied) or None (skip)."""
    name = name.lower()
    if "empty" in name:
        return 0
    if "occupied" in name:
        return 1
    return None


def process_coco_dataset(train_folder, output_dir, meta_csv=None, max_crops_per_class=0, seed=42):
    for c in CLASS_NAMES:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)

    with open(os.path.join(train_folder, "_annotations.coco.json"), "r") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    print("Categories found in dataset:", categories)
    images = {im["id"]: im["file_name"] for im in coco["images"]}

    # Optional external metadata keyed by recording date: columns date[,lot][,weather]
    date_meta = None
    if meta_csv:
        date_meta = pd.read_csv(meta_csv, dtype=str).set_index("date").to_dict("index")

    annotations = coco["annotations"]
    if max_crops_per_class:
        # Optional cap to limit disk use: random (not first-N) per class.
        df_ann = pd.DataFrame(
            {"i": range(len(annotations)),
             "lab": [label_from_category(categories.get(a["category_id"], "")) for a in annotations]}
        ).dropna()
        keep = set()
        for _, g in df_ann.groupby("lab"):
            keep.update(g.sample(n=min(len(g), max_crops_per_class), random_state=seed)["i"])
        annotations = [a for i, a in enumerate(annotations) if i in keep]

    rows, skipped_cat, n_missing_img = [], Counter(), 0
    frame_cache = {"id": None, "img": None}   # annotations of a frame are usually adjacent

    for ann in annotations:
        label = label_from_category(categories.get(ann["category_id"], ""))
        if label is None:
            skipped_cat[categories.get(ann["category_id"], "?")] += 1
            continue
        image_id = ann["image_id"]
        if image_id not in images:
            continue
        file_name = images[image_id]

        if frame_cache["id"] != image_id:
            frame = cv2.imread(os.path.join(train_folder, file_name))
            frame_cache = {"id": image_id, "img": frame}
        frame = frame_cache["img"]
        if frame is None:
            n_missing_img += 1
            continue

        # COCO bbox = [x_top_left, y_top_left, width, height]; clip to the frame on all sides
        H, W = frame.shape[:2]
        x, y, w, h = ann["bbox"]
        x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
        x2, y2 = min(W, int(round(x + w))), min(H, int(round(y + h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        crop = standardize(frame[y1:y2, x1:x2])
        rel = f"{CLASS_NAMES[label]}/spot_{len(rows):07d}.png"
        cv2.imwrite(os.path.join(output_dir, rel), crop)

        info = parse_image_name(file_name)
        if date_meta and info["date"] in date_meta:
            row = date_meta[info["date"]]
            if "weather" in row:
                info["weather"] = str(row["weather"]).strip().lower()
            if "lot" in row:
                info["lot"] = str(row["lot"]).strip().upper()
        rows.append({
            "crop_path": rel, "label": label, "label_name": CLASS_NAMES[label],
            "category": categories[ann["category_id"]],
            "source_image": file_name, "frame_id": frame_id_from_name(file_name),
            "image_id": image_id, "ann_id": ann.get("id", -1),
            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, **info,
        })

    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print(f"\nExtraction complete: {len(meta)} crops saved to {output_dir}")
    print("Class counts:\n", meta["label_name"].value_counts().to_string())
    if skipped_cat:
        print("Skipped annotations with unmapped categories:", dict(skipped_cat))
    if n_missing_img:
        print(f"Skipped {n_missing_img} annotations whose image could not be read")
    print(f"Distinct source frames: {meta['frame_id'].nunique()}")
    for col in ("date", "lot", "weather"):
        vc = meta[col].value_counts()
        print(f"{col}: {vc.to_dict() if len(vc) <= 12 else str(len(vc)) + ' distinct values'}")
    if (meta["weather"] == "unknown").all():
        print("NOTE: weather could not be recovered from filenames -> weather-stratified "
              "evaluation will be skipped unless you pass --meta-csv (see README).")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-folder", default="./raw_dataset/train")
    ap.add_argument("--output-dir", default="./processed_data")
    ap.add_argument("--meta-csv", default=None, help="CSV with columns date[,lot][,weather]")
    ap.add_argument("--max-crops-per-class", type=int, default=0,
                    help="optional random cap per class (0 = keep all)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    process_coco_dataset(a.train_folder, a.output_dir, a.meta_csv, a.max_crops_per_class, a.seed)
