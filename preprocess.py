"""Convert COCO parking-space annotations into standardized crops and metadata."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

from common import CLASS_NAMES, frame_id_from_name, parse_image_name, standardize

def label_from_category(name):
    """Return 0=empty, 1=occupied; reject ambiguous labels."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    empty_names = {"empty", "vacant", "available", "free"}
    occupied_names = {"occupied", "full", "taken"}
    if normalized in empty_names:
        return 0
    if normalized in occupied_names:
        return 1
    return None

def process_coco_dataset(train_folder, output_dir, meta_csv=None,
                          max_crops_per_class=0, seed=42):
    train_folder, output_dir = Path(train_folder), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    annotation_file = train_folder / "_annotations.coco.json"
    with annotation_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    images = {im["id"]: im["file_name"] for im in coco.get("images", [])}

    date_meta = {}
    if meta_csv:
        meta_df = pd.read_csv(meta_csv, dtype=str).fillna("")
        if "date" not in meta_df.columns:
            raise ValueError("meta CSV must contain a 'date' column")
        date_meta = meta_df.set_index("date").to_dict("index")

    annotations = coco.get("annotations", [])
    if max_crops_per_class:
        eligible = []
        for i, ann in enumerate(annotations):
            label = label_from_category(categories.get(ann.get("category_id"), ""))
            if label is not None:
                eligible.append((i, label))
        rng = __import__("random").Random(seed)
        keep = set()
        for label in (0, 1):
            indices = [i for i, lab in eligible if lab == label]
            rng.shuffle(indices)
            keep.update(indices[:max_crops_per_class])
        annotations = [a for i, a in enumerate(annotations) if i in keep]

    rows, skipped = [], Counter()
    frame_cache = {"id": None, "image": None}
    for ann in annotations:
        label = label_from_category(categories.get(ann.get("category_id"), ""))
        if label is None:
            skipped["unmapped_category"] += 1
            continue
        image_id = ann.get("image_id")
        if image_id not in images:
            skipped["missing_image_record"] += 1
            continue
        filename = images[image_id]
        if frame_cache["id"] != image_id:
            frame_cache = {
                "id": image_id,
                "image": cv2.imread(str(train_folder / filename), cv2.IMREAD_COLOR)
            }
        frame = frame_cache["image"]
        if frame is None:
            skipped["unreadable_image"] += 1
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            skipped["invalid_bbox"] += 1
            continue
        try:
            x, y, w, h = map(float, bbox)
        except (TypeError, ValueError):
            skipped["invalid_bbox"] += 1
            continue
        if w <= 0 or h <= 0:
            skipped["invalid_bbox"] += 1
            continue
        H, W = frame.shape[:2]
        x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
        x2, y2 = min(W, int(round(x + w))), min(H, int(round(y + h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            skipped["empty_or_tiny_crop"] += 1
            continue

        crop = standardize(frame[y1:y2, x1:x2])
        relative_path = Path(CLASS_NAMES[label]) / f"spot_{len(rows):07d}.png"
        destination = output_dir / relative_path
        if not cv2.imwrite(str(destination), crop):
            skipped["write_failure"] += 1
            continue

        info = parse_image_name(filename)
        if info["date"] in date_meta:
            external = date_meta[info["date"]]
            if external.get("weather"):
                info["weather"] = external["weather"].strip().lower()
            if external.get("lot"):
                info["lot"] = external["lot"].strip().upper()

        rows.append({
            "crop_path": relative_path.as_posix(), "label": label,
            "label_name": CLASS_NAMES[label],
            "category": categories.get(ann.get("category_id"), "unknown"),
            "source_image": filename, "frame_id": frame_id_from_name(filename),
            "image_id": image_id, "ann_id": ann.get("id", -1),
            "x": x1, "y": y1, "w": x2-x1, "h": y2-y1, **info
        })

    meta = pd.DataFrame(rows)
    meta.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Saved {len(meta)} crops to {output_dir}")
    if not meta.empty:
        print("Class counts:\n", meta["label_name"].value_counts().to_string())
        print("Distinct frames:", meta["frame_id"].nunique())
    print("Skipped:", dict(skipped))
    if meta.empty:
        raise RuntimeError("No crops were generated. Check dataset paths and COCO category names.")
    return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-folder", default="./raw_dataset/train")
    parser.add_argument("--output-dir", default="./processed_data")
    parser.add_argument("--meta-csv", default=None)
    parser.add_argument("--max-crops-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process_coco_dataset(args.train_folder, args.output_dir, args.meta_csv,
                         args.max_crops_per_class, args.seed)
