"""Shared preprocessing, HOG extraction, metadata grouping and dataset loading."""
from pathlib import Path
import hashlib
import json
import re

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

TARGET_W = 64
TARGET_H = 128
SEED = 42
CLASS_NAMES = ["empty", "occupied"]
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": False,
    "feature_vector": True,
}

def to_gray(image):
    if image is None:
        raise ValueError("Input image is None or could not be decoded.")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def standardize(image):
    gray = to_gray(image)
    return cv2.resize(gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

def hog_features(image):
    # Avoid resizing twice when the caller already supplies a standardized ROI.
    if image is not None and image.ndim == 2 and image.shape == (TARGET_H, TARGET_W):
        gray = image
    else:
        gray = standardize(image)
    return hog(gray, **HOG_PARAMS).astype(np.float32)

def extract_from_image(image):
    standardized = standardize(image)
    return standardized, hog(standardized, **HOG_PARAMS).astype(np.float32)

def frame_id_from_name(filename):
    """Stable frame key; strips common Roboflow augmentation suffixes."""
    name = Path(str(filename)).name
    name = re.sub(r"_jpg\.rf\.[^.]+$", "", name, flags=re.IGNORECASE)
    return Path(name).stem

def parse_image_name(filename):
    """Best-effort metadata extraction from filename only; unknown is explicit."""
    name = Path(str(filename)).stem
    date_match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", name)
    date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else "unknown"
    upper = name.upper()
    lot = next((x for x in ("PUCPR", "PUC", "UFPR04", "UFPR05") if x in upper), "unknown")
    low = name.lower()
    if "sunny" in low:
        weather = "sunny"
    elif "cloudy" in low or "overcast" in low:
        weather = "cloudy"
    elif "rainy" in low or "rain" in low:
        weather = "rainy"
    else:
        weather = "unknown"
    return {"date": date, "lot": lot, "weather": weather}

def load_samples(data_dir, metadata_path=None, max_per_class=2000, seed=SEED):
    data_dir = Path(data_dir)
    meta_path = Path(metadata_path) if metadata_path else data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    df = pd.read_csv(meta_path)
    required = {"crop_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    if max_per_class and max_per_class > 0:
        parts = []
        for label in (0, 1):
            subset = df[df.label == label]
            if len(subset) > max_per_class:
                subset = subset.sample(n=max_per_class, random_state=seed)
            parts.append(subset)
        df = pd.concat(parts, ignore_index=True)
    if df.empty or df["label"].nunique() != 2:
        raise ValueError("Need at least one sample from each class (empty=0, occupied=1).")
    if "label_name" not in df:
        df["label_name"] = df["label"].map({0: "empty", 1: "occupied"})
    return df.reset_index(drop=True)

def _resolve_group_series(df, mode):
    if mode == "frame":
        col = "frame_id"
    elif mode == "date":
        col = "date"
    elif mode == "lot":
        col = "lot"
    else:
        raise ValueError(f"Unsupported group mode: {mode}")
    if col not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index), col
    values = df[col].fillna("unknown").astype(str)
    # Unknown metadata is isolated by frame rather than all pooled together.
    if mode != "frame":
        fallback = df.get("frame_id", pd.Series(df.index.astype(str), index=df.index)).fillna("unknown").astype(str)
        values = values.where(~values.str.lower().isin(["", "unknown", "nan", "none"]), "frame:" + fallback)
    return values, col

def resolve_groups(df, mode="auto"):
    if mode == "auto":
        dates = df["date"].astype(str) if "date" in df else pd.Series(dtype=str)
        lots = df["lot"].astype(str) if "lot" in df else pd.Series(dtype=str)
        if len(dates) and dates.replace({"unknown": np.nan}).nunique() >= 5:
            mode = "date"
        elif len(lots) and lots.replace({"unknown": np.nan}).nunique() >= 2 and (lots != "unknown").all():
            mode = "lot"
        else:
            mode = "frame"
    groups, col = _resolve_group_series(df, mode)
    if groups.nunique() < 2:
        raise ValueError(f"Grouping by {mode!r} produced fewer than two groups. "
                         "Use --group-by frame or provide reliable date/lot metadata.")
    return groups.to_numpy(), f"{mode} ({groups.nunique()} groups; source={col})"

def compute_features(df, data_dir, cache_path=None):
    data_dir = Path(data_dir)
    paths = [str((data_dir / str(p)).resolve()) for p in df["crop_path"]]
    signature = hashlib.sha256()
    for p in paths:
        path = Path(p)
        signature.update(str(path).encode())
        if path.exists():
            stat = path.stat()
            signature.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    signature.update(json.dumps(HOG_PARAMS, sort_keys=True).encode())
    key = signature.hexdigest()
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        try:
            saved = np.load(cache, allow_pickle=False)
            if str(saved["key"].item()) == key and len(saved["X"]) == len(paths):
                return saved["X"].astype(np.float32)
        except Exception:
            pass
    features = []
    for p in paths:
        image = cv2.imread(p, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read crop: {p}")
        features.append(hog_features(image))
    X = np.vstack(features).astype(np.float32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, key=np.array(key))
    return X
