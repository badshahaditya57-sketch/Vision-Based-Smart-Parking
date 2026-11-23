"""
common.py -- single source of truth for preprocessing, HOG parameters,
metadata handling and group (leakage-aware) logic.

Training, evaluation, latency benchmarking and the Streamlit app all import
from here, so a crop is guaranteed to go through the *same* transformation
everywhere. (In the original code the app did grayscale->resize while
preprocess.py did resize->grayscale, a small train/serve mismatch.)

This module deliberately does NOT import scikit-learn, so the Streamlit app
does not pick up any extra dependency.
"""
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
TARGET_W, TARGET_H = 64, 128           # cv2 uses (width, height)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)
CLASS_NAMES = ["empty", "occupied"]    # label 0 = empty, label 1 = occupied
SEED = 42


# --------------------------------------------------------------------------
# Image -> feature vector
# --------------------------------------------------------------------------
def to_gray(img):
    """BGR (H,W,3) -> gray (H,W). Gray input is returned unchanged."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def standardize(img):
    """Grayscale + resize to 64x128. INTER_AREA is the recommended
    interpolation for shrinking (it averages source pixels, which reduces
    aliasing). No-op if the image already has the target shape."""
    gray = to_gray(img)
    if gray.shape != (TARGET_H, TARGET_W):
        gray = cv2.resize(gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return gray


def hog_features(gray):
    """HOG vector of a standardized 64x128 grayscale image (length 3780)."""
    return hog(gray, **HOG_PARAMS)


def extract_from_image(img):
    """Raw BGR/gray image -> HOG vector. Used by the app."""
    return hog_features(standardize(img))


# --------------------------------------------------------------------------
# Filename -> metadata (best effort; never guesses)
# --------------------------------------------------------------------------
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_LOT_RE = re.compile(r"(PUCPR|PUC|UFPR04|UFPR05)", re.IGNORECASE)
_WEATHER_RE = re.compile(r"(sunny|cloudy|overcast|rainy|rain)", re.IGNORECASE)
_WEATHER_ALIAS = {"overcast": "cloudy", "rain": "rainy"}
# Roboflow export names look like  <orig>_jpg.rf.<hash>.jpg
_ROBOFLOW_RE = re.compile(r"^(?P<base>.*?)(?:_(?:jpg|jpeg|png))?\.rf\.[0-9a-fA-F]+\.[A-Za-z]+$")


def frame_id_from_name(file_name):
    """Identifier of the ORIGINAL camera frame.
    Roboflow can export several augmented copies of one frame with different
    hashes; stripping '.rf.<hash>' maps them back to the same frame_id so they
    can be kept on the same side of a split."""
    name = os.path.basename(file_name)
    m = _ROBOFLOW_RE.match(name)
    if m:
        return m.group("base")
    return os.path.splitext(name)[0]


def parse_image_name(file_name):
    """Extract date / lot / weather if they are encoded in the filename,
    otherwise 'unknown'. Never invents a value."""
    name = os.path.basename(file_name)
    date = _DATE_RE.search(name)
    lot = _LOT_RE.search(name)
    weather = _WEATHER_RE.search(name)
    w = weather.group(1).lower() if weather else "unknown"
    w = _WEATHER_ALIAS.get(w, w)
    return {
        "date": date.group(1) if date else "unknown",
        "lot": lot.group(1).upper() if lot else "unknown",
        "weather": w,
    }


# --------------------------------------------------------------------------
# Loading, sampling, features
# --------------------------------------------------------------------------
def load_samples(data_dir, meta_path=None, max_per_class=2000, seed=SEED):
    """Read metadata.csv and draw a class-balanced RANDOM sample.

    (The original train_svm.py took the first N files of os.listdir, which is
    an arbitrary, possibly clustered subset -- not a random sample.)
    max_per_class=0 or None means "use everything".
    """
    meta_path = meta_path or os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        sys.exit(
            f"ERROR: {meta_path} not found. Re-run preprocess.py (the new version "
            "writes metadata.csv, which is required for group-aware evaluation)."
        )
    df = pd.read_csv(meta_path, keep_default_na=False)
    counts = df["label"].value_counts()
    n = int(counts.min())
    if max_per_class:
        n = min(n, int(max_per_class))
    parts = [g.sample(n=n, random_state=seed) for _, g in df.groupby("label")]
    out = pd.concat(parts).sort_values("crop_path").reset_index(drop=True)
    return out


def compute_features(df, data_dir, dtype=np.float32, cache_path=None, verbose=True):
    """HOG matrix for every crop in df (row order preserved).
    If cache_path is given, features are cached and reused only when the exact
    same list of crops and HOG parameters are requested."""
    key = np.array(list(df["crop_path"]) + [repr(sorted(HOG_PARAMS.items())), str(dtype)])
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if z["key"].shape == key.shape and np.array_equal(z["key"], key):
            if verbose:
                print(f"Loaded cached features from {cache_path}")
            return z["X"]

    feats = []
    total = len(df)
    for i, rel in enumerate(df["crop_path"]):
        img = cv2.imread(os.path.join(data_dir, rel), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read crop: {rel}")
        feats.append(hog_features(standardize(img)).astype(dtype, copy=False))
        if verbose and (i + 1) % 1000 == 0:
            print(f"  HOG features: {i + 1}/{total}")
    X = np.stack(feats)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, X=X, key=key)
    return X


# --------------------------------------------------------------------------
# Groups (the heart of leakage prevention)
# --------------------------------------------------------------------------
def resolve_groups(df, group_by="auto", min_groups=5):
    """Return (groups: np.ndarray[str], description: str).

    group_by:
      'frame' : one group per original camera frame (always available).
                Stops crops from the SAME frame landing in train and test.
                Does NOT stop consecutive frames of the same car from doing so.
      'date'  : one group per recording day (parsed from filename).
                Much stronger: a whole day is held out together. Rows whose
                date is unknown fall back to their frame group.
      'lot'   : one group per parking lot (leave-one-lot-out).
      'auto'  : 'date' if >= min_groups distinct dates were parsed, else 'frame'.
    """
    frame = df["frame_id"].astype(str)
    if group_by == "auto":
        n_dates = df.loc[df["date"] != "unknown", "date"].nunique()
        group_by = "date" if n_dates >= min_groups else "frame"

    if group_by == "frame":
        return frame.values, f"frame ({frame.nunique()} groups)"
    if group_by == "date":
        g = np.where(df["date"] != "unknown", "date:" + df["date"], "frame:" + frame)
        n_unknown = int((df["date"] == "unknown").sum())
        extra = f", {n_unknown} crops had no parsable date and fell back to frame groups" if n_unknown else ""
        return g, f"date ({len(set(g))} groups{extra})"
    if group_by == "lot":
        if (df["lot"] == "unknown").any():
            sys.exit("ERROR: --group-by lot requires a known lot for every crop "
                     "(see --meta-csv in preprocess.py).")
        return df["lot"].values, f"lot ({df['lot'].nunique()} groups)"
    sys.exit(f"Unknown group_by: {group_by}")
