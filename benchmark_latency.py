"""Single-thread CPU latency benchmark for one parking-space ROI.

Example:
  python benchmark_latency.py --data-dir processed_data --model-path hog_svm_parking.pkl
Reports preprocessing, HOG, SVM prediction and total latency separately.
"""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from common import compute_features, hog_features, load_samples, standardize

def summarize(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(np.mean(a)), "median_ms": float(np.median(a)),
        "std_ms": float(np.std(a)), "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99))
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./processed_data")
    ap.add_argument("--model-path", default="./hog_svm_parking.pkl")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be >=1 and warmup >=0")
    model = joblib.load(args.model_path)
    df = load_samples(args.data_dir, None, max_per_class=1)
    image_path = Path(args.data_dir) / str(df.iloc[0]["crop_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    prep_times, hog_times, predict_times, total_times = [], [], [], []
    with threadpool_limits(limits=1):
        for i in range(args.warmup + args.iterations):
            t0 = time.perf_counter_ns()
            standardized = standardize(image)
            t1 = time.perf_counter_ns()
            feature = hog_features(standardized)
            t2 = time.perf_counter_ns()
            _ = model.predict(feature.reshape(1, -1))
            t3 = time.perf_counter_ns()
            if i >= args.warmup:
                prep_times.append((t1-t0)/1e6)
                hog_times.append((t2-t1)/1e6)
                predict_times.append((t3-t2)/1e6)
                total_times.append((t3-t0)/1e6)

    result = {
        "iterations": args.iterations, "warmup": args.warmup,
        "preprocessing": summarize(prep_times), "hog": summarize(hog_times),
        "svm_prediction": summarize(predict_times), "end_to_end": summarize(total_times),
        "svm_end_to_end_median_ms": float(np.median(total_times)),
        "svm_end_to_end_p95_ms": float(np.percentile(total_times, 95)),
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor() or "unknown",
            "cpu": platform.processor() or platform.machine(), "opencv_threads": cv2.getNumThreads(),
            "threadpool_limit": 1
        },
        "measurement_note": "One preloaded crop; excludes disk I/O, camera capture, network, and Streamlit rendering."
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "latency.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
