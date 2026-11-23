"""
benchmark_latency.py -- measured (not assumed) inference latency per ROI.

Method
  * single thread (BLAS/OpenMP thread counts forced to 1, RandomForest n_jobs=1)
  * batch size 1 (one parking space at a time = the real use case)
  * images are pre-loaded, so disk I/O is excluded
  * warm-up calls are discarded; every crop is timed individually with
    time.perf_counter_ns() over several repetitions
  * reports median, mean, std, p95, p99 per stage and end-to-end
  * extra row: linear SVM evaluated as the single dot product w.x + b
    (valid because a linear-kernel SVM has w = sum_i alpha_i y_i x_i); the
    script checks it gives the same predictions as SVC.predict.

Limitation: crops in processed_data are ALREADY 64x128 grayscale, so the
'standardize' stage here is a lower bound (a deployed system would resize a
larger camera ROI). Numbers are specific to the machine they were run on.

Usage:
  python benchmark_latency.py --data-dir processed_data --n-train 2000 --n-eval 300
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # must be set BEFORE numpy is imported

import argparse
import gc
import json
import pickle
import platform
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from common import SEED, compute_features, hog_features, load_samples, standardize


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def time_calls(fn, inputs, repeats, warmup):
    for a in inputs[:warmup]:
        fn(a)
    out = []
    gc_was = gc.isenabled(); gc.disable()
    try:
        for _ in range(repeats):
            for a in inputs:
                t0 = time.perf_counter_ns(); fn(a); out.append(time.perf_counter_ns() - t0)
    finally:
        if gc_was: gc.enable()
    return np.asarray(out) / 1e6      # ms


def describe(ms):
    return {"n": int(ms.size), "median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()),
            "std_ms": float(ms.std()), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./processed_data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--n-train", type=int, default=2000, help="crops used to fit the models (timing only)")
    ap.add_argument("--n-eval", type=int, default=300, help="crops timed")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Fit models on float64 features (what hog() returns at inference time)
    train_df = load_samples(a.data_dir, None, a.n_train // 2, a.seed)
    Xtr = compute_features(train_df, a.data_dir, dtype=np.float64, verbose=False)
    ytr = train_df["label"].values

    models = {
        "HOG+SVM (SVC.predict)": SVC(kernel="linear", C=1.0),
        "HOG+RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=a.seed),
        "HOG+LogReg": LogisticRegression(C=1.0, max_iter=2000),
    }
    for m in models.values():
        m.fit(Xtr, ytr)
    svm = models["HOG+SVM (SVC.predict)"]
    w = svm.coef_.ravel().astype(np.float64); b = float(svm.intercept_[0])

    # Timing inputs: pre-loaded grayscale crops
    eval_df = load_samples(a.data_dir, None, a.n_eval // 2, a.seed + 1)
    imgs = [cv2.imread(os.path.join(a.data_dir, p), cv2.IMREAD_GRAYSCALE) for p in eval_df["crop_path"]]
    feats = [hog_features(standardize(im)) for im in imgs]
    feats_2d = [f[None, :] for f in feats]

    # Sanity: dot-product SVM must reproduce SVC.predict
    agree = float(np.mean(svm.predict(np.stack(feats)) == (np.stack(feats) @ w + b > 0).astype(int)))
    print(f"Linear-SVM w.x+b agrees with SVC.predict on {agree:.2%} of eval crops")

    rows = []
    def add(stage, ms): rows.append({"stage": stage, **describe(ms)})
    kw = dict(repeats=a.repeats, warmup=a.warmup)

    add("standardize (gray+resize)", time_calls(standardize, imgs, **kw))
    add("hog_features", time_calls(hog_features, [standardize(im) for im in imgs], **kw))
    for name, m in models.items():
        add(f"predict only: {name}", time_calls(lambda x, m=m: m.predict(x), feats_2d, **kw))
        add(f"END-TO-END: {name}",
            time_calls(lambda im, m=m: m.predict(hog_features(standardize(im))[None, :]), imgs, **kw))
    add("predict only: SVM as w.x+b", time_calls(lambda x: x @ w + b > 0, feats, **kw))
    add("END-TO-END: SVM as w.x+b", time_calls(lambda im: hog_features(standardize(im)) @ w + b > 0, imgs, **kw))

    lat = pd.DataFrame(rows)
    lat.to_csv(os.path.join(a.out, "latency.csv"), index=False)
    print(lat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    sizes = {n: round(len(pickle.dumps(m)) / 1024, 1) for n, m in models.items()}
    e2e = lat[lat["stage"] == "END-TO-END: HOG+SVM (SVC.predict)"].iloc[0]
    info = {
        "svm_end_to_end_median_ms": float(e2e["median_ms"]), "svm_end_to_end_p95_ms": float(e2e["p95_ms"]),
        "n_timed_calls_per_stage": int(e2e["n"]), "dot_product_agreement": agree,
        "svm_n_support_vectors": int(svm.n_support_.sum()), "feature_dim": int(Xtr.shape[1]),
        "model_pickle_kb": sizes, "n_train": int(len(train_df)), "n_eval": int(len(imgs)),
        "hardware": {"cpu": cpu_name(), "cores": os.cpu_count(), "platform": platform.platform(), "threads": 1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__, "sklearn": sklearn.__version__},
    }
    with open(os.path.join(a.out, "latency.json"), "w") as f:
        json.dump(info, f, indent=2)

    e = lat[lat["stage"].str.startswith("END-TO-END")].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(e))
    ax.barh(y, e["median_ms"], xerr=[np.zeros(len(e)), e["p95_ms"] - e["median_ms"]], capsize=3, color="#1f77b4")
    ax.set_yticks(y); ax.set_yticklabels([s.replace("END-TO-END: ", "") for s in e["stage"]], fontsize=9)
    ax.set_xlabel("Latency per ROI (ms): bar = median, whisker = p95")
    ax.set_title(f"End-to-end latency, single thread\n{info['hardware']['cpu']}", fontsize=9)
    ax.grid(axis="x", alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_latency.png"), dpi=200); fig.savefig(os.path.join(a.out, "fig_latency.pdf"))
    print("\nWrote latency.csv, latency.json, fig_latency.*")


if __name__ == "__main__":
    main()
