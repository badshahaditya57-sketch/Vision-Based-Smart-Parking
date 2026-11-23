"""
evaluate.py -- leakage-aware evaluation of HOG + {SVM, RandomForest, LogisticRegression}

What it does
  1. Samples a class-balanced random subset of crops (seeded).
  2. Computes HOG features (cached).
  3. Evaluates every model under TWO protocols on identical data:
       random  : stratified K-fold over individual crops  (the original protocol; leaky)
       grouped : StratifiedGroupKFold by frame/date, or leave-one-lot-out
  4. Audits leakage: fraction of test crops whose source frame / date also occurs in training.
  5. Reports accuracy, precision/recall/F1 per class, macro-F1, pooled confusion matrices.
  6. Evaluates the grouped out-of-fold predictions separately per weather subset
     (cluster-bootstrap 95% CIs), if weather labels are available.
  7. Writes tables (CSV + Markdown), figures (PNG + PDF) and run_info.json to --out.

Nothing is hard-coded: every number in results/ comes from running this script.

Usage:
  python evaluate.py --data-dir processed_data --max-per-class 2000 --n-splits 5
"""
import argparse
import json
import os
import platform
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC

from common import CLASS_NAMES, HOG_PARAMS, SEED, compute_features, load_samples, resolve_groups


# --------------------------------------------------------------------------
# Models: default hyper-parameters on purpose. Tuning on the test folds would
# itself be a form of leakage; proper tuning needs a nested inner CV (see README).
# --------------------------------------------------------------------------
def make_models(seed):
    return {
        "HOG+SVM (linear)": SVC(kernel="linear", C=1.0),
        "HOG+RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "HOG+LogReg": LogisticRegression(C=1.0, max_iter=2000),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def fold_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    return {
        "accuracy": accuracy_score(y, p),
        "precision_empty": pr[0], "recall_empty": rc[0], "f1_empty": f1[0],
        "precision_occupied": pr[1], "recall_occupied": rc[1], "f1_occupied": f1[1],
        "macro_f1": f1.mean(),
    }


def acc_macro_f1(y, p):
    """Fast accuracy + macro-F1 (used inside the bootstrap loop).
    F1_c = 2TP / (2TP + FP + FN), 0 when TP = 0 (same as zero_division=0)."""
    f = []
    for c in (0, 1):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float((y == p).mean()), float(np.mean(f))


def cluster_bootstrap_ci(y, p, groups, n_boot=1000, seed=0):
    """95% percentile CI for (accuracy, macro-F1), resampling whole GROUPS.
    Crops from one frame/day are strongly correlated, so resampling individual
    crops would give CIs that are far too narrow."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(ug))]
    stats = np.empty((n_boot, 2))
    for b in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        stats[b] = acc_macro_f1(y[idx], p[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return (lo[0], hi[0]), (lo[1], hi[1]), len(ug)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def get_splits(kind, X, y, groups, n_splits, seed, group_mode):
    if kind == "random":
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))
    if group_mode == "lot":
        return list(LeaveOneGroupOut().split(X, y, groups))
    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        raise SystemExit("Need at least 2 groups for a grouped split.")
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, groups))


def leakage_audit(df, splits, groups):
    """% of test crops whose frame / group also appears in the training fold."""
    frame_ov, group_ov = [], []
    frames = df["frame_id"].values
    for tr, te in splits:
        frame_ov.append(np.isin(frames[te], np.unique(frames[tr])).mean())
        group_ov.append(np.isin(groups[te], np.unique(groups[tr])).mean())
    return float(np.mean(frame_ov)), float(np.mean(group_ov))


# --------------------------------------------------------------------------
# Figures / tables
# --------------------------------------------------------------------------
def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def md_table(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def plot_model_comparison(summary, out):
    models = list(summary["model"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.38
    for j, (kind, color) in enumerate([("random", "#9aa5b1"), ("grouped", "#1f77b4")]):
        sub = summary[summary["split"] == kind].set_index("model").loc[models]
        ax.bar(np.arange(len(models)) + (j - 0.5) * w, sub["macro_f1_mean"], w,
               yerr=sub["macro_f1_std"], capsize=3, label=f"{kind} split", color=color)
    ax.set_xticks(np.arange(len(models))); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Macro-F1 (mean ± std over folds)")
    lo = max(0.0, float(summary["macro_f1_mean"].min()) - 0.1)
    ax.set_ylim(lo, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Random crop-level split vs. group-based split")
    savefig(fig, out, "fig_model_comparison")


def plot_confusions(cms, models, out):
    fig, axes = plt.subplots(2, len(models), figsize=(3.3 * len(models), 6))
    axes = np.atleast_2d(axes)
    for i, kind in enumerate(["random", "grouped"]):
        for j, m in enumerate(models):
            cm = cms[(kind, m)]
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{cm[a, b]}\n({norm[a, b]:.1%})", ha="center", va="center",
                            color="white" if norm[a, b] > 0.5 else "black", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES, fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
            ax.set_title(f"{m}\n[{kind}]", fontsize=9)
            if j == 0: ax.set_ylabel("True")
            if i == 1: ax.set_xlabel("Predicted")
    savefig(fig, out, "fig_confusion_matrices")


def plot_weather(wdf, out):
    weathers = [w for w in ["sunny", "cloudy", "rainy"] if w in set(wdf["weather"])] + \
               [w for w in sorted(set(wdf["weather"])) if w not in ("sunny", "cloudy", "rainy")]
    models = list(wdf["model"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        sub = wdf[wdf["model"] == m].set_index("weather").loc[weathers]
        err = np.vstack([sub["macro_f1"] - sub["macro_f1_ci_lo"], sub["macro_f1_ci_hi"] - sub["macro_f1"]])
        ax.bar(np.arange(len(weathers)) + (j - (len(models) - 1) / 2) * w, sub["macro_f1"], w,
               yerr=err, capsize=3, label=m)
    ax.set_xticks(np.arange(len(weathers)))
    ax.set_xticklabels([f"{x}\n(n={int(wdf[wdf.weather == x]['n_crops'].iloc[0])})" for x in weathers])
    ax.set_ylabel("Macro-F1 (grouped out-of-fold)")
    ax.set_ylim(max(0.0, float(wdf["macro_f1_ci_lo"].min()) - 0.05), 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(models))
    ax.set_title("Performance by weather (95% cluster-bootstrap CI)")
    savefig(fig, out, "fig_weather")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./processed_data")
    ap.add_argument("--meta", default=None, help="default: <data-dir>/metadata.csv")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--group-by", default="auto", choices=["auto", "frame", "date", "lot"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(a.seed)

    df = load_samples(a.data_dir, a.meta, a.max_per_class, a.seed)
    print(f"Sampled {len(df)} crops: {df['label_name'].value_counts().to_dict()}")
    X = compute_features(df, a.data_dir, cache_path=os.path.join(a.out, "cache", "features.npz"))
    y = df["label"].values
    groups, group_desc = resolve_groups(df, a.group_by)
    print(f"Feature matrix: {X.shape} | grouping: {group_desc}")
    group_mode = a.group_by if a.group_by != "auto" else group_desc.split(" ")[0]

    models = make_models(a.seed)
    fold_rows, cms, oof = [], {}, {}
    audit = {}
    for kind in ["random", "grouped"]:
        splits = get_splits(kind, X, y, groups, a.n_splits, a.seed, group_mode)
        fo, go = leakage_audit(df, splits, groups)
        audit[kind] = {"test_crops_sharing_frame_with_train": fo, "test_crops_sharing_group_with_train": go}
        print(f"[{kind}] {len(splits)} folds | test crops whose FRAME is also in train: {fo:.1%} "
              f"| whose GROUP is also in train: {go:.1%}")
        if kind == "grouped":
            assert go == 0.0, "group leakage in grouped split!"
        for name, proto in models.items():
            pred = np.full(len(y), -1)
            cm = np.zeros((2, 2), dtype=int)
            for k, (tr, te) in enumerate(splits):
                m = clone(proto)
                t0 = time.perf_counter(); m.fit(X[tr], y[tr]); fit_s = time.perf_counter() - t0
                p = m.predict(X[te])
                pred[te] = p
                cm += confusion_matrix(y[te], p, labels=[0, 1])
                fold_rows.append({"split": kind, "model": name, "fold": k, "n_train": len(tr),
                                  "n_test": len(te), "fit_seconds": fit_s, **fold_metrics(y[te], p)})
            cms[(kind, name)] = cm
            oof[(kind, name)] = pred
            r = [x for x in fold_rows if x["split"] == kind and x["model"] == name]
            print(f"   {name:18s} acc={np.mean([x['accuracy'] for x in r]):.4f} "
                  f"macroF1={np.mean([x['macro_f1'] for x in r]):.4f}")

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(os.path.join(a.out, "fold_metrics.csv"), index=False)

    metric_cols = ["accuracy", "precision_empty", "recall_empty", "f1_empty",
                   "precision_occupied", "recall_occupied", "f1_occupied", "macro_f1"]
    agg = folds.groupby(["split", "model"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    summary = agg.reset_index()
    summary.to_csv(os.path.join(a.out, "summary_metrics.csv"), index=False)

    # Compact Markdown table (mean ± std over folds)
    pretty = pd.DataFrame({
        "split": summary["split"], "model": summary["model"],
        **{c: [f"{m:.3f} ± {s:.3f}" for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_std"].fillna(0))]
           for c in ["accuracy", "precision_occupied", "recall_occupied", "f1_occupied", "macro_f1"]},
    })
    with open(os.path.join(a.out, "results_table.md"), "w") as f:
        f.write(f"Grouping for the 'grouped' split: {group_desc}\n\n")
        f.write(md_table(pretty) + "\n\n")
        f.write("Leakage audit (mean over folds):\n\n")
        f.write(md_table(pd.DataFrame([{"split": k, **v} for k, v in audit.items()]), "{:.3f}") + "\n")
    pd.DataFrame([{"split": k[0], "model": k[1], "TN": int(v[0, 0]), "FP": int(v[0, 1]),
                   "FN": int(v[1, 0]), "TP": int(v[1, 1])} for k, v in cms.items()]
                 ).to_csv(os.path.join(a.out, "confusion_matrices.csv"), index=False)
    print("\nWrote summary_metrics.csv, fold_metrics.csv, results_table.md, confusion_matrices.csv")

    plot_model_comparison(summary, a.out)
    plot_confusions(cms, list(models), a.out)

    # ---- Weather subsets (grouped out-of-fold predictions) ----
    known = df["weather"] != "unknown"
    weather_note = None
    if known.sum() == 0:
        weather_note = "Weather labels could not be recovered; weather analysis skipped (see README)."
        print("\n" + weather_note)
    else:
        rows = []
        for name in models:
            p_all = oof[("grouped", name)]
            for wth in sorted(df.loc[known, "weather"].unique()):
                mask = (df["weather"] == wth).values
                yy, pp, gg = y[mask], p_all[mask], groups[mask]
                (alo, ahi), (flo, fhi), ng = cluster_bootstrap_ci(yy, pp, gg, a.n_boot, a.seed)
                acc, f1 = acc_macro_f1(yy, pp)
                rows.append({"model": name, "weather": wth, "n_crops": int(mask.sum()), "n_groups": ng,
                             "accuracy": acc, "accuracy_ci_lo": alo, "accuracy_ci_hi": ahi,
                             "macro_f1": f1, "macro_f1_ci_lo": flo, "macro_f1_ci_hi": fhi})
        wdf = pd.DataFrame(rows)
        wdf.to_csv(os.path.join(a.out, "weather_metrics.csv"), index=False)
        with open(os.path.join(a.out, "weather_table.md"), "w") as f:
            f.write(md_table(wdf, "{:.3f}") + "\n")
        plot_weather(wdf, a.out)
        if (wdf["n_groups"] < 5).any():
            print("WARNING: some weather subsets have <5 groups; their CIs are unreliable.")
        print("Wrote weather_metrics.csv, weather_table.md, fig_weather.*")

    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), "args": vars(a),
        "n_samples": int(len(df)), "class_counts": df["label_name"].value_counts().to_dict(),
        "n_frames": int(df["frame_id"].nunique()), "grouping": group_desc,
        "leakage_audit": audit, "hog_params": {k: list(v) if isinstance(v, tuple) else v for k, v in HOG_PARAMS.items()},
        "feature_dim": int(X.shape[1]), "weather_counts": df["weather"].value_counts().to_dict(),
        "weather_note": weather_note,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    with open(os.path.join(a.out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print("Wrote run_info.json. Done.")


if __name__ == "__main__":
    main()
