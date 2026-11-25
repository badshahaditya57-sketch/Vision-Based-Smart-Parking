
"""
Lightweight CNN for Vision-Based Smart Parking.

Evaluation:
- 5-fold StratifiedGroupKFold using date groups
- Inner group-based validation for early stopping
- Out-of-fold predictions on outer test folds
- Accuracy, precision, recall, macro-F1 and confusion matrix

Run:
    python train_cnn.py
"""

import copy
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupShuffleSplit,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed_data"
METADATA_PATH = DATA_DIR / "metadata.csv"

OUTPUT_DIR = ROOT / "outputs" / "cnn_date_cv"

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 128

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 5

N_SPLITS = 5
RANDOM_SEED = 42

CLASS_NAMES = ["empty", "occupied"]
NUM_CLASSES = 2

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class ParkingDataset(Dataset):

    def __init__(self, paths, labels):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_path = self.paths[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        image = image.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)

        image_tensor = torch.from_numpy(image)

        label_tensor = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return image_tensor, label_tensor


# ============================================================
# LIGHTWEIGHT CNN
# ============================================================

class ParkingCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.35),

            nn.Linear(128, 64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.20),

            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = [
        "crop_path",
        "label",
        "date"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing metadata column: {column}"
            )

    df = df.copy()

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["label", "date", "crop_path"]
    ).reset_index(drop=True)

    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Expected labels 0 and 1 only."
        )

    df["date"] = df["date"].astype(str)

    df["image_path"] = df["crop_path"].apply(
        lambda p: DATA_DIR / str(p)
    )

    missing = [
        str(p)
        for p in df["image_path"]
        if not p.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing. "
            f"First missing image: {missing[0]}"
        )

    y = df["label"].to_numpy(dtype=np.int64)

    groups = df["date"].to_numpy()

    if len(np.unique(groups)) < N_SPLITS:
        raise ValueError(
            "Not enough unique dates for 5-fold CV."
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Dataset must contain both classes."
        )

    print("\nDATASET SUMMARY")
    print("------------------------------")
    print("Total images:", len(df))
    print("Empty:", int(np.sum(y == 0)))
    print("Occupied:", int(np.sum(y == 1)))
    print("Unique dates:", len(np.unique(groups)))
    print("Missing images:", len(missing))

    return df, y, groups


# ============================================================
# INNER GROUP VALIDATION SPLIT
# ============================================================

def create_inner_split(indices, y, groups, seed):

    X_dummy = np.zeros(len(indices))

    splitter = GroupShuffleSplit(
        n_splits=40,
        test_size=0.20,
        random_state=seed
    )

    for train_local, val_local in splitter.split(
        X_dummy,
        y[indices],
        groups[indices]
    ):

        train_indices = indices[train_local]
        val_indices = indices[val_local]

        train_groups = set(groups[train_indices])
        val_groups = set(groups[val_indices])

        if train_groups.intersection(val_groups):
            continue

        if len(np.unique(y[train_indices])) != 2:
            continue

        if len(np.unique(y[val_indices])) != 2:
            continue

        return train_indices, val_indices

    raise RuntimeError(
        "Could not create a valid inner date-grouped "
        "training/validation split. Check date-wise "
        "class distribution."
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy_score(
            y_true, y_pred
        ),

        "macro_precision": np.mean(precision),

        "macro_recall": np.mean(recall),

        "macro_f1": np.mean(f1),

        "empty_precision": precision[0],

        "empty_recall": recall[0],

        "empty_f1": f1[0],

        "occupied_precision": precision[1],

        "occupied_recall": recall[1],

        "occupied_f1": f1[1],
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

    return total_loss / max(total_samples, 1)


# ============================================================
# EVALUATION DURING TRAINING
# ============================================================

def evaluate_model(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = total_loss / max(total_samples, 1)

    metrics = calculate_metrics(
        np.asarray(all_labels),
        np.asarray(all_predictions)
    )

    return average_loss, metrics


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold,
    train_indices,
    val_indices,
    test_indices,
    df,
    y
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 60)

    train_dataset = ParkingDataset(
        df.iloc[train_indices]["image_path"],
        y[train_indices]
    )

    val_dataset = ParkingDataset(
        df.iloc[val_indices]["image_path"],
        y[val_indices]
    )

    test_dataset = ParkingDataset(
        df.iloc[test_indices]["image_path"],
        y[test_indices]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Calculate class weights using inner training data only.
    counts = np.bincount(
        y[train_indices],
        minlength=NUM_CLASSES
    )

    class_weights = len(train_indices) / (
        NUM_CLASSES * np.maximum(counts, 1)
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE
    )

    model = ParkingCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_metrics = evaluate_model(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        current_f1 = val_metrics["macro_f1"]

        history.append({
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": current_f1,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {current_f1:.4f}"
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print("Early stopping triggered.")

            break

    if best_state is None:
        raise RuntimeError(
            f"No valid model checkpoint for fold {fold}."
        )

    model.load_state_dict(best_state)

    # Evaluate outer test fold only after model selection.
    model.eval()

    test_labels = []
    test_predictions = []
    test_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            test_labels.extend(
                labels.numpy()
            )

            test_predictions.extend(
                predictions.cpu().numpy()
            )

            test_probabilities.extend(
                probabilities.cpu().numpy()
            )

    test_labels = np.asarray(test_labels)
    test_predictions = np.asarray(test_predictions)
    test_probabilities = np.asarray(test_probabilities)

    metrics = calculate_metrics(
        test_labels,
        test_predictions
    )

    elapsed = time.time() - start_time

    print("\nOUTER TEST RESULTS")
    print("------------------------------")

    for name, value in metrics.items():

        print(f"{name}: {value:.4f}")

    print("Best epoch:", best_epoch)
    print(f"Training time: {elapsed:.2f} seconds")

    checkpoint_path = (
        OUTPUT_DIR / f"cnn_fold_{fold}.pth"
    )

    torch.save(
        {
            "fold": fold,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "class_names": CLASS_NAMES,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
        },
        checkpoint_path
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_DIR / f"fold_{fold}_history.csv",
        index=False
    )

    prediction_df = df.iloc[test_indices][
        [
            "crop_path",
            "source_image",
            "date",
            "frame_id"
        ]
    ].copy()

    prediction_df["fold"] = fold

    prediction_df["true_label"] = test_labels

    prediction_df["true_class"] = [
        CLASS_NAMES[i] for i in test_labels
    ]

    prediction_df["predicted_label"] = test_predictions

    prediction_df["predicted_class"] = [
        CLASS_NAMES[i] for i in test_predictions
    ]

    prediction_df["prob_empty"] = test_probabilities[:, 0]

    prediction_df["prob_occupied"] = test_probabilities[:, 1]

    prediction_df["correct"] = (
        test_labels == test_predictions
    )

    return (
        metrics,
        prediction_df,
        best_epoch,
        elapsed
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    torch.set_num_threads(2)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("DEVICE:", DEVICE)
    print("PyTorch:", torch.__version__)
    print("Output:", OUTPUT_DIR)

    df, y, groups = load_metadata()

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    fold_metrics = []
    all_predictions = []

    fold_details = []

    # Generate outer folds once.
    outer_splits = list(
        splitter.split(
            np.zeros(len(df)),
            y,
            groups
        )
    )

    for fold, (outer_train, outer_test) in enumerate(
        outer_splits,
        start=1
    ):

        train_groups = set(groups[outer_train])
        test_groups = set(groups[outer_test])

        if train_groups.intersection(test_groups):
            raise RuntimeError(
                f"Date leakage detected in fold {fold}."
            )

        if len(np.unique(y[outer_train])) != 2:
            raise RuntimeError(
                f"Training fold {fold} lacks a class."
            )

        inner_train, inner_val = create_inner_split(
            outer_train,
            y,
            groups,
            RANDOM_SEED + fold
        )

        # Verify inner group separation.
        if set(groups[inner_train]).intersection(
            set(groups[inner_val])
        ):
            raise RuntimeError(
                f"Inner date leakage in fold {fold}."
            )

        if set(groups[inner_train]).intersection(
            test_groups
        ):
            raise RuntimeError(
                f"Training/test date leakage in fold {fold}."
            )

        if set(groups[inner_val]).intersection(
            test_groups
        ):
            raise RuntimeError(
                f"Validation/test date leakage in fold {fold}."
            )

        print(
            f"\nFold {fold}: "
            f"Train={len(inner_train)}, "
            f"Validation={len(inner_val)}, "
            f"Test={len(outer_test)}"
        )

        print(
            "Unique dates - "
            f"Train: {len(set(groups[inner_train]))}, "
            f"Validation: {len(set(groups[inner_val]))}, "
            f"Test: {len(test_groups)}"
        )

        metrics, predictions, best_epoch, elapsed = train_fold(
            fold,
            inner_train,
            inner_val,
            outer_test,
            df,
            y
        )

        metrics["fold"] = fold
        metrics["best_epoch"] = best_epoch
        metrics["training_seconds"] = elapsed

        fold_metrics.append(metrics)

        all_predictions.append(predictions)

        fold_details.append({
            "fold": fold,
            "outer_train_samples": len(outer_train),
            "inner_train_samples": len(inner_train),
            "inner_validation_samples": len(inner_val),
            "outer_test_samples": len(outer_test),
            "inner_train_dates": len(set(groups[inner_train])),
            "inner_validation_dates": len(set(groups[inner_val])),
            "outer_test_dates": len(test_groups),
            "best_epoch": best_epoch,
            "training_seconds": elapsed
        })

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(fold_metrics)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    details_df = pd.DataFrame(fold_details)

    metrics_df.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        index=False
    )

    details_df.to_csv(
        OUTPUT_DIR / "fold_details.csv",
        index=False
    )

    # --------------------------------------------------------
    # POOLED METRICS
    # --------------------------------------------------------

    true_labels = predictions_df[
        "true_label"
    ].to_numpy()

    predicted_labels = predictions_df[
        "predicted_label"
    ].to_numpy()

    pooled_metrics = calculate_metrics(
        true_labels,
        predicted_labels
    )

    pooled_cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        pooled_cm,
        index=["Actual_empty", "Actual_occupied"],
        columns=["Predicted_empty", "Predicted_occupied"]
    )

    cm_df.to_csv(
        OUTPUT_DIR / "pooled_confusion_matrix.csv"
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {}

    for metric in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1"
    ]:

        summary[metric + "_mean"] = float(
            metrics_df[metric].mean()
        )

        summary[metric + "_std"] = float(
            metrics_df[metric].std(ddof=1)
        )

    summary["pooled_accuracy"] = float(
        pooled_metrics["accuracy"]
    )

    summary["pooled_macro_f1"] = float(
        pooled_metrics["macro_f1"]
    )

    summary["total_samples"] = int(
        len(predictions_df)
    )

    summary["total_errors"] = int(
        (~predictions_df["correct"]).sum()
    )

    summary["total_training_seconds"] = float(
        metrics_df["training_seconds"].sum()
    )

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # --------------------------------------------------------

    config = {
        "model": "Lightweight ParkingCNN",
        "input_shape": [
            1,
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ],
        "classes": CLASS_NAMES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "optimizer": "AdamW",
        "loss": "Class-weighted CrossEntropyLoss",
        "outer_validation": "StratifiedGroupKFold",
        "inner_validation": "GroupShuffleSplit",
        "group_column": "date",
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "pytorch_version": torch.__version__,
        "note": (
            "Outer folds are date-grouped. "
            "Inner date-grouped validation is used for "
            "early stopping. Results are cross-validation "
            "estimates, not an independent external test."
        )
    }

    with open(
        OUTPUT_DIR / "experiment_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CNN CROSS-VALIDATION COMPLETE")
    print("=" * 60)

    print("\nFOLD METRICS:")
    print(metrics_df.to_string(index=False))

    print("\nPOOLED METRICS:")

    for name, value in pooled_metrics.items():

        print(f"{name}: {value:.4f}")

    print("\nPOOLED CONFUSION MATRIX:")
    print(cm_df)

    print("\nSUMMARY:")
    for name, value in summary.items():
        print(f"{name}: {value}")

    print("\nSAVED OUTPUTS:")
    print(OUTPUT_DIR)

    print("\nFiles:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print("-", path.name)


if __name__ == "__main__":
    main()
