
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from train_cnn import ParkingCNN


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed_data"
METADATA_PATH = DATA_DIR / "metadata.csv"

OUTPUT_DIR = ROOT / "outputs" / "cnn_final"

MODEL_PATH = OUTPUT_DIR / "final_model.pth"

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 128

BATCH_SIZE = 32
EPOCHS = 9

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

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


# ============================================================
# DATASET
# ============================================================

class FinalParkingDataset(Dataset):

    def __init__(self, paths, labels):

        self.paths = list(paths)

        self.labels = np.asarray(
            labels,
            dtype=np.int64
        )

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
# LOAD ALL DATA
# ============================================================

def load_all_data():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    required = [
        "crop_path",
        "label",
        "date"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metadata columns: {missing}"
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["label", "date", "crop_path"]
    ).copy()

    df = df[
        df["label"].isin([0, 1])
    ].copy()

    df["label"] = df["label"].astype(int)

    df = df.reset_index(drop=True)

    df["image_path"] = df["crop_path"].apply(
        lambda path: DATA_DIR / str(path)
    )

    missing_images = [
        str(path)
        for path in df["image_path"]
        if not path.exists()
    ]

    if missing_images:

        raise FileNotFoundError(
            f"{len(missing_images)} image files are missing. "
            f"First missing file: {missing_images[0]}"
        )

    paths = df["image_path"].tolist()

    labels = df["label"].to_numpy(
        dtype=np.int64
    )

    return df, paths, labels


# ============================================================
# TRAINING
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    torch.set_num_threads(2)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("FINAL CNN TRAINING")
    print("=" * 65)

    df, paths, labels = load_all_data()

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    print(f"Total images: {len(paths)}")
    print(f"Empty images: {counts[0]}")
    print(f"Occupied images: {counts[1]}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Device: {DEVICE}")

    if len(paths) == 0:
        raise ValueError("No training images found.")

    if np.any(counts == 0):
        raise ValueError(
            "Both classes must be present."
        )

    dataset = FinalParkingDataset(
        paths,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Class weights calculated from the full training dataset.
    class_weights = len(labels) / (
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

    history = []

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for images, targets in loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        epoch_loss = (
            running_loss / total_samples
        )

        history.append({
            "epoch": epoch,
            "training_loss": epoch_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Training Loss: {epoch_loss:.6f}"
        )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model.eval()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss": "Class-weighted CrossEntropyLoss",
        "random_seed": RANDOM_SEED,
        "training_samples": len(paths),
        "class_counts": {
            "empty": int(counts[0]),
            "occupied": int(counts[1])
        },
        "unique_dates": int(df["date"].nunique()),
        "training_strategy": (
            "Full-dataset training for fixed 9 epochs; "
            "epoch count selected using median CV best epoch."
        )
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "training_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                key: value
                for key, value in checkpoint.items()
                if key != "model_state_dict"
            },
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("FINAL MODEL SAVED SUCCESSFULLY")
    print("=" * 65)

    print(f"Model: {MODEL_PATH}")
    print(
        f"Training samples: {len(paths)}"
    )
    print(
        f"Final training loss: {history[-1]['training_loss']:.6f}"
    )

    print("\nSaved files:")
    print("- final_model.pth")
    print("- training_history.csv")
    print("- training_config.json")

    print(
        "\nNote: Training loss is not an independent "
        "evaluation metric."
    )


if __name__ == "__main__":
    main()
