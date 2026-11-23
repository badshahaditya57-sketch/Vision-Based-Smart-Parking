# Vision-Based Smart Parking: HOG + Classical ML

A CPU-oriented parking-space **crop classifier** using Histogram of Oriented Gradients (HOG) and classical machine-learning models. The current application predicts whether one already-cropped parking space is empty or occupied. It does **not** detect parking-space boundaries or process a complete CCTV scene automatically.

## Models
- HOG + Linear SVM (deployment model)
- HOG + Random Forest
- HOG + Logistic Regression

## Dataset and preprocessing
The preprocessing script reads COCO-format annotations from a PKLot-derived dataset, crops annotated parking spaces, converts each crop to grayscale, resizes it to 64×128 pixels and writes a `metadata.csv` file.

The category mapper accepts explicit labels (`empty`, `vacant`, `available`, `free`, `occupied`, `full`, `taken`). Inspect your dataset's COCO category names before running; unknown labels are skipped rather than guessed.

```bash
python preprocess.py --train-folder raw_dataset/train --output-dir processed_data
```

Optional date metadata CSV:

```csv
date,lot,weather
2012-06-22,PUCPR,sunny
```

```bash
python preprocess.py --train-folder raw_dataset/train --output-dir processed_data --meta-csv date_meta.csv
```

Filename parsing is best-effort. If the original dataset stores lot/date/weather in parent directories rather than filenames, supply metadata explicitly or extend the parser. Do not assume missing metadata has been recovered.

## HOG feature definition
Images are standardized to 64×128 grayscale pixels. HOG uses 9 orientation bins, 8×8-pixel cells, 2×2-cell blocks and L2-Hys normalization. This produces 3,780 features for the specified geometry.

## Installation

Application:

```bash
pip install -r requirements.txt
```

Experiments:

```bash
pip install -r requirements-experiments.txt
```

## Train the deployment SVM

```bash
python train_svm.py --data-dir processed_data --max-per-class 2000 --group-by frame
```

This script writes `hog_svm_parking.pkl` and `results/model_card.json`. Its single grouped holdout score is not the same as cross-validation performance. Verify that both classes occur in the training and test partitions before interpreting the metrics.

## Cross-validated model comparison

```bash
python evaluate.py --data-dir processed_data --max-per-class 2000 --group-by frame --n-splits 5
```

The evaluation script writes:
- `results/fold_metrics.csv`
- `results/summary_metrics.csv`
- `results/confusion_matrices.csv`
- `results/weather_metrics.csv` when usable weather labels exist
- `results/fig_model_comparison.png`
- `results/run_info.json`

The random crop-level split is a diagnostic baseline and may be optimistic because crops from the same source frame can be highly correlated. Treat grouped evaluation as the primary estimate. Frame grouping only prevents overlap of identified frames; it does not establish generalization to unseen parking lots. Use `--group-by lot` only when lot metadata is complete and reliable.

Weather metrics are descriptive subgroup results, not proof of weather robustness. Check sample and group counts before drawing conclusions.

## Latency benchmark

```bash
python benchmark_latency.py --data-dir processed_data --model-path hog_svm_parking.pkl --iterations 500
```

Writes `results/latency.json`. The benchmark uses one preloaded crop and reports preprocessing, HOG, prediction and total latency. It excludes image capture, disk I/O, network transfer and Streamlit rendering. Report the hardware and measurement scope with any latency claim.

## Run the Streamlit app

```bash
streamlit run app.py
```

The app expects `hog_svm_parking.pkl` in the repository root. It reads optional metrics from `results/model_card.json` and `results/latency.json`.

## Suggested repository layout

```text
.
├── app.py
├── common.py
├── preprocess.py
├── train_svm.py
├── evaluate.py
├── benchmark_latency.py
├── requirements.txt
├── requirements-experiments.txt
├── hog_svm_parking.pkl
├── processed_data/             # local dataset output; do not commit large data
├── results/                    # commit reproducible result artifacts
└── README.md
```

## Limitations and reproducibility
- This is a single-ROI binary classifier, not a full parking-space detector.
- Grouping quality depends on correct source-frame/date/lot metadata.
- Filename-based metadata inference is not guaranteed to recover parent-directory information.
- No hyperparameter tuning or nested cross-validation is performed by default.
- Probabilities from `SVC(probability=True)` are not automatically evidence of calibration; assess calibration separately on held-out data.
- Results are not included as claims until the scripts have been run on the stated dataset and hardware.
- Do not report unseen-lot generalization unless lot-held-out evaluation has actually been run with enough independent lots.

## Reproducibility checklist
Record dataset version/source, sampling cap, random seed, group mode, fold count, package versions, hardware, and the generated CSV/JSON outputs. Keep raw data and large generated crops out of Git unless licensing and repository-size constraints permit them.
