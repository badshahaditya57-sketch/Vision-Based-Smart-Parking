
# Vision-Based Smart Parking

### Parking-Space Classification Using Classical Machine Learning and a Lightweight CNN

A computer vision and machine learning project that classifies an individual parking-space image as **Empty** or **Occupied**.

The project compares a lightweight Convolutional Neural Network (CNN) against three classical machine learning approaches using Histogram of Oriented Gradients (HOG) features.

It includes dataset preprocessing, date-grouped cross-validation, model comparison, error analysis, bootstrap confidence intervals, and a Streamlit inference dashboard.

Link: **[https://vision-based-smart-parking-np9xyckqnkdgepkuva8gx6.streamlit.app/]*

---

## Features

- COCO annotation-based parking-space crop extraction.
- Grayscale image preprocessing and standardization.
- HOG feature extraction.
- Lightweight CNN architecture using PyTorch.
- Classical ML baselines:
  - HOG + Linear SVM
  - HOG + Random Forest
  - HOG + Logistic Regression
- Five-fold date-grouped model evaluation.
- Nested date-grouped validation for CNN early stopping.
- Accuracy, precision, recall, macro-F1 and confusion matrices.
- Out-of-fold prediction and error analysis.
- Date-level bootstrap confidence intervals.
- Streamlit dashboard for image classification and model comparison.

---

## Models

Four models are evaluated:

| Model | Feature representation |
|---|---|
| Lightweight CNN | Grayscale image pixels |
| Linear SVM | HOG features |
| Random Forest | HOG features |
| Logistic Regression | HOG features |

The lightweight CNN is the primary deployment model. The three HOG-based models are retained as experimental baselines.

---

## Dataset

The project uses a PKLot-derived parking-space dataset with COCO-format annotations.

### Processed dataset statistics

| Property | Value |
|---|---:|
| Total image crops | 3,991 |
| Empty samples | 1,999 |
| Occupied samples | 1,992 |
| Distinct acquisition dates | 98 |
| Image width | 64 pixels |
| Image height | 128 pixels |
| Number of classes | 2 |

The preprocessing pipeline extracts annotated parking-space regions and creates a metadata CSV containing labels and available source-image information.

### Obtaining and preprocessing the dataset

The raw dataset and processed image crops are not included in this repository. Obtain the dataset separately and provide the appropriate COCO annotation and image directory structure.

Example:

```powershell
python preprocess.py --train-folder "PATH_TO_DATASET\train" --output-dir "data/processed_data" --max-crops-per-class 2000 --seed 42
```

Replace `PATH_TO_DATASET\train` with your actual dataset location.

The generated metadata includes fields such as:

- `crop_path`
- `label`
- `label_name`
- `source_image`
- `frame_id`
- `date`
- `lot`
- `weather`

**Metadata limitation:** The processed dataset currently has unknown lot and weather values. Consequently, this project does not claim cross-lot or cross-weather generalization.

---

## Image preprocessing and HOG features

Images are converted to grayscale and standardized to:

- Width: 64 pixels
- Height: 128 pixels
- Channels: 1

HOG configuration:

| Parameter | Value |
|---|---|
| Orientations | 9 |
| Pixels per cell | 8 x 8 |
| Cells per block | 2 x 2 |
| Block normalization | L2-Hys |
| Feature dimension | 3,780 |

The CNN receives normalized grayscale pixels, with pixel values scaled to the range [0, 1].

---

## Model evaluation methodology

The four models are evaluated on the same five outer date-grouped test folds.

The evaluation procedure:

1. Uses `StratifiedGroupKFold` with five folds.
2. Keeps acquisition dates separated between training and testing.
3. Creates inner date-grouped training and validation partitions.
4. Uses the inner validation partition for CNN early stopping.
5. Trains classical models using the same inner training partitions.
6. Evaluates every model on the same outer test partitions.
7. Aggregates out-of-fold predictions for pooled metrics.

The classical models use fixed hyperparameters, while the CNN uses validation-based early stopping. Therefore, the training and testing partitions are aligned, but the model-selection procedures are not identical.

Random crop-level splitting is not the primary reported evaluation protocol because crops from the same acquisition date may be correlated.

---

## Experimental results

### Pooled out-of-fold performance

The following results are from the five-fold date-grouped evaluation.

| Model | Accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|
| **Lightweight CNN** | **95.99%** | **95.99%** | **160** |
| HOG + Random Forest | 92.06% | 92.06% | 317 |
| HOG + Logistic Regression | 91.26% | 91.25% | 349 |
| HOG + Linear SVM | 90.13% | 90.13% | 394 |

The CNN achieves approximately **3.93 percentage points higher pooled accuracy** than the Random Forest baseline on the same outer test samples.

It also produces 157 fewer errors than the Random Forest model.

### CNN class-wise performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Empty | 99.20% | 92.75% | 95.86% |
| Occupied | 93.17% | 99.25% | 96.11% |

The CNN has particularly high occupied-space recall, while empty-space recall remains an area for improvement.

### Confusion matrix

Rows represent actual classes and columns represent predicted classes.

| Actual / Predicted | Empty | Occupied |
|---|---:|---:|
| Empty | 1,854 | 145 |
| Occupied | 15 | 1,977 |

The primary error pattern is empty spaces being incorrectly classified as occupied.

---

## Confidence interval analysis

Date-level cluster bootstrap analysis was performed using 5,000 repetitions.

| Model | Accuracy | 95% Bootstrap CI |
|---|---:|---:|
| Lightweight CNN | 95.99% | 92.39%–98.71% |
| Random Forest | 92.06% | 88.58%–94.73% |
| Logistic Regression | 91.26% | 87.98%–93.83% |
| Linear SVM | 90.13% | 86.78%–92.69% |

### Paired CNN vs Random Forest

| Metric | Difference | 95% CI |
|---|---:|---:|
| Accuracy | +3.93 percentage points | +2.80 to +5.19 percentage points |
| Macro-F1 | +3.93 percentage points | +2.83 to +5.21 percentage points |

These intervals are conditional on the existing out-of-fold predictions. They account for date-level clustering but do not include variability from retraining the models and do not establish generalization to independent datasets.

---

## Error analysis

The CNN produced 160 errors across the 3,991 out-of-fold predictions.

Important findings:

- 145 empty spaces were classified as occupied.
- 15 occupied spaces were classified as empty.
- Three dates — 2012-10-30, 2012-11-06 and 2012-11-07 — accounted for 106 of the 160 errors.
- CNN errors partially differ from those of the classical models.
- The CNN correctly classifies some samples that all three classical models misclassify.

These findings suggest date-specific distribution differences may affect model performance. Their underlying causes have not yet been established.

---

## Final deployment model

A separate CNN checkpoint was trained on all 3,991 processed samples for nine epochs.

The nine-epoch training duration was selected using the median best epoch from the five cross-validation folds.

Deployment checkpoint:

```text
outputs/
└── cnn_final/
    ├── final_model.pth
    └── training_config.json
```

The final checkpoint is intended for inference.

**Its performance must not be reported as independently tested accuracy.** The reported evaluation metrics come from the separate out-of-fold cross-validation predictions.

---

## Streamlit application

The application provides:

- Image upload for individual parking-space crops.
- Empty or occupied prediction.
- Class-wise softmax scores.
- Processed model-input image preview.
- Model comparison charts.
- Pooled evaluation metrics.
- Confusion matrix visualization.
- Per-class precision, recall and F1-score.
- Bootstrap confidence interval tables.
- Model architecture and research limitations.

### Run the application

```powershell
python -m streamlit run app.py
```

The application loads:

```text
outputs/cnn_final/final_model.pth
```

It also reads experiment results from:

```text
outputs/fair_model_comparison/
outputs/bootstrap_analysis/
```

The final model checkpoint and selected dashboard CSV files are included in the repository. The dataset and most intermediate experiment artifacts are not included.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/badshahaditya57-sketch/Vision-Based-Smart-Parking.git

cd Vision-Based-Smart-Parking
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

For the dashboard and core application:

```powershell
python -m pip install -r requirements.txt
```

For the complete research and experimentation environment:

```powershell
python -m pip install -r requirements-experiments.txt
```

The experimental requirements file includes the core requirements.

The requirements configuration installs CPU-compatible PyTorch. CUDA acceleration is not required for the current implementation.

### 4. Launch the dashboard

```powershell
python -m streamlit run app.py
```

---

## Project structure

```text
Vision-Based-Smart-Parking/
│
├── README.md
├── .gitignore
│
├── app.py
├── common.py
├── preprocess.py
│
├── train_svm.py
├── train_cnn.py
├── train_final_cnn.py
├── evaluate.py
│
├── fair_model_comparison.py
├── preprocessing_experiment.py
├── bootstrap_confidence_intervals.py
├── tune_random_forest.py
├── benchmark_latency.py
│
├── compare_image_conditions.py
├── image_error_stats.py
├── inspect_date_errors.py
├── inspect_errors.py
├── inspect_problem_dates.py
│
├── requirements.txt
├── requirements-experiments.txt
│
├── hog_svm_parking.pkl
│
└── outputs/
    ├── cnn_final/
    │   ├── final_model.pth
    │   └── training_config.json
    │
    ├── fair_model_comparison/
    │   ├── summary_metrics.csv
    │   ├── pooled_metrics.csv
    │   ├── per_class_metrics.csv
    │   ├── confusion_matrices.csv
    │   └── experiment_config.json
    │
    └── bootstrap_analysis/
        ├── bootstrap_summary.csv
        ├── paired_comparisons.csv
        └── experiment_config.json
```

The raw dataset, processed image crops, virtual environments, feature caches, and most intermediate experiment outputs are excluded from Git.

---

## Limitations

1. The system classifies individual parking-space crops rather than detecting parking spaces in complete camera scenes.
2. The processed dataset contains 3,991 samples from 98 acquisition dates.
3. Reliable lot and weather metadata are unavailable.
4. Cross-lot and cross-weather generalization have not been established.
5. Three acquisition dates account for a substantial proportion of CNN errors.
6. The final deployment model has not been independently evaluated on an external test dataset.
7. Softmax scores are not independently calibrated confidence estimates.
8. The results come from one dataset and should not be interpreted as proof of performance in every real-world parking environment.

---

## Future improvements

- Evaluate on an independent parking dataset.
- Obtain reliable parking-lot and weather metadata.
- Perform leave-one-lot-out evaluation when sufficient lot metadata is available.
- Investigate date-specific image distribution shifts.
- Improve empty-space recall.
- Explore model calibration.
- Add full-scene parking-space detection.
- Extend the application to video and live camera inference.
- Evaluate inference performance under controlled hardware conditions.

---

## Reproducibility

The experiments use fixed random seeds and save configuration files, fold metrics, predictions, confusion matrices and bootstrap results.

For reproducible research, record:

- Dataset version and source.
- Preprocessing settings.
- Random seed.
- Evaluation protocol and fold count.
- Python and package versions.
- Hardware configuration.
- Experiment configuration and generated outputs.

The repository includes selected final model and evaluation artifacts. Reproducing the complete experiments requires obtaining the original dataset and regenerating the excluded intermediate artifacts.

Do not commit the virtual environment, large raw datasets, or unnecessary generated caches to GitHub.

---

## Author

**Aditya Raj**

GitHub: [badshahaditya57-sketch](https://github.com/badshahaditya57-sketch)

---

*Vision-Based Smart Parking — Computer Vision and Machine Learning Research Project.*
