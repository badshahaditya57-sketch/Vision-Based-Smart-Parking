# Vision-Based Smart Parking via Classical HOG and SVM Pipelines

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)

## Abstract
This repository contains an edge-optimized, real-time computer vision pipeline for parking space occupancy classification. While contemporary solutions heavily rely on computationally expensive Deep Learning architectures (such as Convolutional Neural Networks or YOLO-based object detectors), this project demonstrates a highly efficient, interpretative classical machine learning approach. By utilizing Histogram of Oriented Gradients (HOG) for robust spatial feature extraction and a Linear Support Vector Machine (SVM) for high-dimensional hyperplane classification, this system achieves a **93.25% accuracy** with sub-15ms inference latency, completely bypassing the need for GPU acceleration. 

This architecture is specifically designed for deployment on resource-constrained edge IoT devices (e.g., Raspberry Pi) in smart city infrastructures.

---

Web URL: **https://vision-based-smart-parking-np9xyckqnkdgepkuva8gx6.streamlit.app/**

## 1. Motivation and Research Rationale
The deployment of smart parking infrastructure is often bottlenecked by hardware costs and inference latency. Transmitting high-definition video feeds to centralized cloud servers for neural network inference introduces network dependency, high bandwidth costs, and privacy concerns. Conversely, deploying CNNs directly on edge cameras requires expensive localized AI accelerators.

This project solves this bottleneck by retreating to foundational, mathematically rigorous computer vision techniques. Classical pipelines offer deterministic execution, strict memory boundaries, and complete transparency. Unlike deep learning "black boxes," a HOG-based feature extractor allows engineers to visually map exactly which pixel gradients influence the classifier's decision boundaries. 

---

## 2. Dataset and Preprocessing Methodology
The model is trained on a subset of the **PKLot Dataset**, containing hundreds of thousands of parking space samples across varied weather conditions (sunny, overcast, rainy) and camera angles.

### ROI Extraction and Standardization
The raw dataset utilizes the COCO JSON annotation format to map bounding boxes over full parking lot surveillance frames. The `preprocess.py` engine executes the following pipeline:
1. **JSON Parsing:** Iterates through `_annotations.coco.json` to extract precise `[x_top_left, y_top_left, width, height]` coordinates for individual parking bays.
2. **ROI Cropping:** Extracts the localized Region of Interest (ROI) from the macro frame using OpenCV array slicing.
3. **Dimensional Standardization:** Squashes the cropped matrix into a standardized `64x128` pixel resolution, ensuring uniform vector lengths for the SVM.
4. **Grayscale Conversion:** Strips color channels to reduce array depth from 3D (BGR) to 2D. Shadows and vehicle shapes are defined by intensity gradients, not color, making chromatic data computationally redundant.

---

## 3. Mathematical Architecture

### 3.1 Feature Extraction: Histogram of Oriented Gradients (HOG)
Instead of feeding raw pixel intensities into the classifier—which are highly susceptible to lighting variance—the system extracts structural geometries (edges, corners, and contours) using the HOG algorithm.

* **Gradient Computation:** The pipeline applies 1D centered point discrete derivative masks in both horizontal and vertical directions to calculate the gradient magnitude and orientation at every pixel.
* **Orientation Binning:** The 64x128 image is divided into `8x8` pixel cells. Each cell compiles a histogram of gradient directions spread across `9` orientation bins (0° to 180°).
* **Block Normalization (`L2-Hys`):** Illumination drops (e.g., cloud cover) scale down gradient magnitudes globally. To create illumination invariance, cells are grouped into overlapping `2x2` blocks. The histograms within each block are concatenated and normalized using the L2-Norm, followed by clipping (Hysteresis) to limit the influence of extreme shadows.
* **Output:** A flattened, high-dimensional feature vector mathematically representing the physical geometry of a car's bumpers, wheels, and windshields, ignoring the asphalt's ambient lighting.

### 3.2 Classification: Support Vector Machine (SVM)
The extracted HOG vectors map into a high-dimensional feature space. A Support Vector Machine is deployed to calculate the optimal hyperplane that separates the "Available" class from the "Occupied" class.

* **Linear Kernel:** Because HOG features naturally map linearly separable structures in high-dimensional spaces, a Linear Kernel is utilized. This prevents the severe computational overhead of calculating RBF (Radial Basis Function) kernels.
* **Margin Maximization:** The algorithm calculates the support vectors—the specific images lying closest to the boundary—to maximize the margin between classes, ensuring robust generalization to unseen parking lots.
* **Probabilistic Calibration:** Using Platt Scaling under the hood (`predict_proba`), the SVM translates absolute vector distances from the hyperplane into calibrated percentage confidences, vital for telemetry monitoring.

---

## 4. System Performance and Telemetry
The model was trained and evaluated on a randomly sampled, balanced subset of 4,000 isolated parking spots to prevent memory overflow during local execution.

### Evaluation Metrics
* **Final Test Accuracy:** `93.25%`
* **Precision (Empty/Occupied):** `0.94` / `0.92`
* **Recall (Empty/Occupied):** `0.93` / `0.94`
* **F1-Score:** `0.93`
* **Inference Latency:** `< 12ms` per ROI on standard x86 CPU architecture.

The high recall on the "Occupied" class ensures the system rarely triggers false positives for availability, a critical user-experience metric for parking guidance systems.

---

## 5. Repository Structure
```text
vision-smart-parking/
├── app.py                # Streamlit dashboard and UI routing
├── preprocess.py         # COCO JSON parser and OpenCV cropping engine
├── train_svm.py          # Machine learning training and evaluation script
├── requirements.txt      # Dependency configurations
├── .gitignore            # Security and binary exclusions
└── README.md             # Architectural documentation
