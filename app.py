
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from train_cnn import ParkingCNN


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "outputs" / "cnn_final" / "final_model.pth"

OUTPUT_DIR = ROOT / "outputs" / "fair_model_comparison"
BOOTSTRAP_DIR = ROOT / "outputs" / "bootstrap_analysis"

CLASS_NAMES = ["empty", "occupied"]

DEVICE = torch.device("cpu")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vision-Based Smart Parking",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #38bdf8 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #334155;
        padding: 14px;
        border-radius: 10px;
    }

    .dashboard-note {
        background-color: #172033;
        border-left: 4px solid #38bdf8;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = ParkingCNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):

    file_path = Path(path)

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


@st.cache_data
def load_json(path):

    file_path = Path(path)

    if not file_path.exists():
        return None

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes, width, height):

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Invalid or unsupported image. "
            "Please upload a valid JPG or PNG."
        )

    original_height, original_width = image.shape

    if original_height < 8 or original_width < 8:
        raise ValueError(
            "Image dimensions are too small."
        )

    if original_height > 10000 or original_width > 10000:
        raise ValueError(
            "Image dimensions are too large."
        )

    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    normalized = (
        resized.astype(np.float32) / 255.0
    )

    tensor = torch.from_numpy(
        normalized[np.newaxis, np.newaxis, :, :]
    ).float()

    return image, resized, tensor


# ============================================================
# CNN INFERENCE
# ============================================================

def predict_image(model, tensor):

    tensor = tensor.to(DEVICE)

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

    return (
        predicted_index,
        probabilities.cpu().numpy()
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚗 Vision Engine")

try:

    model, checkpoint = load_model()

    model_loaded = True

    st.sidebar.success("CNN model loaded")

    st.sidebar.metric(
        "Inference Engine",
        "Online",
        "Lightweight CNN"
    )

    st.sidebar.caption(
        f"Training samples: "
        f"{checkpoint['training_samples']:,}"
    )

    st.sidebar.caption(
        f"Training dates: "
        f"{checkpoint['unique_dates']}"
    )

except Exception as error:

    model_loaded = False
    model = None
    checkpoint = None

    st.sidebar.error("CNN model unavailable")

    st.sidebar.exception(error)


st.sidebar.markdown("---")

st.sidebar.markdown("### Model Input")

st.sidebar.write("Image width: 64 pixels")
st.sidebar.write("Image height: 128 pixels")
st.sidebar.write("Channels: 1 (grayscale)")
st.sidebar.write("Normalization: pixel values / 255")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Research prototype | CPU inference"
)


# ============================================================
# LOAD EXPERIMENT RESULTS
# ============================================================

summary = load_csv(
    OUTPUT_DIR / "summary_metrics.csv"
)

pooled = load_csv(
    OUTPUT_DIR / "pooled_metrics.csv"
)

per_class = load_csv(
    OUTPUT_DIR / "per_class_metrics.csv"
)

confusion = load_csv(
    OUTPUT_DIR / "confusion_matrices.csv"
)

bootstrap = load_csv(
    BOOTSTRAP_DIR / "bootstrap_summary.csv"
)

paired = load_csv(
    BOOTSTRAP_DIR / "paired_comparisons.csv"
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title("Vision-Based Smart Parking")

st.markdown(
    """
    A computer vision research prototype for classifying
    individual parking-space images as **empty** or **occupied**.

    The primary inference model is a lightweight CNN.
    Classical HOG-based models are retained for experimental comparison.
    """
)

if not model_loaded:

    st.error(
        "The CNN model could not be loaded. "
        "Check outputs/cnn_final/final_model.pth."
    )

    st.stop()


# ============================================================
# TOP-LEVEL METRICS
# ============================================================

if pooled is not None:

    cnn_row = pooled[
        pooled["model"] == "Lightweight CNN"
    ]

    if not cnn_row.empty:

        row = cnn_row.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "CV Accuracy",
            f"{row['accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Macro F1",
            f"{row['macro_f1'] * 100:.2f}%"
        )

        col3.metric(
            "Correct Predictions",
            f"{int(row['n_samples'] - row['errors']):,}"
        )

        col4.metric(
            "CV Errors",
            f"{int(row['errors']):,}"
        )

    st.caption(
        "These are pooled out-of-fold date-grouped cross-validation "
        "metrics, not measurements from the final deployment checkpoint."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Live Inference",
        "📊 Model Comparison",
        "🧪 Evaluation & Confidence",
        "ℹ️ Model Information"
    ]
)


# ============================================================
# TAB 1: LIVE INFERENCE
# ============================================================

with tab1:

    st.header("Parking Space Classification")

    st.write(
        "Upload an image containing one isolated parking-space crop."
    )

    uploaded_file = st.file_uploader(
        "Upload parking-space image",
        type=["jpg", "jpeg", "png"],
        help="The image will be converted to grayscale and resized to 64 × 128."
    )

    if uploaded_file is not None:

        try:

            image_bytes = uploaded_file.getvalue()

            if len(image_bytes) > 10 * 1024 * 1024:
                raise ValueError(
                    "Maximum upload size is 10 MB."
                )

            original, processed, tensor = preprocess_image(
                image_bytes,
                checkpoint["image_width"],
                checkpoint["image_height"]
            )

            prediction, probabilities = predict_image(
                model,
                tensor
            )

            predicted_class = CLASS_NAMES[prediction]

            st.markdown("---")

            image_col, result_col = st.columns(
                [1, 1],
                gap="large"
            )

            with image_col:

                st.subheader("Input Image")

                st.image(
                    original,
                    caption="Uploaded grayscale image",
                    use_container_width=True
                )

                st.subheader("Processed Image")

                st.image(
                    processed,
                    caption="64 × 128 model input",
                    use_container_width=True
                )

            with result_col:

                st.subheader("Prediction")

                if predicted_class == "occupied":

                    st.error("🚨 PARKING SPACE OCCUPIED")

                else:

                    st.success("✅ PARKING SPACE EMPTY")

                st.metric(
                    "Predicted Class",
                    predicted_class.upper()
                )

                st.metric(
                    "Model Softmax Score",
                    f"{probabilities[prediction] * 100:.2f}%"
                )

                st.caption(
                    "Softmax scores are model outputs, not "
                    "independently calibrated probabilities."
                )

            st.markdown("---")

            st.subheader("Class Score Distribution")

            probability_df = pd.DataFrame({
                "Class": ["Empty", "Occupied"],
                "Softmax Score": probabilities * 100
            })

            fig = px.bar(
                probability_df,
                x="Softmax Score",
                y="Class",
                orientation="h",
                color="Class",
                color_discrete_map={
                    "Empty": "#10b981",
                    "Occupied": "#ef4444"
                },
                text_auto=".2f"
            )

            fig.update_layout(
                xaxis_title="Softmax score (%)",
                yaxis_title="",
                xaxis_range=[0, 100],
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Unable to process the uploaded image: {error}"
            )


# ============================================================
# TAB 2: MODEL COMPARISON
# ============================================================

with tab2:

    st.header("Experimental Model Comparison")

    if summary is None or pooled is None:

        st.warning(
            "Comparison files are missing. "
            "Run fair_model_comparison.py first."
        )

    else:

        chart_df = summary.copy()

        chart_df["accuracy_percent"] = (
            chart_df["accuracy_mean"] * 100
        )

        chart_df["macro_f1_percent"] = (
            chart_df["macro_f1_mean"] * 100
        )

        st.subheader("Mean Date-Grouped CV Accuracy")

        fig = px.bar(
            chart_df.sort_values(
                "accuracy_percent",
                ascending=False
            ),
            x="model",
            y="accuracy_percent",
            error_y=chart_df.sort_values(
                "accuracy_percent",
                ascending=False
            )["accuracy_std"] * 100,
            text_auto=".2f",
            labels={
                "model": "Model",
                "accuracy_percent": "Accuracy (%)"
            }
        )

        fig.update_layout(
            yaxis_range=[0, 105],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Mean Macro F1")

        fig_f1 = px.bar(
            chart_df.sort_values(
                "macro_f1_percent",
                ascending=False
            ),
            x="model",
            y="macro_f1_percent",
            text_auto=".2f",
            labels={
                "model": "Model",
                "macro_f1_percent": "Macro F1 (%)"
            }
        )

        fig_f1.update_layout(
            yaxis_range=[0, 105],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )

        st.plotly_chart(
            fig_f1,
            use_container_width=True
        )

        st.subheader("Pooled Evaluation Metrics")

        display_df = pooled.copy()

        percentage_columns = [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1"
        ]

        for column in percentage_columns:
            display_df[column] = (
                display_df[column] * 100
            ).round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "All four models use the same outer test folds and "
            "the same inner training partitions. The CNN uses "
            "validation-based early stopping; classical models "
            "use fixed hyperparameters."
        )


# ============================================================
# TAB 3: EVALUATION AND CONFIDENCE
# ============================================================

with tab3:

    st.header("Evaluation Analysis")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    if confusion is not None:

        available_models = confusion["model"].unique().tolist()

        selected_model = st.selectbox(
            "Select model",
            available_models,
            index=available_models.index("Lightweight CNN")
            if "Lightweight CNN" in available_models else 0
        )

        selected = confusion[
            confusion["model"] == selected_model
        ]

        matrix = np.zeros((2, 2), dtype=int)

        for _, row in selected.iterrows():

            actual = CLASS_NAMES.index(
                row["true_class"]
            )

            predicted = CLASS_NAMES.index(
                row["predicted_class"]
            )

            matrix[actual, predicted] = int(
                row["count"]
            )

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted Empty", "Predicted Occupied"],
                y=["Actual Empty", "Actual Occupied"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=True
            )
        )

        fig_cm.update_layout(
            title=f"{selected_model} — Confusion Matrix",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

    else:

        st.warning("Confusion matrix file not found.")

    # --------------------------------------------------------
    # PER CLASS METRICS
    # --------------------------------------------------------

    st.subheader("Per-Class Performance")

    if per_class is not None:

        selected_class_model = st.selectbox(
            "Select model for class metrics",
            per_class["model"].unique().tolist(),
            key="class_model"
        )

        class_df = per_class[
            per_class["model"] == selected_class_model
        ].copy()

        for column in ["precision", "recall", "f1"]:
            class_df[column] *= 100

        st.dataframe(
            class_df.round(2),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # BOOTSTRAP CONFIDENCE INTERVALS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("Date-Level Bootstrap Confidence Intervals")

    if bootstrap is not None:

        st.write(
            "95% percentile confidence intervals based on "
            "5,000 date-cluster bootstrap repetitions."
        )

        bootstrap_display = bootstrap.copy()

        for column in [
            "observed_value",
            "ci_lower",
            "ci_upper"
        ]:

            bootstrap_display[column] *= 100

        st.dataframe(
            bootstrap_display.round(2),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("Bootstrap summary file not found.")

    # --------------------------------------------------------
    # PAIRED COMPARISON
    # --------------------------------------------------------

    st.subheader("Paired CNN vs Random Forest Comparison")

    if paired is not None:

        paired_display = paired.copy()

        for column in [
            "observed_difference",
            "ci_lower",
            "ci_upper"
        ]:

            paired_display[column] *= 100

        st.dataframe(
            paired_display.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Both paired confidence intervals are above zero. "
            "This supports a CNN advantage under the chosen "
            "date-cluster bootstrap procedure. The intervals "
            "are conditional on fixed out-of-fold predictions."
        )


# ============================================================
# TAB 4: MODEL INFORMATION
# ============================================================

with tab4:

    st.header("Model Architecture and Methodology")

    st.subheader("Deployment Model")

    st.write("**Architecture:** Lightweight ParkingCNN")

    st.write(
        """
        - Input: one-channel grayscale image.
        - Input dimensions: 128 × 64 (height × width).
        - Four convolutional blocks.
        - Channel progression: 1 → 16 → 32 → 64 → 128.
        - Batch normalization and ReLU activations.
        - Adaptive average pooling.
        - Fully connected classifier: 128 → 64 → 2.
        """
    )

    st.subheader("Training Configuration")

    config_columns = [
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "loss",
        "training_samples",
        "unique_dates"
    ]

    config_df = pd.DataFrame({
        "Parameter": config_columns,
        "Value": [
            checkpoint.get(column, "N/A")
            for column in config_columns
        ]
    })

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Evaluation Protocol")

    st.write(
        """
        - Five-fold StratifiedGroupKFold.
        - Grouping variable: acquisition date.
        - Inner date-grouped validation for CNN early stopping.
        - Classical models trained on the same inner training partitions.
        - Out-of-fold predictions used for pooled metrics.
        - Date-cluster bootstrap used for confidence intervals.
        """
    )

    st.subheader("Research Limitations")

    st.warning(
        """
        1. The dataset contains 3,991 parking-space crops from 98 dates.
        2. Lot and weather metadata are unavailable or marked unknown.
        3. Cross-lot and cross-weather generalization has not been established.
        4. Three dates account for a substantial share of CNN errors.
        5. The final deployment checkpoint is trained on the full dataset;
           its independent test performance has not been measured.
        6. Softmax scores are not verified as calibrated confidence values.
        7. The application classifies isolated crops, not complete parking-lot
           camera feeds or multiple spaces in a single image.
        """
    )

    st.subheader("Model Artifact")

    st.code(str(MODEL_PATH))

    st.caption(
        "The final deployment checkpoint is separate from the "
        "five date-grouped cross-validation checkpoints."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Vision-Based Smart Parking | Research and Portfolio Project | "
    "Lightweight CNN + HOG Baselines"
)
