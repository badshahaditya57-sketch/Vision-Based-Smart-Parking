import json
import os

import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import plotly.express as px
import pandas as pd

# Shared preprocessing: identical to what training / evaluation use.
from common import HOG_PARAMS, hog_features, standardize, to_gray

# 1. Page Configuration
st.set_page_config(page_title="Vision-Based Smart Parking", layout="wide", page_icon="🚘", initial_sidebar_state="expanded")

# 2. Tailwind-Inspired Custom CSS Injection
st.markdown("""
    <style>
    /* Main background and text */
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    
    /* Card containers */
    .dashboard-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Accent headers */
    h1, h2, h3 { color: #38bdf8 !important; font-family: 'Inter', sans-serif; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_model():
    return joblib.load('hog_svm_parking.pkl')

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

def load_json(path):
    """Return parsed JSON or None. Metrics shown in the UI come from files
    written by train_svm.py / benchmark_latency.py -- never hard-coded."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

model_card = load_json(os.path.join("results", "model_card.json"))
latency = load_json(os.path.join("results", "latency.json"))

# 4. Sidebar: System Status & Metrics
with st.sidebar:
    st.markdown("## 👁️ Vision Engine")
    st.metric(label="Inference Engine", value="Online" if model_loaded else "Model missing", delta="HOG + SVM")
    if model_card:
        st.metric(label="Grouped Hold-out Accuracy", value=f"{model_card['holdout_accuracy'] * 100:.2f}%",
                  delta=f"n_test={model_card['n_test']}", delta_color="off")
        st.caption(f"Split by: {model_card['grouping']}")
    else:
        st.metric(label="Grouped Hold-out Accuracy", value="not evaluated", delta="run train_svm.py", delta_color="off")
    if latency:
        st.metric(label="Latency (median, end-to-end)", value=f"{latency['svm_end_to_end_median_ms']:.2f} ms",
                  delta=f"p95 {latency['svm_end_to_end_p95_ms']:.2f} ms", delta_color="off")
        st.caption(f"Measured single-thread on: {latency['hardware']['cpu']}")
    else:
        st.metric(label="Latency", value="not measured", delta="run benchmark_latency.py", delta_color="off")

    st.markdown("---")
    st.markdown("### HOG Parameters")
    st.code("\n".join(f"{k} = {v!r}" for k, v in HOG_PARAMS.items()), language="python")

# 5. Main Content Header
st.title("Vision-Based Smart Parking")
st.markdown("Parking space classification utilizing classical computer vision pipelines (HOG + SVM), designed for low-compute hardware.")

# 6. Tabbed Interface
tab1, tab2, tab3 = st.tabs(["🔴 Live Inference", "⚙️ Feature Extraction Engine", "📊 Model Performance"])

if model_loaded:
    with tab1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            st.markdown("#### Input Feed")
            uploaded_file = st.file_uploader("Upload an isolated space crop (any size; resized to 64x128)", type=["jpg", "png", "jpeg"])
            
        with col2:
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                
                # Preprocess & Predict (same code path as training)
                img_gray = to_gray(img)
                img_resized = standardize(img)
                features = hog_features(img_resized)
                
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
                
                st.markdown("#### Classification Results")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.image(img, channels="BGR", caption="Processed ROI", use_container_width=True)
                    
                with res_col2:
                    if prediction == 1:
                        st.error("🚨 SPACE OCCUPIED")
                        st.metric("Model Probability", f"{probabilities[1]*100:.2f}%")
                    else:
                        st.success("✅ SPACE AVAILABLE")
                        st.metric("Model Probability", f"{probabilities[0]*100:.2f}%")
                    st.caption("Platt-scaled SVM probability; calibration on unseen lots has not been verified.")
                
                # Plotly Gauge Chart for Confidence
                conf_df = pd.DataFrame({'Class': ['Available', 'Occupied'], 'Probability': probabilities})
                fig = px.bar(conf_df, x='Probability', y='Class', orientation='h', 
                             color='Class', color_discrete_map={'Available': '#10b981', 'Occupied': '#ef4444'},
                             title="Confidence Distribution")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e2e8f0", height=200, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### Gradient Vector Visualization")
        st.write("Unlike deep learning black boxes, classical SVMs rely on highly interpretable geometric feature maps. Below is the structural extraction generated by the Histogram of Oriented Gradients before passing to the SVM hyperplane.")
        
        if uploaded_file is not None:
            feat, hog_img = hog(img_resized, visualize=True, **HOG_PARAMS)
            from skimage import exposure
            hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
            
            h_col1, h_col2, h_col3 = st.columns(3)
            h_col1.image(img_resized, caption="1. Grayscale 64x128 ROI", use_container_width=True)
            h_col2.image(hog_image_rescaled, caption="2. Extracted HOG Geometries", use_container_width=True, clamp=True)
            h_col3.write(f"**Feature Vector Array:**\n\nFlattened 1D array of `{len(feat)}` elements passed to linear SVM.")
            h_col3.dataframe(pd.DataFrame(feat, columns=["Gradient Magnitude"]).head(10))
        else:
            st.info("Upload an image in the Live Inference tab to view its gradient extraction.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### Experimental Results")
        summary_path = os.path.join("results", "summary_metrics.csv")
        if os.path.exists(summary_path):
            s = pd.read_csv(summary_path)
            view = s[["split", "model", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]].round(4)
            st.dataframe(view, use_container_width=True)
            st.caption("'random' = crop-level split (leaky); 'grouped' = whole frames/days held out. Mean/std over folds.")
            for fname in ["fig_model_comparison.png", "fig_weather.png", "fig_confusion_matrices.png", "fig_latency.png"]:
                p = os.path.join("results", fname)
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
        else:
            st.info("No experiment results found. Run `python evaluate.py` and commit the `results/` folder.")

        st.markdown("### Deployment Architecture")
        arch_data = {
            "Component": ["Image Capture", "Preprocessing", "Feature Extraction", "Classification", "Output UI"],
            "Technology": ["Static Feed", "OpenCV (Grayscale, Resize)", "scikit-image (HOG)", "scikit-learn (Linear SVM)", "Streamlit / Plotly"]
        }
        st.table(pd.DataFrame(arch_data))
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("Model file `hog_svm_parking.pkl` not found. Please run the training script first.")
