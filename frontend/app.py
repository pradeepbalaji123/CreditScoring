import os
import sys
from pathlib import Path
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shap
from src.explainability.grad_cam import make_gradcam_heatmap
from src.explainability.saliency import compute_saliency
from src.explainability.shap_explain import compute_shap
from src.preprocessing.feature_selection_image_generation import convert_to_images
MODEL_PATHS = {
    "HMEQ": PROJECT_ROOT / "src/models/HMEQ_model.keras",
    "AUSTRALIAN": PROJECT_ROOT / "src/models/AUSTRALIAN_model.keras"
}
DATA_PATH = PROJECT_ROOT / "data/processed"
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        models[name] = tf.keras.models.load_model(path)
    return models
models = load_models()
@st.cache_resource
def load_preprocessing(dataset):
    bin_info = pickle.load(
        open(DATA_PATH / f"{dataset}_bin_info.pkl", "rb")
    )
    selected_features = pickle.load(
        open(DATA_PATH / f"{dataset}_selected_features.pkl", "rb")
    )
    return bin_info, selected_features
st.title("Explainable Credit Risk Prediction")
st.write("CNN-based Credit Scoring with Explainable AI")
dataset = st.selectbox(
    "Select Dataset",
    ["HMEQ", "AUSTRALIAN"]
)
model = models[dataset]
bin_info, selected_features = load_preprocessing(dataset)
st.subheader("Borrower Information")
inputs = {}
if dataset == "HMEQ":
    inputs["LOAN"] = st.number_input("Loan Amount", 1000, 100000, 15000)
    inputs["MORTDUE"] = st.number_input("Mortgage Due", 0, 500000, 40000)
    inputs["VALUE"] = st.number_input("Property Value", 10000, 500000, 80000)
    inputs["DEROG"] = st.number_input("Derogatory Reports", 0, 10, 0)
    inputs["DELINQ"] = st.number_input("Delinquencies", 0, 15, 0)
    inputs["CLAGE"] = st.number_input("Credit Line Age", 0, 1000, 120)
    inputs["NINQ"] = st.number_input("Recent Credit Enquiries", 0, 10, 1)
    inputs["DEBTINC"] = st.number_input("Debt Income Ratio", 0.0, 200.0, 30.0)
    inputs["JOB"] = st.selectbox(
        "Job",
        ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]
    )
elif dataset == "AUSTRALIAN":
    inputs["A2"] = st.number_input("A2", 13.0, 80.0, 25.0)
    inputs["A3"] = st.number_input("A3", 0.0, 30.0, 2.0)
    inputs["A4"] = st.selectbox("A4", [1,2,3])
    inputs["A5"] = st.selectbox("A5", list(range(1,15)))
    inputs["A6"] = st.selectbox("A6", list(range(1,10)))
    inputs["A7"] = st.number_input("A7", 0.0, 30.0, 1.0)
    inputs["A8"] = st.selectbox("A8", [0,1])
    inputs["A9"] = st.selectbox("A9", [0,1])
    inputs["A10"] = st.number_input("A10", 0, 70, 5)
    inputs["A12"] = st.selectbox("A12", [1,2,3])
    inputs["A13"] = st.number_input("A13", 0, 2000, 200)
    inputs["A14"] = st.number_input("A14", 1, 100000, 1000)
if st.button("Predict Credit Risk"):
    df = pd.DataFrame([inputs])
    df["TARGET"] = 0
    images, _, B, D, features = convert_to_images(
        df,
        target="TARGET",
        bin_info=bin_info,
        selected_features=selected_features
    )
    sample = images[0]
    sample_input = np.expand_dims(sample, axis=(0, -1)).astype(np.float32)
    prediction = model.predict(sample_input)
    pred_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    st.subheader("Prediction Result")
    if pred_class == 0:
        st.success(f"Low Credit Risk (Confidence: {confidence:.4f})")
    else:
        st.error(f"High Credit Risk (Confidence: {confidence:.4f})")
    st.subheader("Feature Image")
    fig1, ax1 = plt.subplots()
    ax1.imshow(sample, cmap="gray")
    ax1.set_title("Generated Feature Image")
    st.pyplot(fig1)
    st.subheader("Saliency Map")
    saliency = compute_saliency(model, sample_input)
    fig2, ax2 = plt.subplots()
    ax2.imshow(saliency[0], cmap="hot")
    ax2.set_title("Saliency Map")
    st.pyplot(fig2)
    st.subheader("Grad-CAM")
    conv_layers = [
        layer.name for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    last_conv = conv_layers[-1]
    heatmap = make_gradcam_heatmap(sample_input, model, last_conv)
    fig3, ax3 = plt.subplots()
    ax3.imshow(heatmap, cmap="jet")
    ax3.set_title("Grad-CAM")
    st.pyplot(fig3)
    st.subheader("SHAP Explanation")
    background = sample_input
    shap_values = compute_shap(model, background, sample_input)
    shap.image_plot(shap_values, sample_input, show=False)
    fig4 = plt.gcf()
    st.pyplot(fig4)