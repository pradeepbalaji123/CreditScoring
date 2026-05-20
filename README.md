# Explainable Deep Learning Based Credit Scoring

An explainable deep learning framework for credit risk prediction using Convolutional Neural Networks (CNNs) on transformed tabular financial data.
The system integrates Explainable AI (XAI) techniques such as SHAP, Grad-CAM, and Saliency Maps to provide transparent and interpretable credit scoring decisions.

---

# Overview

Traditional credit scoring systems primarily rely on statistical and classical machine learning techniques that often struggle to capture complex relationships in financial data. Deep learning models provide improved predictive performance but suffer from poor interpretability, making them difficult to deploy in regulated financial environments.

This project addresses these challenges by:

* Transforming tabular credit data into image representations
* Applying 2D CNN architectures for credit risk prediction
* Integrating explainability techniques to interpret model predictions
* Providing a Streamlit-based interactive interface for visualization and prediction

The proposed framework balances predictive performance with transparency and interpretability, making it suitable for real-world credit scoring applications.

---

# Features

* Deep Learning based credit risk prediction
* Tabular-to-image transformation pipeline
* Weight of Evidence (WoE) encoding
* Information Value (IV) based feature selection
* CNN-based classification
* SHAP explanations
* Grad-CAM visualizations
* Saliency Map analysis
* Streamlit web application
* Interactive explainability visualizations

---

# System Architecture

```text
Raw Credit Dataset
        │
        ▼
Data Preprocessing
(Missing value handling, WoE, IV)
        │
        ▼
Feature Image Generation
(Tabular → Image Transformation)
        │
        ▼
CNN-Based Credit Scoring
        │
        ▼
Explainability Layer
(SHAP, Grad-CAM, Saliency Maps)
        │
        ▼
Prediction + Visualization
```

---

# Project Structure

```text
CreditScoring/
│
├── data/
│   ├── processed/
│   └── raw/
│
├── frontend/
│   └── app.py
│
├── src/
│   ├── evaluation/
│   │   └── test_model.py
│   │
│   ├── explainability/
│   │   ├── grad_cam.py
│   │   ├── run_explainability.py
│   │   ├── saliency.py
│   │   └── shap_explain.py
│   │
│   ├── models/
│   │   ├── AUSTRALIAN_model.keras
│   │   └── HMEQ_model.keras
│   │
│   └── preprocessing/
│       ├── binning.py
│       ├── feature_selection_image_generation.py
│       └── preprocessor.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Datasets Used

The project uses publicly available credit scoring datasets:

## 1. Australian Credit Approval Dataset

* Source: UCI Machine Learning Repository
* Binary classification dataset for credit approval prediction

## 2. HMEQ Dataset

* Home Equity Loan dataset
* Used for default risk prediction

Dataset Links:

* Australian Dataset:
  https://archive.ics.uci.edu/ml/datasets/Statlog+(Australian+Credit+Approval)

* HMEQ Dataset:
  https://www.kaggle.com/ajay1735/hmeq-data

---

# Methodology

## 1. Data Preprocessing

The preprocessing pipeline includes:

* Missing value handling
* Feature binning
* Weight of Evidence (WoE) transformation
* Information Value (IV) computation
* Feature selection

## 2. Tabular-to-Image Transformation

Financial tabular features are converted into image representations using:

* One-hot encoding
* Feature bin mapping
* Binary image generation

Each pixel represents a feature bin derived from the original dataset.

## 3. CNN-Based Credit Scoring

A 2D Convolutional Neural Network is trained on the generated feature images.

### CNN Architecture

| Layer        | Configuration           |
| ------------ | ----------------------- |
| Conv Layer 1 | 128 filters, 3×3 kernel |
| Conv Layer 2 | 256 filters, 3×3 kernel |
| FC Layer 1   | 64 neurons              |
| Output Layer | Softmax classifier      |

## 4. Explainability Layer

The project integrates multiple XAI techniques:

### SHAP

* Feature contribution analysis
* Local interpretability

### Grad-CAM

* Heatmap-based CNN explanation
* Important region visualization

### Saliency Maps

* Gradient-based importance visualization
* Pixel-level influence detection

---

# Performance Results

| Dataset    | Accuracy | AUC    |
| ---------- | -------- | ------ |
| HMEQ       | 88.31%   | 0.9088 |
| Australian | 83.09%   | 0.9060 |

The model demonstrates strong discriminative capability for identifying default and non-default borrowers.

---

# Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd CreditScoring
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Application

Start the Streamlit application:

```bash
streamlit run frontend/app.py
```

---

# Explainability Outputs

The system generates multiple visual explanations:

* SHAP feature contribution plots
* Grad-CAM heatmaps
* Saliency maps
* ROC curves
* Confusion matrices

These explanations improve transparency and trust in automated credit decisions.

---

# Tech Stack

## Languages & Frameworks

* Python
* Streamlit
* TensorFlow / Keras

## Libraries

* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* SHAP

## Machine Learning Techniques

* CNN
* WoE Encoding
* Information Value Analysis
* Explainable AI (XAI)

---

# Applications

* Automated credit approval systems
* Financial risk assessment
* Banking decision support systems
* Explainable AI research
* Credit default prediction

---

# Future Enhancements

* Integration of larger financial datasets
* Deployment-ready API architecture
* Advanced CNN architectures (ResNet, EfficientNet)
* Real-time prediction pipeline
* Additional XAI methods such as LIME and LRP
* Cloud deployment support
---

# License

This project is developed for academic and research purposes.
