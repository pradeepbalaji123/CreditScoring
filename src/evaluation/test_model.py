import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.config.optimizer.set_jit(False)
DATASETS = ["HMEQ", "AUSTRALIAN"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
print("TensorFlow Version:", tf.__version__)
print("=" * 50)
def plot_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Default","Default"],
                yticklabels=["Non-Default","Default"])
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
models = {}
roc_data = {}
for dataset in DATASETS:
    model_path = MODEL_DIR / f"{dataset}_model.keras"
    if model_path.exists():
        models[dataset] = tf.keras.models.load_model(model_path)
for DATASET in DATASETS:
    print(f"\nEvaluating {DATASET}")
    print("-"*50)
    if DATASET not in models:
        print("Model not found")
        continue
    model = models[DATASET]
    X_test = np.load(DATA_DIR / f"{DATASET}_test_images.npy")
    y_test = np.load(DATA_DIR / f"{DATASET}_test_labels.npy")
    X_test = X_test.astype(np.float32)
    X_test = np.expand_dims(X_test, axis=-1)
    print("Test shape:", X_test.shape)
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs[:,1])
    print("Accuracy:", round(accuracy,4))
    print("AUC:", round(auc,4))
    plot_confusion_matrix(y_test, y_pred, DATASET)
    fpr, tpr, _ = roc_curve(y_test, y_probs[:,1])
    roc_data[DATASET] = (fpr, tpr, auc)
print("\nEvaluation Completed.")
plt.figure(figsize=(6,5))
for dataset in roc_data:
    fpr, tpr, auc = roc_data[dataset]
    plt.plot(fpr, tpr, label=f"{dataset} (AUC = {auc:.4f})")
plt.plot([0,1],[0,1], linestyle="--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

