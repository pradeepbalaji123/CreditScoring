import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="shap")
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
from .grad_cam import make_gradcam_heatmap
from .saliency import compute_saliency
from .shap_explain import compute_shap
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "src/models/AUSTRALIAN_model.keras"
DATA_PATH = PROJECT_ROOT / "data/processed/AUSTRALIAN_test_images.npy"
print("Loading model from:", MODEL_PATH)
print("Loading data from:", DATA_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
images = np.load(DATA_PATH)
sample = images[23]
sample = np.expand_dims(sample, axis=(0, -1))
prediction = model.predict(sample)
print("\nPrediction probability:", prediction)
saliency = compute_saliency(model, sample)
plt.figure(figsize=(6,5))
plt.imshow(saliency[0], cmap="hot")
plt.title("Saliency Map")
plt.colorbar()
plt.show()
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break
print("Last Conv Layer:", last_conv_layer)
heatmap = make_gradcam_heatmap(sample, model, last_conv_layer)
plt.figure(figsize=(6,5))
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.colorbar()
plt.show()
background = images[:50]
background = np.expand_dims(background, axis=-1)
shap_sample = sample
print("\nRunning SHAP explanation...")
shap_values = compute_shap(model, background, shap_sample)
shap.image_plot(shap_values, shap_sample)