from imblearn.over_sampling import SMOTE
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .binning import bin_continuous, bin_categorical
from .feature_selection_image_generation import (
    build_bins_and_select,
    convert_to_images
)
def save_dataset(name, images, labels, features, B, D):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SAVE_PATH = PROJECT_ROOT / "data" / "processed"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    np.save(SAVE_PATH / f"{name}_images.npy", images)
    np.save(SAVE_PATH / f"{name}_labels.npy", labels)
    metadata = {"features": features, "B": B, "D": D}
    with open(SAVE_PATH / f"{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\nSaved {name} → {SAVE_PATH}")
    print("Image shape:", images.shape)
def run_pipeline(df, target, name):
    print(f"\nProcessing {name}...")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SAVE_PATH = PROJECT_ROOT / "data" / "processed"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df[target],
        random_state=42
    )
    bin_info, selected = build_bins_and_select(
        train_df,
        target,
        iv_threshold=0.1,
        bin_continuous=bin_continuous,
        bin_categorical=bin_categorical,
        dataset_name=name
    )
    print("\nSelected Features (IV ≥ 0.1):")
    for f, iv in selected.items():
        print(f"  {f}: {iv:.4f}")
    import pickle
    with open(SAVE_PATH / f"{name}_bin_info.pkl", "wb") as f:
        pickle.dump(bin_info, f)
    with open(SAVE_PATH / f"{name}_selected_features.pkl", "wb") as f:
        pickle.dump(selected, f)
    X_train, y_train, B, D, features = convert_to_images(
        train_df,
        target,
        bin_info,
        selected
    )
    X_test, y_test, _, _, _ = convert_to_images(
        test_df,
        target,
        bin_info,
        selected
    )
    save_dataset(name + "_train", X_train, y_train, features, B, D)
    save_dataset(name + "_test", X_test, y_test, features, B, D)
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    hmeq = pd.read_csv(DATA_RAW / "hmeq.csv")
    hmeq = hmeq.dropna(subset=['BAD'])
    run_pipeline(hmeq, "BAD", "HMEQ")
    australian = pd.read_excel(DATA_RAW / "Australian_data.xlsx")
    aus_target = australian.columns[-1]
    run_pipeline(australian, aus_target, "AUSTRALIAN")