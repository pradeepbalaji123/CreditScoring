import numpy as np
from .binning import PAPER_BINS
def build_bins_and_select(df, target, iv_threshold, bin_continuous, bin_categorical, dataset_name):
    if dataset_name not in PAPER_BINS:
        raise ValueError(f"{dataset_name} not defined in PAPER_BINS")
    bin_info = {}
    iv_dict = {}
    paper_features = list(PAPER_BINS[dataset_name].keys())
    for col in paper_features:
        definition = PAPER_BINS[dataset_name][col]
        if isinstance(definition[0], tuple):
            bins, IV = bin_continuous(df, col, target, dataset_name)
        else:
            bins, IV = bin_categorical(df, col, target, dataset_name)
        bin_info[col] = bins
        iv_dict[col] = IV
    print("\nIV Values (Paper Bins):")
    for f, iv in iv_dict.items():
        print(f"  {f}: {iv:.4f}")
    selected = {f: iv for f, iv in iv_dict.items() if iv >= iv_threshold}
    print(f"\nSelected Features (IV ≥ {iv_threshold}):")
    for f, iv in selected.items():
        print(f"  {f}: {iv:.4f}")
    return bin_info, selected
def convert_to_images(df, target, bin_info, selected_features):
    selected_cols = list(selected_features.keys())
    B = max(len(bin_info[col]) for col in selected_cols)
    D = len(selected_cols)
    N = len(df)
    images = np.zeros((N, B, D), dtype=np.uint8)
    labels = df[target].values
    for d, col in enumerate(selected_cols):
        bins = bin_info[col]
        if 'lower' in bins.columns:
            for i in range(N):
                value = df.iloc[i][col]
                for b in range(len(bins)):
                    lower = bins.iloc[b]['lower']
                    upper = bins.iloc[b]['upper']
                    if b < len(bins) - 1:
                        if lower <= value < upper:
                            images[i, b, d] = 1
                            break
                    else:
                        if lower <= value <= upper:
                            images[i, b, d] = 1
                            break
        else:
            for i in range(N):
                value = df.iloc[i][col]
                for b in range(len(bins)):
                    if value == bins.iloc[b][col]:
                        images[i, b, d] = 1
                        break
    return images, labels, B, D, selected_cols