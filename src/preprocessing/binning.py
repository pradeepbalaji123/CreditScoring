import pandas as pd
from .woe_iv import compute_woe_iv
PAPER_BINS = {
    "AUSTRALIAN": {
        "A2": [(13.75,24.08),(24.17,34.08),(34.17,80.25)],
        "A3": [(0,1.5),(1.54,5.04),(5.09,28.00)],
        "A4": [1,2,3],
        "A5": list(range(1,15)),
        "A6": list(range(1,10)),
        "A7": [(0,0.17),(0.21,1),(1.54,2.63),(2.71,28.5)],
        "A8": [0,1],
        "A9": [0,1],
        "A10": [(0,1),(2,67)],
        "A12": [1,2,3],
        "A13": [(0,160),(163,2000)],
        "A14": [(1,6),(7,100001)],
    },
    "HMEQ": {
        "LOAN": [(1100,12900),(13000,20800),(20900,89900)],
        "MORTDUE": [(2063,42000),(42003,61712),
                    (61767,88197),(88210,399550)],
        "VALUE": [(8000,72000),(72045,106431),(10645,85590)],
        "JOB": ["Mgr","Office","Other","ProfExe","Sales","Self"],
        "DEROG": [(0,1),(2,10)],
        "DELINQ": [(0,1),(2,15)],
        "CLAGE": [(0,90),(91,107),(108,129),(130,166),
                  (167,194),(195,226),(227,278),(279,1168)],
        "NINQ": [(0,1),(2,7)],
        "DEBTINC": [(0.52,31.92),(31.93,203.31)]
        
    }
}
def bin_continuous(df, feature, target, dataset_name):
    if dataset_name not in PAPER_BINS:
        raise ValueError(f"{dataset_name} not defined in PAPER_BINS")
    if feature not in PAPER_BINS[dataset_name]:
        raise ValueError(f"{feature} not part of paper-selected features")
    intervals = PAPER_BINS[dataset_name][feature]
    if not isinstance(intervals[0], tuple):
        raise ValueError(f"{feature} is not defined as continuous in paper")
    bins_list = []
    for lower, upper in intervals:
        subset = df[(df[feature] >= lower) & (df[feature] <= upper)]
        total = len(subset)
        bads = subset[target].sum()
        goods = total - bads
        bins_list.append({
            "lower": lower,
            "upper": upper,
            "total": total,
            "goods": goods,
            "bads": bads
        })
    bins = pd.DataFrame(bins_list)
    bins, IV = compute_woe_iv(bins)
    return bins, IV
def bin_categorical(df, feature, target, dataset_name):
    if dataset_name not in PAPER_BINS:
        raise ValueError(f"{dataset_name} not defined in PAPER_BINS")
    if feature not in PAPER_BINS[dataset_name]:
        raise ValueError(f"{feature} not part of paper-selected features")
    categories = PAPER_BINS[dataset_name][feature]
    if isinstance(categories[0], tuple):
        raise ValueError(f"{feature} is not defined as categorical in paper")
    bins_list = []
    for cat in categories:
        subset = df[df[feature] == cat]
        total = len(subset)
        bads = subset[target].sum()
        goods = total - bads
        bins_list.append({
            feature: cat,
            "total": total,
            "goods": goods,
            "bads": bads
        })
    bins = pd.DataFrame(bins_list)
    bins, IV = compute_woe_iv(bins)
    return bins, IV