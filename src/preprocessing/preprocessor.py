import pickle
import numpy as np
class CreditPreprocessor:
    def __init__(self, bin_info, selected_features):
        self.bin_info = bin_info
        self.selected_features = selected_features
        self.selected_cols = list(selected_features.keys())
        self.B = max(len(bin_info[col]) for col in self.selected_cols)
        self.D = len(self.selected_cols)
    def transform(self, df):
        N = len(df)
        images = np.zeros((N, self.B, self.D), dtype=np.uint8)
        for d, col in enumerate(self.selected_cols):
            bins = self.bin_info[col]
            if 'lower' in bins.columns:
                for i in range(N):
                    value = df.iloc[i][col]
                    for b in range(len(bins)):
                        lower = bins.iloc[b]['lower']
                        upper = bins.iloc[b]['upper']
                        if b < len(bins)-1:
                            if lower <= value < upper:
                                images[i,b,d] = 1
                                break
                        else:
                            if lower <= value <= upper:
                                images[i,b,d] = 1
                                break
            else:
                for i in range(N):
                    value = df.iloc[i][col]
                    for b in range(len(bins)):
                        if value == bins.iloc[b][col]:
                            images[i,b,d] = 1
                            break
        return images
    def save(self, path):
        with open(path,"wb") as f:
            pickle.dump(self,f)
    @staticmethod
    def load(path):
        with open(path,"rb") as f:
            return pickle.load(f)