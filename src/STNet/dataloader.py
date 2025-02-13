from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
