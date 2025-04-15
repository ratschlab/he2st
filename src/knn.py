import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


import torch
import torch.nn.functional as F

def find_matches_cos_similarity(spot_vectors, query_vectors, n_neighbors=1, batch_size=32, device='cuda'):
    """
    Find the top-k most similar scRNA-Seq cells for each Xenium cell using cosine similarity with GPU acceleration and batching.

    Args:
        spot_vectors (array): 2D array of gene expression of scRNA-Seq cells.
        query_vectors (array): 2D array of gene expression of Xenium cells.
        n_neighbors (int): Number of top matches to return.
        batch_size (int): Number of query vectors to process per batch.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        tuple: Indices and similarity values of top-k matches.
    """
    # Convert input arrays to PyTorch tensors and move to device
    spot_vectors = torch.tensor(spot_vectors, dtype=torch.float32, device=device)
    query_vectors = torch.tensor(query_vectors, dtype=torch.float32, device=device)
    
    # Normalize vectors
    spot_vectors = F.normalize(spot_vectors, p=2, dim=-1)
    query_vectors = F.normalize(query_vectors, p=2, dim=-1)
    
    # Process in batches
    num_queries = query_vectors.shape[0]
    all_indices = []
    all_values = []
    
    for i in range(0, num_queries, batch_size):
        batch_queries = query_vectors[i:i+batch_size]
        dot_similarity = batch_queries @ spot_vectors.T
        values, indices = torch.topk(dot_similarity, k=n_neighbors, dim=-1)
        
        all_indices.append(indices.cpu())
        all_values.append(values.cpu())
    
    return torch.cat(all_values).numpy(), torch.cat(all_indices).numpy()

class KNeighborsRegressorTorch:
    def __init__(self, n_neighbors=5, batch_size=4096):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X = X
        #self.y = y

    def kneighbors(self, X, n_neighbors=None):
        n_neighbors = n_neighbors if n_neighbors else self.n_neighbors
        return find_matches_cos_similarity(spot_vectors=self.X, query_vectors=X, n_neighbors=n_neighbors, batch_size=self.batch_size)


class EnhancedKNNRegressor:
    def __init__(self, n_neighbors=100, batch_size=32):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.knn = KNeighborsRegressorTorch(n_neighbors=self.n_neighbors, batch_size=batch_size)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.knn.fit(self.X, self.y)

    def predict(self, X, n=None):
        n = n if n else self.n_neighbors
        weights, neighbors = self.knn.kneighbors(X, n)
        predictions_raw = np.array([np.mean(self.y[neighbor], axis=0) for neighbor in neighbors])
        return predictions_raw

from collections import Counter

class EnhancedKNNClassifier:
    def __init__(self, n_neighbors=100, batch_size=32):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsRegressorTorch(n_neighbors=self.n_neighbors, batch_size=batch_size)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.knn.fit(self.X, self.y)

    def predict(self, X, n=None):
        n = n if n else self.n_neighbors
        weights, neighbors = self.knn.kneighbors(X, n)
        # Majority voting
        predictions_raw = np.array([
            Counter(self.y[neighbor]).most_common(1)[0][0]
            for neighbor in neighbors
        ])
        return predictions_raw