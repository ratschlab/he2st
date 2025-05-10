from sklearn.neural_network import MLPClassifier
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd

class CellTypeModel:
    
    def __init__(self, resolution=0.1):
        self.resolution = resolution
        self.clf = MLPClassifier(early_stopping=True)
        self.mean_expr_dict = {}

    def fit(self, X, y, batch):
        adata = ad.AnnData(y)
        adata.obs["batch"] = np.array(batch)
        sc.tl.pca(adata)
        sc.external.pp.harmony_integrate(adata, key="batch")
        sc.pp.neighbors(adata, use_rep="X_pca_harmony")
        sc.tl.leiden(adata, resolution=self.resolution)
        adata.obs.leiden = adata.obs.leiden.astype(int)
        # Convert expression matrix to DataFrame (genes as columns)
        expr_df = pd.DataFrame(adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X,
                               index=adata.obs_names,
                               columns=adata.var_names)
        
        # Add cluster labels
        expr_df['cluster'] = adata.obs['leiden'].values
        
        # Group by cluster and compute mean expression
        cluster_means = expr_df.groupby('cluster').mean()

        self.mean_expr_dict = {
            cluster: np.array(list(row.values))
            for cluster, row in cluster_means.iterrows()
        }
        self.clf.fit(X, adata.obs['leiden'].values)

    def predict(self, X):
        cell_types = self.clf.predict(X)
        expression = np.array([self.mean_expr_dict[l] for l in cell_types])
        return expression