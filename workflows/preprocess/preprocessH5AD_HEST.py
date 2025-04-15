from tqdm import tqdm
import pandas as pd
import scanpy as sc
import numpy as np
import json
import sys
import os
import glob

import sys
sys.path.append('../')

SAMPLE = str(sys.argv[1])
downsample_factor = int(sys.argv[2])
out_folder = str(sys.argv[3])

adata_in = f"data/h5ad/{SAMPLE}.h5ad"
adata_out = f"{out_folder}/data/h5ad/{SAMPLE}.h5ad"

adata = sc.read_h5ad(adata_in)
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_counts=1)

adata.obs['ground_truth'] = adata.obs.leiden.values
adata.var['gene_name'] = adata.var.index
print("ground_truth loaded...")
adata = adata.copy()
if not isinstance(adata.X, np.ndarray):
    adata.X = adata.X.toarray()
adata.write_h5ad(adata_out)
