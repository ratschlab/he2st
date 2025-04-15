from tqdm import tqdm
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import squidpy as sq
from scipy.stats.mstats import gmean
import json
import glob
import sys
import yaml

#import sys
#sys.path.append('../')
#from src.utils import select_highly_variable_genes

def reorder_genes(adata, gene_name_idx):
    adata = adata.copy()
    adata = adata[:, adata.var.index.isin(gene_name_idx)]
    X_new = pd.DataFrame(np.zeros(shape=(len(gene_name_idx), len(adata))), index=gene_name_idx)
    X_new.loc[adata.var.index] = adata.X.T
    return X_new.T


out_folder = 'out_benchmark'#str(sys.argv[1])
#top_n_genes_to_predict = int(sys.argv[2])

with open("config_dataset.yaml", "r") as stream:
    config_dataset = yaml.safe_load(stream)

samples = set(config_dataset["SAMPLE_LQ"])
present_genes = [sc.read_h5ad(f"out_benchmark/data/h5ad/{s}.h5ad", backed="r").var_names.values for s in samples]
present_genes = sorted({gene for sublist in present_genes for gene in sublist})

df = pd.read_csv("/cluster/home/knonchev/code/projects2024-cell-embeddings/data/metadata/hg38_gtf.csv")
df = df[~df.gene_name.isna()]
df = df[~df.gene_name.duplicated()]
gene_name = df.gene_name.values  # protein coding genes
gene_name = [g for g in gene_name if g in present_genes] # keep only protein coding genes that are observed in the training set
adatas = []
old_adatas = []
for sample in samples:
    adata = sc.read_h5ad(f"{out_folder}/data/h5ad/{sample}.h5ad")
    old_adatas.append(adata)
    X_new = reorder_genes(adata, gene_name)

    new_adata = ad.AnnData(X_new)
    new_adata.var_names = gene_name
    new_adata.obs["ground_truth"] = adata.obs["ground_truth"].values

    counts = pd.DataFrame(X_new.values, index=adata.obs_names.values, columns=gene_name)
    counts.index = [f"{b}_{sample}" for b in counts.index]
    counts.to_pickle(f"{out_folder}/data/inputX/{sample}.pkl")

    new_adata.obs["sampleID"] = sample
