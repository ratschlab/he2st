from src.utils import select_highly_variable_genes
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

import sys
sys.path.append('../')


def reorder_genes(adata, gene_name_idx):
    adata = adata.copy()
    adata = adata[:, adata.var.index.isin(gene_name_idx)]
    X_new = pd.DataFrame(np.zeros(shape=(len(gene_name_idx), len(adata))), index=gene_name_idx)
    X_new.loc[adata.var.index] = adata.X.T
    return X_new.T


out_folder = str(sys.argv[1])
top_n_genes_to_predict = int(sys.argv[2])

with open("config_dataset.yaml", "r") as stream:
    config_dataset = yaml.safe_load(stream)

samples = set(config_dataset["SAMPLE"])

df = pd.read_csv("/cluster/home/knonchev/code/projects2024-cell-embeddings/data/metadata/hg38_gtf.csv")
df = df[~df.gene_name.isna()]
df = df[~df.gene_name.duplicated()]
gene_name = df.gene_name.values  # protein coding genes

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

    adatas.append(new_adata)

adatas = ad.concat(adatas)
old_adatas = ad.concat(old_adatas)  # needed to compute overlaping genes present in all samples

adatas.var["gene_name"] = adatas.var.index.values

tab = adatas[:, adatas.var.index.isin(old_adatas.var.index)].copy()
sc.pp.highly_variable_genes(tab, flavor='seurat_v3_paper', batch_key="sampleID", n_top_genes=100000)
genes = tab.var.sort_values(["highly_variable_rank", "variances_norm"], ascending=[True, False])
genes["variances_norm_rank"] = list(range(1, len(genes) + 1))
genes = adatas.var.merge(genes, how="left", on="gene_name")
genes = adatas.var.merge(genes, how="left")
genes["isPresentInAll"] = genes.gene_name.isin(old_adatas.var.index)
genes["isPredicted"] = genes.variances_norm_rank <= top_n_genes_to_predict

adatas.write_h5ad(f"{out_folder}/data/h5ad/all_samples.h5ad")
genes.to_csv(f"{out_folder}/info_highly_variable_genes.csv", index=False)
