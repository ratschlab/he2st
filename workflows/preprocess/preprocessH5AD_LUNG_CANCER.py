import pyvips
from tqdm import tqdm
from PIL import Image
import pandas as pd
import scanpy as sc
import numpy as np
import json
import sys
import os
import glob
import cv2

import sys
sys.path.append('../')
from src.preprocess_utils.preprocess_image import get_low_res_image

import glob
import scanpy as sc
from src.preprocess_utils.preprocess_image import get_low_res_image
import yaml
import anndata as ad
import json

with open("image_visium_match.yaml", "r") as stream:
    he_to_visium = yaml.safe_load(stream)
visium_to_he = {v:k for k,v in he_to_visium.items()}

sample = str(sys.argv[1])
org_sample = sample.replace("-", "_")
downsample_factor = int(sys.argv[2])
out_folder = str(sys.argv[3])

image_file = f"data/LUNG_VISIUM/gdrive/{visium_to_he[org_sample]}_cropped_90-rotated_dsmpled2.tif"
json_file = f"data/LUNG_VISIUM/ebi_downloads/{org_sample}-spatial/scalefactors_json.json"
adata_file = f"data/LUNG_VISIUM/ebi_downloads/{org_sample}-filtered_feature_bc_matrix.h5"
adata_out = f"{out_folder}/data/h5ad/{sample}.h5ad"
tissue_positions_list = f"data/LUNG_VISIUM/ebi_downloads/{org_sample}-spatial/tissue_positions_list.csv"


spot_diameter_fullres = json.load(open(json_file))["spot_diameter_fullres"]
spot_diameter_fullres

adata = sc.read_10x_h5(adata_file)
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_counts=1)

clusters = pd.read_csv("data/sample_leiden_cluster.csv", index_col=0)
clusters["barcode"] = clusters.index.str.split("_").str[0]
clusters["sample"] = clusters.index.to_series().apply(lambda x: "_".join(x.split("_")[1:]))
clusters = clusters.query(f"sample == '{org_sample}'")
clusters = clusters.set_index("barcode")


tissue_positions_list = pd.read_csv(tissue_positions_list, header=None)
tissue_positions_list.columns = ["barcode", "in_tissue", "x_array", "y_array", "x_pixel", "y_pixel"]
tissue_positions_list = tissue_positions_list.set_index("barcode")

adata.obs = adata.obs.merge(tissue_positions_list, left_index=True, right_index=True)
adata.obs["ground_truth"] = clusters.loc[adata.obs.index].leiden.values

img = get_low_res_image(image_file, downsample_factor)

adata.obsm['spatial'] = adata.obs[["y_pixel", "x_pixel"]].values
# adjust coordinates to new image dimensions
adata.obsm['spatial'] = adata.obsm['spatial'] / downsample_factor
# create 'spatial' entries
adata.uns['spatial'] = dict()
adata.uns['spatial']['library_id'] = dict()
adata.uns['spatial']['library_id']['images'] = dict()
adata.uns['spatial']['library_id']['images']['hires'] = img


adata = adata.copy()

if isinstance(adata.X, np.ndarray):
    pass
else:
    adata.X = adata.X.toarray()


adata.write_h5ad(adata_out)
















