from src.preprocess_utils.preprocess_image import get_low_res_image
from aestetik.utils.utils_transcriptomics import preprocess_adata
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm
from PIL import Image
import pandas as pd
import scanpy as sc
import numpy as np
import pyvips
import json
import sys
import os
import glob
import cv2

import sys
sys.path.append('../')

sample = str(sys.argv[1])
downsample_factor = int(sys.argv[2])
out_folder = str(sys.argv[3])

mtx_in_file = f"data/o30773_SpaceRangerCount_v2_1_0_2023-05-18--18-23-28/{sample}/filtered_feature_bc_matrix/"
tissue_positions_file = f"data/o30773_SpaceRangerCount_v2_1_0_2023-05-18--18-23-28/{sample}/spatial/tissue_positions.csv"
image_file = f"data/o30773_SpaceRangerCount_v2_1_0_2023-05-18--18-23-28/manual_loupe_alignment/{sample}.tif"
manual_position_file = f"data/o30773_SpaceRangerCount_v2_1_0_2023-05-18--18-23-28/manual_loupe_alignment/{sample}.json"
annotation_path = f"data/aestetik_supervised_clusters/aestetik_{sample}_supervised_1.5_bgm.csv"

adata_out = f"{out_folder}/data/h5ad/{sample}.h5ad"

refined_cluster = pd.read_csv(annotation_path, index_col=0)


# adata = sc.read_10x_h5(h5_file)
adata = sc.read_10x_mtx(mtx_in_file)
tissue_positions = pd.read_csv(tissue_positions_file, index_col=0)
tissue_positions.drop({"pxl_row_in_fullres", "pxl_col_in_fullres"}, axis=1, inplace=True)

manual_position = json.load(open(manual_position_file))
manual_position = pd.DataFrame.from_dict(manual_position['oligo'])
tissue_positions = tissue_positions.reset_index().merge(
    manual_position, left_on=[
        'array_row', 'array_col'], right_on=[
            'row', 'col']).set_index('barcode')


adata.obs["in_tissue"] = tissue_positions.loc[adata.obs.index].in_tissue
adata.obs["x_array"] = tissue_positions.loc[adata.obs.index].array_row
adata.obs["y_array"] = tissue_positions.loc[adata.obs.index].array_col
adata.obs["x_pixel"] = tissue_positions.loc[adata.obs.index].imageY
adata.obs["y_pixel"] = tissue_positions.loc[adata.obs.index].imageX
adata = adata[adata.obs.in_tissue == 1, :]  # only spots with tissue
adata.var["gene_name"] = adata.var.index

scale_factors = tissue_positions.dia.values[0]

img = get_low_res_image(image_file, downsample_factor=downsample_factor)

adata.obsm['spatial'] = adata.obs[["y_pixel", "x_pixel"]].values
# adjust coordinates to new image dimensions
adata.obsm['spatial'] = adata.obsm['spatial'] / downsample_factor
# create 'spatial' entries
adata.uns['spatial'] = dict()
adata.uns['spatial']['library_id'] = dict()
adata.uns['spatial']['library_id']['images'] = dict()
adata.uns['spatial']['library_id']['images']['hires'] = img

adata.obs = adata.obs.merge(refined_cluster, left_index=True, right_index=True, how="left")
adata = adata[adata.obs["manual_anno"] != "EXCL", :]  # remove empty spots
adata = adata[~adata.obs["manual_anno"].isna(), :]  # remove empty spots
adata.obs["ground_truth"] = adata.obs["manual_anno"].values


# remove blur

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lapl = cv2.Laplacian(gray, cv2.CV_64F)

scores = []
for _, row in adata.obs.iterrows():
    x_start = int(row.x_pixel - scale_factors / 2)
    x_end = int(row.x_pixel + scale_factors)

    y_start = int(row.y_pixel - scale_factors / 2)
    y_end = int(row.y_pixel + scale_factors)

    score = lapl[x_start:x_end, y_start:y_end].std()
    scores.append(score)

adata.obs["blur_score"] = scores
adata = adata[adata.obs["blur_score"] > 10, :]


# clusters = adata.obs.ground_truth.value_counts()[adata.obs.ground_truth.value_counts() / len(adata) >= 0.01].index.values
# adata = adata[adata.obs.ground_truth.isin(clusters),:] # remove rare clusters 1% (noise)
adata = adata.copy()
print("ground_truth loaded...")
if isinstance(adata.X, if)    pass
else:
    adata.X = adata.X.toarray()
adata.write_h5ad(adata_out)
