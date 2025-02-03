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

import sys
sys.path.append('../')

SAMPLE = str(sys.argv[1])
org_sample_name = SAMPLE.replace('-1-', '-1_').replace('-2-', '-2_')
downsample_factor = int(sys.argv[2])
out_folder = str(sys.argv[3])

adata_h5 = glob.glob(f"data/spaceranger/*/results/spaceranger/{org_sample_name}/outs/filtered_feature_bc_matrix.h5")[0]
annotation_path = f"data/spot_annotations_kalin/{org_sample_name}.csv"
adata_out = f"{out_folder}/data/h5ad/{SAMPLE}.h5ad"
img_path = glob.glob(f"data/tif_image/*/{org_sample_name}/{org_sample_name}.tif")[0]
spatial_coord_barcode_file = glob.glob(
    f"data/spaceranger/*/results/spaceranger/{org_sample_name}/outs/spatial/tissue_positions.csv")[0]
spatial_coord_px_file = glob.glob(f"data/tif_image/*/{org_sample_name}/*.json")[0]

image_info_file = f"data/10x_image_info_updated.tsv"

df = pd.read_csv(image_info_file, sep="\t")
replicate_info = df[df.replicate == org_sample_name]

tile_size = replicate_info.spot_diameter_fullres.values[0]

if tile_size == 317:
    downsample_factor = 15


adata = sc.read_10x_h5(adata_h5)
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_counts=1)

spatial_coord_px = json.load(open(spatial_coord_px_file))
spatial_coord_px = pd.DataFrame.from_dict(spatial_coord_px["oligo"], orient="columns")
spatial_coord_px = spatial_coord_px[spatial_coord_px.tissue]
spatial_coord_barcode = pd.read_csv(spatial_coord_barcode_file)
spatial_coord_barcode = spatial_coord_barcode[spatial_coord_barcode.in_tissue == 1][[
    "barcode", "array_row", "array_col"]]
spatial_coord = spatial_coord_barcode.merge(
    spatial_coord_px, left_on=[
        "array_row", "array_col"], right_on=[
            "row", "col"])
spatial_coord = spatial_coord[["barcode", "array_row", "array_col", "imageX", "imageY"]]

spatial_coord = spatial_coord.rename({
    "array_col": "y_array",
    "array_row": "x_array",
    "imageX": "y_pixel",
    "imageY": "x_pixel"}, axis=1)
adata = adata[adata.obs.index.isin(spatial_coord.barcode.values), :]  # only in_tissue spots
adata.obs = adata.obs.merge(spatial_coord, left_index=True, right_on="barcode").set_index("barcode")
adata.obs.index.rename("index", inplace=True)

adata.var_names = [i.upper() for i in list(adata.var_names)]
adata.var["gene_name"] = adata.var.index.astype("str")

img = get_low_res_image(img_path, downsample_factor)

adata.obsm['spatial'] = adata.obs[["y_pixel", "x_pixel"]].values
# adjust coordinates to new image dimensions
adata.obsm['spatial'] = adata.obsm['spatial'] / downsample_factor
# create 'spatial' entries
adata.uns['spatial'] = dict()
adata.uns['spatial']['library_id'] = dict()
adata.uns['spatial']['library_id']['images'] = dict()
adata.uns['spatial']['library_id']['images']['hires'] = img

annotations = pd.read_csv(annotation_path)
annotations.barcode = annotations.barcode.apply(lambda x: f"{x}-1")

adata.obs["barcode"] = adata.obs_names.values

adata = adata[adata.obs.barcode.isin(annotations.barcode), :]  # tbd

adata.obs = adata.obs.merge(annotations, on=["barcode"]).set_index("barcode")
adata.obs.index = adata.obs.index.rename("index")
adata.obs["ground_truth"] = adata.obs.label.values

# remove bad labels

adata = adata[adata.obs.label != "Whitespace", :]
adata = adata[adata.obs.label != "Mixed", :]

# clusters = adata.obs.ground_truth.value_counts()[adata.obs.ground_truth.value_counts() / len(adata) >= 0.01].index.values
# adata = adata[adata.obs.ground_truth.isin(clusters),:] # remove rare clusters 1% (noise)

print("ground_truth loaded...")

adata = adata.copy()

if isinstance(adata.X, if )    pass
else:
    adata.X = adata.X.toarray()


adata.write_h5ad(adata_out)
