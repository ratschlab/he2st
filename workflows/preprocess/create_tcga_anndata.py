import glob
import yaml
import json
from tqdm import tqdm
import PIL
import math
import anndata as ad
import scanpy as sc
import pandas as pd
from openslide import open_slide
from src.preprocess_utils.preprocess_image import get_low_res_image
import numpy as np
from IPython.display import Image
import pyvips
import sys
sys.path.append('../')

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

downsample_factor = int(sys.argv[1])
spot_diameter = int(sys.argv[2])
spot_distance = int(sys.argv[3])
white_cutoff = int(sys.argv[4])
id_pair = int(sys.argv[5])
out_folder = str(sys.argv[6])
metadata_path = str(sys.argv[7])
source_data_path = str(sys.argv[8])

metadata = pd.read_csv(metadata_path)
metadata['id_pair'] = metadata['id_pair'].astype(int)
metadata = metadata.set_index('id_pair')
row = metadata.loc[id_pair]
image_path = row.image_path
rna_path = row.rna_path


adata_out = f'{out_folder}/data/h5ad/{id_pair}.h5ad'
json_out = f'{out_folder}/data/meta/{id_pair}.json'
image_path = glob.glob(f'data/*/*/{image_path}')[0]
rna_path = glob.glob(f'data/*/*/{rna_path}')[0]

# LOAD CONFIGS

genes = pd.read_csv(f"../{source_data_path}/out_benchmark/info_highly_variable_genes.csv")
selected_genes_bool = genes.isPredicted.values


# MAGNIFICATION

img = open_slide(image_path)
magnification = img.properties["aperio.AppMag"]


if magnification == "40":
    spot_distance = int(spot_distance * 2)
    spot_diameter = int(spot_diameter * 2)
    dot_size = 20
elif magnification == "20":
    downsample_factor = downsample_factor // 2
    dot_size = 20


image = pyvips.Image.new_from_file(image_path)

coord = []
for i, x in enumerate(range(spot_diameter + 1, image.height - spot_diameter - 1, spot_distance)):
    for j, y in enumerate(range(spot_diameter + 1, image.width - spot_diameter - 1, spot_distance)):
        coord.append([i, j, x, y])
coord = pd.DataFrame(coord, columns=['x_array', 'y_array', 'x_pixel', 'y_pixel'])
coord.index = coord.index.astype(str)

is_white = []
counts = []
for _, row in tqdm(coord.iterrows()):
    x = row.x_pixel - int(spot_diameter // 2)
    y = row.y_pixel - int(spot_diameter // 2)

    spot = image.crop(y, x, spot_diameter, spot_diameter)
    try:
        main_tile = np.ndarray(buffer=spot.write_to_memory(),
                               dtype=format_to_dtype[spot.format],
                               shape=[spot.height, spot.width, spot.bands])
        main_tile = main_tile[:, :, :3]
        white = np.mean(main_tile)
    except BaseException:
        white = 255  # error
        image = pyvips.Image.new_from_file(image_path)

    is_white.append(white)

counts = np.empty((len(is_white), selected_genes_bool.sum()))  # empty count matrix

coord['is_white'] = is_white

# CREATE ANNDATA

adata = ad.AnnData(counts)
adata.var.index = genes[selected_genes_bool].gene_name.values
adata.obs = adata.obs.merge(coord, left_index=True, right_index=True)
adata.obs['is_white'] = coord['is_white'].values
adata.obs['is_white_bool'] = (coord['is_white'].values > white_cutoff).astype(int)
adata.obs['sampleID'] = id_pair
adata.obs['barcode'] = adata.obs.index
adata = adata[adata.obs.is_white_bool == 0, ]

# CREATE IMAGE

n_level = len(img.level_dimensions) - 1  # 0 based


large_w, large_h = img.dimensions
new_w = math.floor(large_w / downsample_factor)
new_h = math.floor(large_h / downsample_factor)

whole_slide_image = img.read_region((0, 0), n_level, img.level_dimensions[-1])
whole_slide_image = whole_slide_image.convert("RGB")
img_downsample = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)


adata.obsm['spatial'] = adata.obs[["y_pixel", "x_pixel"]].values
# adjust coordinates to new image dimensions
adata.obsm['spatial'] = adata.obsm['spatial'] / downsample_factor
# create 'spatial' entries
adata.uns['spatial'] = dict()
adata.uns['spatial']['library_id'] = dict()
adata.uns['spatial']['library_id']['images'] = dict()
adata.uns['spatial']['library_id']['images']['hires'] = np.array(img_downsample)

# LOAD BULK RNA

expr_bulk = pd.read_csv(rna_path, sep='\t', skiprows=1)
expr_bulk = expr_bulk[4:]
expr_bulk = expr_bulk[expr_bulk.gene_name.isin(adata.var.index)]
expr_bulk = expr_bulk[~expr_bulk.gene_name.duplicated()]
expr_bulk = expr_bulk.set_index('gene_name')
# expr_bulk = expr_bulk.loc[adata.var.index]
expr_bulk = expr_bulk.reindex(adata.var.index)
expr_bulk = expr_bulk.fillna(0)
expr_bulk = expr_bulk[['tpm_unstranded', 'fpkm_unstranded']]

# RAW
adata.var['bulk_tpm_unstranded'] = expr_bulk['tpm_unstranded'].values
adata.var['bulk_fpkm_unstranded'] = expr_bulk['fpkm_unstranded'].values

# NORM

expr_bulk.values[:] = np.log1p((expr_bulk.values / expr_bulk.values.sum(axis=0)) * 10000)

adata.var['bulk_norm_tpm_unstranded'] = expr_bulk['tpm_unstranded'].values
adata.var['bulk_norm_fpkm_unstranded'] = expr_bulk['fpkm_unstranded'].values

adata.write(adata_out)

# METADATA


json_info = {}
json_info['image_path'] = image_path
json_info['rna_path'] = rna_path
json_info['downsample_factor'] = downsample_factor
json_info['spot_diameter_fullres'] = spot_diameter
json_info['spot_distance'] = spot_distance
json_info['white_cutoff'] = white_cutoff
json_info['magnification'] = magnification
json_info['dot_size'] = dot_size

with open(json_out, 'w') as f:
    f.write(json.dumps(json_info))
