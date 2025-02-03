import pyvips
import pandas as pd
import scanpy as sc
import numpy as np
import json
import glob
import sys
import os

from tqdm import tqdm
import numpy as np

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

sample = str(sys.argv[1])
out_folder = str(sys.argv[2])

adata_in = f"{out_folder}/data/h5ad/{sample}.h5ad"
json_path = f"{out_folder}/data/meta/{sample}.json"
img_path = glob.glob(f"{out_folder}/data/image/{sample}*")[0]

tiles_out = f"{out_folder}/data/tiles/{sample}.npy"

adata = sc.read_h5ad(adata_in)

spot_diameter_fullres = round(json.load(open(json_path))["spot_diameter_fullres"])

image = pyvips.Image.new_from_file(img_path)

barcode = adata.obs_names
x_pixel = adata.obs.x_pixel
y_pixel = adata.obs.y_pixel


tiles = np.zeros([len(adata), spot_diameter_fullres, spot_diameter_fullres, 3])

for i, (b, x, y) in tqdm(enumerate(zip(barcode, y_pixel, x_pixel))):

    x = x - int(spot_diameter_fullres // 2)
    y = y - int(spot_diameter_fullres // 2)

    spot = image.crop(x, y, spot_diameter_fullres, spot_diameter_fullres)
    spot = np.ndarray(buffer=spot.write_to_memory(),
                      dtype=format_to_dtype[spot.format],
                      shape=[spot.height, spot.width, spot.bands])

    tiles[i, :] = spot
    np.save(f"{out_folder}/data/tiles/{b}_{sample}.npy", spot)


np.save(tiles_out, tiles)
