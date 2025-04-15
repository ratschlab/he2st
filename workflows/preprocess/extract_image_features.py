from tqdm import tqdm
import torch
import os
from src.preprocess_utils.preprocess_image import compute_mini_tiles
from src.morphology_model import get_morphology_model_and_preprocess
import pyvips
import pandas as pd
import scanpy as sc
import numpy as np
import json
import glob
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

sample = str(sys.argv[1])
image_feature_model = str(sys.argv[2])
n_mini_tiles = int(sys.argv[3])  # 4 or 9
out_folder = str(sys.argv[4])

adata_in = f"{out_folder}/data/h5ad/{sample}.h5ad"
json_path = f"{out_folder}/data/meta/{sample}.json"
img_path = glob.glob(f"{out_folder}/data/image/{sample}*")[0]

image_feature_model_features_out = f"{out_folder}/data/image_features/{sample}_{image_feature_model}.pkl"

adata = sc.read_h5ad(adata_in)

spot_diameter_fullres = round(json.load(open(json_path))["spot_diameter_fullres"])

image = pyvips.Image.new_from_file(img_path)

device = torch.device("cuda")
morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(
    model_name=image_feature_model, device=device)


morphology_model = morphology_model.to(device)
barcode = adata.obs_names
x_pixel = adata.obs.x_pixel
y_pixel = adata.obs.y_pixel


image = pyvips.Image.new_from_file(img_path)
main_features = np.zeros([len(adata), feature_dim])

for i, (b, x, y) in tqdm(enumerate(zip(barcode, y_pixel, x_pixel))):  # x and y switched

    x = x - int(spot_diameter_fullres // 2)
    y = y - int(spot_diameter_fullres // 2)

    spot = image.crop(x, y, spot_diameter_fullres, spot_diameter_fullres)
    main_tile = np.ndarray(buffer=spot.write_to_memory(),
                           dtype=format_to_dtype[spot.format],
                           shape=[spot.height, spot.width, spot.bands])

    preprocess_main_tile = preprocess(main_tile)

    X = np.zeros([n_mini_tiles + 1, 3, preprocess_main_tile.shape[1], preprocess_main_tile.shape[1]])
    X[0, :] = preprocess_main_tile

    mini_tiles = compute_mini_tiles(main_tile, 4)

    for j, mini_tile in enumerate(mini_tiles):

        X[j + 1, :] = preprocess(mini_tile)

    X = torch.from_numpy(X)
    X = X.to(device)
    X = X.float()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            output = morphology_model(X)
            output = output.detach().cpu().numpy()
            assert not np.isnan(output).any()

    main_features[i, :] = output[0]

    np.save(f"{out_folder}/data/image_features/{image_feature_model}/{b}_{sample}.npy", output)

main_features = pd.DataFrame(main_features, index=adata.obs.index)
main_features.index = [f"{b}_{sample}" for b in main_features.index]
main_features.to_pickle(image_feature_model_features_out)
