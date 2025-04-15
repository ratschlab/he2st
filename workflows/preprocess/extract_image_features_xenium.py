from tqdm import tqdm
import torch
import os
import pyvips
import pandas as pd
import scanpy as sc
import numpy as np
import json
import glob
import sys
sys.path.append('../')
from src.morphology_model import get_morphology_model_and_preprocess
from deepspot.utils.utils_image import crop_tile

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
cell_diameter = int(sys.argv[3])
out_folder = str(sys.argv[4])

adata_in = f"{out_folder}/data/h5ad/{sample}.h5ad"
json_path = f"{out_folder}/data/meta/{sample}.json"
img_path = glob.glob(f"{out_folder}/data/image/{sample}*")[0]


image_feature_model_features_out = f"{out_folder}/data/image_features/{image_feature_model}_{cell_diameter}/{sample}.pkl"

adata = sc.read_h5ad(adata_in)

image = pyvips.Image.new_from_file(img_path)

device = torch.device("cuda")
morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(
    model_name=image_feature_model, device=device)


morphology_model = morphology_model.to(device)
barcode = adata.obs_names
x_pixel = adata.obs.x_pixel
y_pixel = adata.obs.y_pixel


image = pyvips.Image.new_from_file(img_path)

main_features = []
batch_barcode = []
batch_X = []
batch_size = 128

for i, (b, x, y) in tqdm(enumerate(zip(barcode, x_pixel, y_pixel))): # x and y switched 

    patch = crop_tile(image, x, y, cell_diameter)
    X = preprocess(patch)
    #X = torch.from_numpy(X)
    X = X.unsqueeze(0)
    X = X.to(device)
    X = X.float()

    # Accumulate the preprocessed patches into a batch
    batch_X.append(X)
    batch_barcode.append(b)

    if len(batch_X) == batch_size:
        batch_tensor = torch.cat(batch_X, dim=0)  # Combine the list of tensors into one tensor
        
        
        # We recommend using mixed precision for faster inference.
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            with torch.inference_mode():
                output = morphology_model(batch_tensor)
                output = output.detach().cpu().numpy()
        assert not np.isnan(output).any(), f"NaN detected! {sample}_{b}_{x}_{y}"
                    
        main_features.append(output)

        #for b, embedding in zip(batch_barcode, output):
        #    np.save(f"{out_folder}/data/image_features/{image_feature_model}_{cell_diameter}/{sample}/{b}.npy", embedding)

        batch_barcode.clear() # Reset the batch list
        batch_X.clear()  # Reset the batch list
        
# Process any remaining patches if they exist
if batch_X and batch_barcode:
    batch_tensor = torch.cat(batch_X, dim=0)
    
    # We recommend using mixed precision for faster inference.
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        with torch.inference_mode():
            output = morphology_model(batch_tensor)
            output = output.detach().cpu().numpy()
    assert not np.isnan(output).any(), f"NaN detected! {sample}_{b}_{x_center}_{y_center}"
                
    main_features.append(output)

    #for b, embedding in zip(batch_barcode, output):
    #    np.save(f"{out_folder}/data/image_features/{image_feature_model}_{cell_diameter}/{sample}/{b}.npy", embedding)

main_features = np.concatenate(main_features, axis=0)
main_features = pd.DataFrame(main_features, index=adata.obs.index)
main_features.index = [f"{b}_{sample}" for b in main_features.index]
main_features.to_pickle(image_feature_model_features_out)
