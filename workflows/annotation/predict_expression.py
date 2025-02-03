import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
from src.utils import preprocess_adata
from pickle import load
import yaml
import json
from tqdm import tqdm
import squidpy as sq
import PIL
import math
import anndata as ad
import scanpy as sc
import pandas as pd
from openslide import open_slide
import torch
from src.morphology_model import get_morphology_model_and_preprocess
from src.preprocess_utils.preprocess_image import BLEEP_predict_spatial_transcriptomics_from_image_path
from src.preprocess_utils.preprocess_image import THItoGene_predict_spatial_transcriptomics_from_image_path
from src.preprocess_utils.preprocess_image import Hist2ST_predict_spatial_transcriptomics_from_image_path
from src.preprocess_utils.preprocess_image import HisToGene_predict_spatial_transcriptomics_from_image_path
from src.preprocess_utils.preprocess_image import sklearn_predict_spatial_transcriptomics_from_image_path
from deepspot.utils.utils_image import predict_spatial_transcriptomics_from_image_path
from src.preprocess_utils.preprocess_image import get_low_res_image
import numpy as np
from IPython.display import Image
import pyvips
import openslide
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

model = str(sys.argv[1])
sample = str(sys.argv[2])
out_folder = str(sys.argv[3])
source_data_path = str(sys.argv[4])


model_path = f'../{source_data_path}/out_benchmark/evaluation/{model}/final_model.pkl'
model_config_path = f'../{source_data_path}/out_benchmark/evaluation/{model}/top_param_overall.yaml'
adata_in = f'{out_folder}/data/h5ad/{sample}.h5ad'
adata_out = f'{out_folder}/prediction/{model}/data/h5ad/{sample}.h5ad'
json_path = f'{out_folder}/data/meta/{sample}.json'

if model in ["THItoGene", "HisToGene", "Hist2ST"]:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


json_info = json.load(open(json_path))
image_path = json_info['image_path'] if "image_path" in json_info else glob.glob(
    f"{out_folder}/data/image/{sample}*")[0]
spot_diameter = json_info['spot_diameter_fullres']

genes = pd.read_csv(f"../{source_data_path}/out_benchmark/info_highly_variable_genes.csv")
selected_genes_bool = genes.isPredicted.values

with open(model_config_path, "r") as stream:
    MODEL_PARAM = yaml.safe_load(stream)

image_feature_model = MODEL_PARAM.get("image_feature_model", None)
spot_context = MODEL_PARAM.get('spot_context', None)
top_k = MODEL_PARAM.get('top_k', None)

with open(f"../{source_data_path}/config_dataset.yaml", "r") as stream:
    config_dataset_source = yaml.safe_load(stream)

n_mini_tiles = config_dataset_source['n_mini_tiles']
training_samples = config_dataset_source.get("SAMPLE", None)

# LOAD EXPRESSION MODEL

if model in ["DeepSpot", "HisToGene", "THItoGene", "Hist2ST", "BLEEP"]:  # pytorch
    model_expression = torch.load(model_path, map_location=device)
    model_expression.to(device)
    model_expression.eval()
else:
    with open(model_path, 'rb') as f:
        model_expression = load(f)


# LOAD MORPHOLOGY MODEL

if image_feature_model:
    morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(
        model_name=image_feature_model, device=device)
    morphology_model.to(device)
    morphology_model.eval()
else:
    fig_size = model_expression.fig_size
    import torchvision.transforms as transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(fig_size),
    ])


adata = sc.read_h5ad(adata_in)
adata.obs["sampleID"] = str(sample)
adata.obs["barcode"] = adata.obs.index.astype(str)
adata.obs.index = adata.obs.index.astype(str)


if model == "DeepSpot":
    counts = predict_spatial_transcriptomics_from_image_path(image_path,
                                                             adata,
                                                             spot_diameter,
                                                             n_mini_tiles,
                                                             preprocess,
                                                             morphology_model,
                                                             model_expression,
                                                             device,
                                                             super_resolution=False,
                                                             neighbor_radius=1)
elif model == "HisToGene":
    counts = HisToGene_predict_spatial_transcriptomics_from_image_path(image_path,
                                                                       adata,
                                                                       spot_diameter,
                                                                       preprocess,
                                                                       model_expression,
                                                                       device)
elif model == "Hist2ST":
    counts = Hist2ST_predict_spatial_transcriptomics_from_image_path(image_path,
                                                                     adata,
                                                                     spot_diameter,
                                                                     preprocess,
                                                                     model_expression,
                                                                     device)
elif model == "THItoGene":
    counts = THItoGene_predict_spatial_transcriptomics_from_image_path(image_path,
                                                                       adata,
                                                                       spot_diameter,
                                                                       preprocess,
                                                                       model_expression,
                                                                       device)
elif model == "BLEEP":
    counts = BLEEP_predict_spatial_transcriptomics_from_image_path(image_path,
                                                                   adata,
                                                                   spot_diameter,
                                                                   preprocess,
                                                                   selected_genes_bool=selected_genes_bool,
                                                                   training_samples=training_samples,
                                                                   out_folder=f"../{source_data_path}/out_benchmark",
                                                                   morphology_model=morphology_model,
                                                                   image_feature_model=image_feature_model,
                                                                   model_expression=model_expression,
                                                                   top_k=top_k,
                                                                   device=device)
else:
    counts = sklearn_predict_spatial_transcriptomics_from_image_path(image_path,
                                                                     adata,
                                                                     spot_diameter,
                                                                     preprocess,
                                                                     morphology_model,
                                                                     model_expression,
                                                                     device)


if len(adata.var) != selected_genes_bool.sum():
    adata_predicted = ad.AnnData(counts, obs=adata.obs, uns=adata.uns, obsm=adata.obsm).copy()
    adata_predicted.var.index = genes[selected_genes_bool].gene_name.values
else:
    adata_predicted = adata.copy()
    adata_predicted.X = counts


adata_predicted.write_h5ad(adata_out)
