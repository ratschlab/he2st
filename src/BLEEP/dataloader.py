import torchvision.transforms as transforms
from distutils.dir_util import copy_tree
from torch.utils.data import Dataset
from src.utils import load_data
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import numpy as np
import tempfile
import torch
import json
import os

from collections import Counter
import anndata as ad


class BLEEPCustomDataLoader(Dataset):
    def __init__(self,
                 out_folder,
                 samples,
                 genes_to_keep,
                 is_train,
                 morphology_model_name,
                 target_sum=10000):
        super().__init__()
        self.out_folder = out_folder
        self.samples = list(samples)
        self.genes_to_keep = genes_to_keep
        self.target_sum = target_sum
        self.is_train = is_train
        self.morphology_model_name = morphology_model_name
        self.cache = {}

        coordinates_df = []
        for sample in tqdm(samples):

            json_info = json.load(open(f"{out_folder}/data/meta/{sample}.json"))

            adata_path = f"{out_folder}/data/h5ad/{sample}.h5ad"

            adata = sc.read_h5ad(adata_path)
            coordinates = adata.obs  # pd.DataFrame(adata.obs_names.values, columns=["barcode"])
            coordinates["barcode"] = coordinates.index.values
            coordinates["sampleID"] = sample

            coordinates.index = [f"{i}_{sample}" for i in adata.obs_names]
            coordinates_df.append(coordinates)

        self.coordinates_df = pd.concat(coordinates_df)

        data = load_data(samples, self.out_folder, load_image_features=True, feature_model=self.morphology_model_name)
        self.transcriptomics_df = pd.DataFrame(data["y"][:, self.genes_to_keep],
                                               index=data["barcode"])

        self.image_features_df = pd.DataFrame(data["X"], index=data["barcode"])

        assert (self.transcriptomics_df.index == self.coordinates_df.index).all()
        assert (self.image_features_df.index == self.coordinates_df.index).all()

        self.image_feature_source = f"{self.out_folder}/data/image_features/{self.morphology_model_name}"

    def __len__(self):

        return len(self.coordinates_df)

    def __getitem__(self, idx):

        spot_info = self.coordinates_df.iloc[idx]

        image = self.image_features_df.iloc[idx].values

        coord = np.array([spot_info.x_array, spot_info.y_array])
        y = self.transcriptomics_df.loc[spot_info.name].values
        item = {}
        item["image_features"] = image.astype(np.float32)
        item['reduced_expression'] = y.astype(np.float32)
        item['barcode'] = spot_info.barcode
        item['spatial_coords'] = coord
        return item
