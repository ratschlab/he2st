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


class HisToGeneCustomDataLoader(Dataset):
    def __init__(self,
                 out_folder,
                 samples,
                 genes_to_keep,
                 is_train,
                 patch_size=160,
                 sample_n=128,
                 target_sum=10000):
        super().__init__()
        self.out_folder = out_folder
        self.samples = list(samples)
        self.genes_to_keep = genes_to_keep
        self.target_sum = target_sum
        self.is_train = is_train
        self.patch_size = patch_size
        self.sample_n = sample_n
        coordinates_df = []
        self.cache = {}
        for sample in tqdm(samples):

            json_info = json.load(open(f"{out_folder}/data/meta/{sample}.json"))
            patch_size = int(json_info["spot_diameter_fullres"]) + 1
            if self.is_train:
                self.patch_size = min(self.patch_size, patch_size)

            adata_path = f"{out_folder}/data/h5ad/{sample}.h5ad"

            adata = sc.read_h5ad(adata_path)
            coordinates = adata.obs  # pd.DataFrame(adata.obs_names.values, columns=["barcode"])
            coordinates["barcode"] = coordinates.index.values
            coordinates["sampleID"] = sample

            coordinates.index = [f"{i}_{sample}" for i in adata.obs_names]
            coordinates_df.append(coordinates)

        self.coordinates_df = pd.concat(coordinates_df)

        data = load_data(samples, self.out_folder, load_image_features=False)
        self.transcriptomics_df = pd.DataFrame(data["y"][:, self.genes_to_keep],
                                               index=data["barcode"])

        self.barcode_sample_idx = self.coordinates_df.index.values
        assert (self.transcriptomics_df.index == self.coordinates_df.index).all()

        if self.is_train:
            idx = self.coordinates_df.groupby("sampleID").sample(self.sample_n).index
            self.coordinates_df = self.coordinates_df.loc[idx]
            self.transcriptomics_df = self.transcriptomics_df.loc[idx]
            self.barcode_sample_idx = idx.values

        self.image_feature_source = f"{self.out_folder}/data/tiles"

        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.patch_size),
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=180),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.patch_size),
            ])

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample_id = self.samples[idx]

        if sample_id not in self.cache:

            spots = self.coordinates_df.query(f"sampleID == '{sample_id}'")

            patches, coordinates, expression = [], [], []
            for _, spot_info in spots.iterrows():

                patch = np.load(f"{self.image_feature_source}/{spot_info.barcode}_{sample_id}.npy")
                patch = self.transforms(patch)
                patches.append(patch)

                coord = np.array([spot_info.x_array, spot_info.y_array])
                coordinates.append(coord)

                y = self.transcriptomics_df.loc[spot_info.name].values
                y = y.astype(np.float32)

                expression.append(y)

            patches = np.array(patches)
            coordinates = np.array(coordinates)
            expression = np.array(expression)

            if self.is_train:
                self.cache[sample_id] = [patches, coordinates, expression]

        else:
            patches, coordinates, expression = self.cache[sample_id]

        return patches, coordinates, expression
