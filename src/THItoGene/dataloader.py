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


import torch
import numpy as np
from scipy.spatial import distance_matrix, minkowski_distance, distance


def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    spatialMatrix = coord  # .cpu().numpy()
    nodes = spatialMatrix.shape[0]
    Adj = torch.zeros((nodes, nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp = spatialMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0] - 1
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)  # optional
        for j in np.arange(1, k + 1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'STD':
                if distMat[0, res[0][j]] <= boundary:
                    Adj[i][res[0][j]] = 1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':
                if distMat[0, res[0][j]] <= 2.0:
                    Adj[i][res[0][j]] = 1.0
    return Adj


class THItoGeneCustomDataLoader(Dataset):
    def __init__(self,
                 out_folder,
                 samples,
                 genes_to_keep,
                 is_train,
                 patch_size=112,
                 sample_n=256,
                 target_sum=10000):
        super().__init__()
        self.out_folder = out_folder
        self.samples = list(samples)
        self.genes_to_keep = genes_to_keep
        self.target_sum = target_sum
        self.is_train = is_train
        self.cache = {}
        self.patch_size = patch_size
        self.sample_n = sample_n
        coordinates_df = []
        for sample in tqdm(samples):

            json_info = json.load(open(f"{out_folder}/data/meta/{sample}.json"))
            sample_patch_size = int(json_info["spot_diameter_fullres"]) + 1

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

        data_raw = load_data(samples, self.out_folder, load_image_features=False, raw_counts=True)

        self.barcode_sample_idx = self.coordinates_df.index.values

        if self.is_train:
            idx = self.coordinates_df.groupby("sampleID").sample(self.sample_n).index
            self.coordinates_df = self.coordinates_df.loc[idx]
            self.transcriptomics_df = self.transcriptomics_df.loc[idx]
            self.barcode_sample_idx = idx.values

        assert (self.transcriptomics_df.index == self.coordinates_df.index).all()

        self.image_feature_source = f"{self.out_folder}/data/tiles"

        self.calculate_adjacency()

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

    def split_samples(self):

        new_samples = []
        for sample in tqdm(self.samples):
            tab = self.coordinates_df.query(f"sampleID == '{sample}'")

            split_n = ((tab.x_array.max() - tab.x_array.min()) // self.split_samples_n) + 1

            for i in range(self.split_samples_n):
                min_x = tab.x_array.min() + i * split_n
                max_x = min_x + split_n
                new_sample_id_suffix = f"+{i}"

                # Add new sample with suffix
                new_samples.append(f"{sample}{new_sample_id_suffix}")

                # Filter DataFrame rows within the range [min_x, max_x)
                t = tab[(tab.x_array >= min_x) & (tab.x_array < max_x)]
                idx = t.index

                # Update sampleID column for filtered rows
                self.coordinates_df.loc[idx, "sampleID"] = t["sampleID"] + new_sample_id_suffix

                # Update indices with new sample suffix for all relevant DataFrames
                for df in [self.coordinates_df, self.transcriptomics_df]:
                    df.index = [f"{x}{new_sample_id_suffix}" if x in idx else x for x in df.index]

        self.samples = new_samples

    def calculate_adjacency(self):

        self.adjacency = {}

        for sample_id in self.samples:
            coordinates = self.coordinates_df.query(f"sampleID == '{sample_id}'")[["x_array", "y_array"]].values
            self.adjacency[sample_id] = calcADJ(coordinates)

    def __getitem__(self, idx):

        sample_id = self.samples[idx]
        if sample_id not in self.cache:

            spots = self.coordinates_df.query(f"sampleID == '{sample_id}'")
            patches, expression, array_coordinates = [], [], []
            for _, spot_info in spots.iterrows():

                patch = np.load(f"{self.image_feature_source}/{spot_info.barcode}_{sample_id}.npy")
                patch = self.transforms(patch)
                patches.append(patch)

                array_coordinates.append(np.array([spot_info.x_array, spot_info.y_array]))

                y = self.transcriptomics_df.loc[spot_info.name].values
                y = y.astype(np.float32)

                expression.append(y)

            patches = np.array(patches)
            array_coordinates = np.array(array_coordinates)

            expression = np.array(expression)
            if self.is_train:
                self.cache[sample_id] = [patches, array_coordinates, expression]
        else:
            patches, array_coordinates, expression = self.cache[sample_id]

        return patches, array_coordinates, expression, self.adjacency[sample_id]
