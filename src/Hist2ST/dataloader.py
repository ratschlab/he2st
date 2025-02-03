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


def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA'):
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


class Hist2STCustomDataLoader(Dataset):
    def __init__(self,
                 out_folder,
                 samples,
                 genes_to_keep,
                 is_train,
                 patch_size=160,
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

        data_raw = load_data(samples, self.out_folder, load_image_features=False, raw_counts=True)
        self.raw_transcriptomics_df = pd.DataFrame(data_raw["y"][:, self.genes_to_keep],
                                                   index=data_raw["barcode"])
        self.barcode_sample_idx = self.coordinates_df.index.values

        if self.is_train:
            idx = self.coordinates_df.groupby("sampleID").sample(self.sample_n).index
            self.coordinates_df = self.coordinates_df.loc[idx]
            self.transcriptomics_df = self.transcriptomics_df.loc[idx]
            self.raw_transcriptomics_df = self.raw_transcriptomics_df.loc[idx]
            self.barcode_sample_idx = idx.values

        assert (self.transcriptomics_df.index == self.coordinates_df.index).all()
        assert (self.raw_transcriptomics_df.index == self.coordinates_df.index).all()

        self.image_feature_source = f"{self.out_folder}/data/tiles"

        self.calculate_adjacency()
        self.calclulate_ori_counts()

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.patch_size),
        ])

    def __len__(self):

        return len(self.samples)

    def calculate_adjacency(self):

        self.adjacency = {}

        for sample_id in self.samples:
            coordinates = self.coordinates_df.query(f"sampleID == '{sample_id}'")[["x_array", "y_array"]].values
            self.adjacency[sample_id] = calcADJ(coordinates)

    def calclulate_ori_counts(self):
        # https://github.com/biomed-AI/Hist2ST/blob/main/dataset.py

        self.ori = {}
        self.counts = {}
        idx = self.raw_transcriptomics_df.index.to_series().apply(lambda x: x.split("_")[1]).values

        for sample in self.samples:
            raw_expression = self.raw_transcriptomics_df.iloc[idx == sample].values

            self.ori[sample] = raw_expression

            n_counts = raw_expression.sum(1)
            sf = n_counts / np.median(n_counts)
            self.counts[sample] = sf

    def __getitem__(self, idx):

        sample_id = self.samples[idx]
        if sample_id not in self.cache:

            spots = self.coordinates_df.query(f"sampleID == '{sample_id}'")
            patches, expression, array_coordinates, pixel_coordinates = [], [], [], []
            for _, spot_info in spots.iterrows():

                patch = np.load(f"{self.image_feature_source}/{spot_info.barcode}_{sample_id}.npy")
                patch = self.transforms(patch)
                patches.append(patch)

                array_coordinates.append(np.array([spot_info.x_array, spot_info.y_array]))
                pixel_coordinates.append(np.array([spot_info.x_pixel, spot_info.y_pixel]))

                y = self.transcriptomics_df.loc[spot_info.name].values
                y = y.astype(np.float32)

                expression.append(y)

            patches = np.array(patches)
            array_coordinates = np.array(array_coordinates)
            pixel_coordinates = np.array(pixel_coordinates)

            expression = np.array(expression)

            if self.is_train:
                self.cache[sample_id] = [patches, array_coordinates, expression, pixel_coordinates]
        else:
            patches, array_coordinates, expression, pixel_coordinates = self.cache[sample_id]

        return patches, array_coordinates, expression, self.adjacency[
            sample_id], self.ori[sample_id], self.counts[sample_id], pixel_coordinates
