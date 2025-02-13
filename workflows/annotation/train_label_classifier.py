from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from pickle import dump
import plotnine as p9
import pandas as pd
import scanpy as sc
import numpy as np
import yaml
import glob
import anndata as ad
import torch
from tqdm import tqdm

import sys

# Convert np.array and masked_array to lists


def convert_to_lists(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, np.ma.MaskedArray):
            d[key] = value.data.tolist()  # Convert masked array's data to list
    return d


model = sys.argv[1]
out_folder = sys.argv[2]

with open("config_dataset.yaml", "r") as stream:
    config_dataset = yaml.safe_load(stream)


source_data_path = config_dataset['source_data_path']
metadata_path = config_dataset['metadata_path']

id_pair = pd.read_csv(metadata_path).id_pair.values.astype(str)
len(id_pair)

model_path = f"{out_folder}/prediction/{model}_label_model.pkl"
yaml_path = f"{out_folder}/prediction/{model}_label_model_cv_scores.yaml"


files = glob.glob(f'../{source_data_path}/out_benchmark/prediction/{model}/data/h5ad/*')

adata = [sc.read_h5ad(f) for f in tqdm(files)]
adata = ad.concat(adata, merge='same')


parameters = {"n_estimators": [100, 200],
              'max_depth': [15, 20, 30, 50]}


clf = RandomForestClassifier(n_jobs=-1)

grid_search_cv = GridSearchCV(clf, parameters,
                              scoring='balanced_accuracy',
                              return_train_score=True,
                              n_jobs=-1, cv=3,
                              )
grid_search_cv.fit(adata.X, adata.obs.ground_truth, groups=adata.obs.ground_truth)


with open(model_path, "wb") as f:
    dump(grid_search_cv.best_estimator_, f, protocol=5)


# Apply the conversion
yaml_dict = convert_to_lists(grid_search_cv.cv_results_)

# Write dictionary to YAML file
with open(yaml_path, 'w') as file:
    yaml.dump(yaml_dict, file, default_flow_style=False)
