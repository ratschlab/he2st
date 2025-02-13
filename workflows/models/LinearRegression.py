from src.utils import load_data
from sklearn.linear_model import LinearRegression
from pickle import dump
import multiprocessing
import pandas as pd
import numpy as np
import yaml
import sys
sys.path.append('../')


model = str(sys.argv[1])
out_folder = str(sys.argv[2])

if len(sys.argv) >= 4:
    param_file = str(sys.argv[3])
    validation_samples = str(sys.argv[4])
    test_samples = str(sys.argv[5])
    param_file_path = f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/parameters/{param_file}.yaml"

else:
    param_file_path = f"{out_folder}/evaluation/{model}/top_param_overall.yaml"


with open(param_file_path, "r") as stream:
    param = yaml.safe_load(stream)


with open("config_dataset.yaml", "r") as stream:
    config_dataset = yaml.safe_load(stream)

all_samples = set(config_dataset["SAMPLE"])

if len(sys.argv) >= 4:
    validation_samples_set = set(validation_samples.split("_"))
    test_samples_set = set(test_samples.split("_"))
    training_samples_set = all_samples - validation_samples_set - test_samples_set
else:
    training_samples_set = all_samples


if len(sys.argv) >= 4:
    training_data = load_data(training_samples_set, out_folder, param["image_feature_model"])
    validation_data = load_data(validation_samples_set, out_folder, param["image_feature_model"])
    test_data = load_data(test_samples_set, out_folder, param["image_feature_model"])
else:
    training_data = load_data(all_samples, out_folder, param["image_feature_model"])

del param["image_feature_model"]

genes = pd.read_csv(f"{out_folder}/info_highly_variable_genes.csv")
selected_genes_bool = genes.isPredicted.values

regressor = LinearRegression(**param, n_jobs=-1)
regressor.fit(training_data["X"], training_data["y"][:, selected_genes_bool])

if len(sys.argv) >= 4:
    validation_prediction = regressor.predict(validation_data["X"])
    validation_prediction = pd.DataFrame(validation_prediction,
                                         index=validation_data["barcode"],
                                         columns=genes.gene_name.values[selected_genes_bool])

    test_prediction = regressor.predict(test_data["X"])
    test_prediction = pd.DataFrame(test_prediction,
                                   index=test_data["barcode"],
                                   columns=genes.gene_name.values[selected_genes_bool])

    validation_prediction.to_pickle(
        f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/prediction/{param_file}_validation.pkl")

    test_prediction.to_pickle(
        f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/prediction/{param_file}_test.pkl")

    model_path = f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/checkpoint/{param_file}.pkl"

else:

    model_path = f"{out_folder}/evaluation/{model}/final_model.pkl"

with open(model_path, "wb") as f:
    dump(regressor, f, protocol=5)
