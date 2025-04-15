import torch
import lightning as L
from deepspot.cell import DeepCell
from deepspot.cell import DeepCellDataLoader
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from aestetik import AESTETIK
import pandas as pd
import numpy as np
import yaml
import sys
sys.path.append('../')
from src.morphology_model import get_morphology_model_and_preprocess
from deepspot.utils.utils import run_inference_from_dataloader
from deepspot.utils.utils import plot_loss_values

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

genes = pd.read_csv(f"{out_folder}/info_highly_variable_genes.csv")
selected_genes_bool = genes.isPredicted.values
genes_to_predict = genes[selected_genes_bool]


num_workers = torch.get_num_threads()

factor_log1p = param.get('factor_log1p', 100)
resample_samples = param.get('resample_samples', False)
smooth_n = param.get('smooth_n', 0)
gene_norm = param.get('gene_norm', None)
cell_context = param.get('cell_context', 'cell_neighbors')


n_neighbors = int(param["n_neighbors"])
batch_size = int(param["batch_size"])
image_feature_model = param["image_feature_model"]
max_epochs = int(param["epochs"])
resolution = int(param["res"])
cell_diameter = param["cell_diameter"]


if "resample_samples" in param:
    del param['resample_samples']
if "smooth_n" in param:
    del param['smooth_n']
if "gene_norm" in param:
    del param['gene_norm']

del param["n_neighbors"]
del param["res"]
del param["image_feature_model"]
del param["batch_size"]
del param["epochs"]
del param["cell_diameter"]


device = torch.device("cuda")
_, _, feature_dim = get_morphology_model_and_preprocess(model_name=image_feature_model,
                                                        device=device)

train_data_loader_custom = DeepCellDataLoader(
                               out_folder=out_folder,
                               cell_diameter=cell_diameter,
                               samples=training_samples_set, 
                               genes_to_keep=selected_genes_bool,
                               morphology_model_name=image_feature_model,
                               cell_context=cell_context,
                               resolution=resolution,
                               n_neighbors=n_neighbors,
                               smooth_n=smooth_n,
                               load_image_features_in_memory=True,
                               normalize=gene_norm)

train_data_loader = torch.utils.data.DataLoader(dataset=train_data_loader_custom,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                persistent_workers=True,
                                                pin_memory=True,
                                                shuffle=True)


regressor = DeepCell(input_size=feature_dim,
                     output_size=selected_genes_bool.sum(),
                     scaler=train_data_loader_custom.scaler,
                     **param)

regressor.to(device)
trainer = L.Trainer(max_epochs=max_epochs, logger=False, enable_checkpointing=False,
                    callbacks=[EarlyStopping(monitor="train_step",
                                             patience=3,
                                             min_delta=0.01,
                                             mode="min")])
trainer.fit(regressor, train_data_loader)

if len(sys.argv) >= 4:

    validation_data_loader_custom = DeepCellDataLoader(out_folder=out_folder,
                                                       cell_diameter=cell_diameter,
                                                       samples=validation_samples_set, 
                                                       genes_to_keep=selected_genes_bool,
                                                       morphology_model_name=image_feature_model, 
                                                       cell_context=cell_context,
                                                       load_image_features_in_memory=True,
                                                       n_neighbors=n_neighbors)

    validation_data_loader = torch.utils.data.DataLoader(dataset=validation_data_loader_custom,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         persistent_workers=True,
                                                         pin_memory=True,
                                                         shuffle=False)

    validation_prediction = run_inference_from_dataloader(regressor, validation_data_loader, device)
    validation_prediction = regressor.inverse_transform(validation_prediction)
    validation_prediction = pd.DataFrame(validation_prediction,
                                         index=validation_data_loader_custom.coordinates_df.index,
                                         columns=genes.gene_name.values[selected_genes_bool])

    test_data_loader_custom = DeepCellDataLoader(
                                       out_folder=out_folder,
                                       cell_diameter=cell_diameter,
                                       samples=test_samples_set, 
                                       genes_to_keep=selected_genes_bool,
                                       morphology_model_name=image_feature_model, 
                                       cell_context=cell_context,
                                       load_image_features_in_memory=True,
                                       n_neighbors=n_neighbors)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_data_loader_custom,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         persistent_workers=True,
                                                         pin_memory=True,
                                                         shuffle=False)

    test_prediction = run_inference_from_dataloader(regressor, test_data_loader, device)
    test_prediction = regressor.inverse_transform(test_prediction)
    test_prediction = pd.DataFrame(test_prediction,
                                   index=test_data_loader_custom.coordinates_df.index,
                                   columns=genes.gene_name.values[selected_genes_bool])

    validation_prediction.to_pickle(
        f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/prediction/{param_file}_validation.pkl")

    test_prediction.to_pickle(
        f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/prediction/{param_file}_test.pkl")

    model_path = f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/checkpoint/{param_file}.pkl"
    plot_path = f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/statistics/{param_file}.png"

else:

    model_path = f"{out_folder}/evaluation/{model}/final_model.pkl"
    plot_path = f"{out_folder}/evaluation/{model}/final_model.png"


torch.save(regressor, model_path)

plot_loss_values(regressor.training_loss)
plt.savefig(plot_path, dpi=300)
