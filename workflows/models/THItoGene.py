import matplotlib.pyplot as plt
import torch
import lightning as L
from src.loss_function import loss_function
from src.THItoGene.model import THItoGene
from src.THItoGene.utils import run_inference_from_dataloader
from src.THItoGene.dataloader import THItoGeneCustomDataLoader
from src.morphology_model import get_morphology_model_and_preprocess
from src.utils import plot_loss_values
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from src.utils import load_data
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

genes = pd.read_csv(f"{out_folder}/info_highly_variable_genes.csv")
selected_genes_bool = genes.isPredicted.values
genes_to_predict = genes[selected_genes_bool]


num_workers = torch.get_num_threads()


batch_size = int(param["batch_size"])
max_epochs = int(param["epochs"])


del param["batch_size"]
del param["epochs"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_data_loader_custom = THItoGeneCustomDataLoader(out_folder,
                                                     training_samples_set,
                                                     selected_genes_bool,
                                                     is_train=True)

train_data_loader = torch.utils.data.DataLoader(dataset=train_data_loader_custom,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=True)


regressor = THItoGene(fig_size=train_data_loader_custom.patch_size,
                      n_genes=selected_genes_bool.sum(),
                      **param)

regressor.to(device)
trainer = L.Trainer(max_epochs=max_epochs, logger=False, enable_checkpointing=False)
trainer.fit(regressor, train_data_loader)

if len(sys.argv) >= 4:

    validation_data_loader_custom = THItoGeneCustomDataLoader(out_folder,
                                                              validation_samples_set,
                                                              selected_genes_bool,
                                                              patch_size=train_data_loader_custom.patch_size,
                                                              is_train=False)

    validation_data_loader = torch.utils.data.DataLoader(dataset=validation_data_loader_custom,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         shuffle=False)

    validation_prediction = run_inference_from_dataloader(regressor, validation_data_loader)
    validation_prediction = pd.DataFrame(validation_prediction,
                                         index=validation_data_loader_custom.barcode_sample_idx,
                                         columns=genes.gene_name.values[selected_genes_bool])

    test_data_loader_custom = THItoGeneCustomDataLoader(out_folder,
                                                        test_samples_set,
                                                        selected_genes_bool,
                                                        patch_size=train_data_loader_custom.patch_size,
                                                        is_train=False)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_data_loader_custom,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=False)

    test_prediction = run_inference_from_dataloader(regressor, test_data_loader)

    test_prediction = pd.DataFrame(test_prediction,
                                   index=test_data_loader_custom.barcode_sample_idx,
                                   columns=genes.gene_name.values[selected_genes_bool])

    validation_prediction.to_pickle(
        f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/prediction/{param_file}_validation.pkl")

    test_prediction.to_pickle(
        f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/prediction/{param_file}_test.pkl")

    model_path = f"{out_folder}/evaluation/{test_samples}/{validation_samples}/{model}/checkpoint/{param_file}.pkl"


else:

    model_path = f"{out_folder}/evaluation/{model}/final_model.pkl"


torch.save(regressor, model_path)
