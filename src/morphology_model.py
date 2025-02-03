from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from huggingface_hub import login, hf_hub_download
from transformers import AutoImageProcessor, ViTModel, AutoModel
from collections import OrderedDict
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import json
import timm
import glob
import sys
import os

from src.config import UNI_PATH_WEIGHTS
from src.config import HOPTIMUS0_PATH_WEIGHTS
from src.config import PHIKON_PATH_WEIGHTS
from src.config import PHIKONV2_PATH_WEIGHTS


def get_morphology_model_and_preprocess(model_name, device):

    if model_name == "uni":
        morphology_model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True)
        morphology_model.load_state_dict(torch.load(UNI_PATH_WEIGHTS, map_location=device), strict=True)
        morphology_model.eval()

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        feature_dim = 1024
    elif model_name == "hoptimus0":

        morphology_model = timm.create_model(
            "vit_giant_patch14_reg4_dinov2",
            img_size=224,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )
        morphology_model.load_state_dict(torch.load(HOPTIMUS0_PATH_WEIGHTS, map_location=device), strict=True)
        morphology_model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])

        feature_dim = 1536

    elif model_name == "phikon":

        # load phikon
        image_processor = AutoImageProcessor.from_pretrained(PHIKON_PATH_WEIGHTS)
        model = ViTModel.from_pretrained(PHIKON_PATH_WEIGHTS, add_pooling_layer=False)

        def image_processor_edit(x):
            x = image_processor(x, return_tensors="pt")["pixel_values"].squeeze()
            return x

        class MyModel(torch.nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                with torch.no_grad():
                    x = self.model(x)
                    x = x.last_hidden_state[:, 0, :]
                    return x

        preprocess = image_processor_edit
        morphology_model = MyModel(model)
        feature_dim = 768

    elif model_name == "phikonv2":

        # load phikon
        image_processor = AutoImageProcessor.from_pretrained(PHIKONV2_PATH_WEIGHTS)
        model = AutoModel.from_pretrained(PHIKONV2_PATH_WEIGHTS)

        def image_processor_edit(x):
            x = image_processor(x, return_tensors="pt")["pixel_values"].squeeze()
            return x

        class MyModel(torch.nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                with torch.no_grad():
                    x = self.model(x)
                    x = x.last_hidden_state[:, 0, :]
                    return x

        preprocess = image_processor_edit
        morphology_model = MyModel(model)
        feature_dim = 1024

    elif model_name == "inception":

        weights = Inception_V3_Weights.DEFAULT
        morphology_model = inception_v3(weights=weights)
        morphology_model.fc = torch.nn.Identity()

        morphology_model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            weights.transforms(antialias=True),
        ])

        feature_dim = 2048

    elif model_name == "resnet50":

        weights = ResNet50_Weights.DEFAULT
        morphology_model = resnet50(weights=weights)
        morphology_model.fc = torch.nn.Identity()

        morphology_model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            weights.transforms(antialias=True),
        ])

        feature_dim = 2048

    elif model_name == "densenet121":

        weights = DenseNet121_Weights.DEFAULT
        morphology_model = densenet121(weights=weights)
        morphology_model.classifier = torch.nn.Identity()

        morphology_model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            weights.transforms(antialias=True),
        ])

        feature_dim = 1024

    return morphology_model, preprocess, feature_dim
