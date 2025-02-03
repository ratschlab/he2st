import torch
from tqdm import tqdm
import numpy as np


def run_inference_from_dataloader(model, dataloader):
    device = torch.device('cpu')
    model.to(device)  # same device
    model.eval()

    out = []

    with torch.no_grad():
        for patch, coord, _ in tqdm(dataloader):

            # patch = patch.to(device)
            # coord = patch.to(device)
            y = model(patch, coord)
            y = y.cpu().detach().numpy()

            out.extend(y)

    return np.concatenate(out).squeeze()
