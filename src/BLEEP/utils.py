import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


def find_matches(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    return indices.cpu().numpy()


def _run_inference_from_dataloader(model, dataloader):
    device = torch.device('cpu')
    model.to(device)  # same device
    model.eval()

    image_embeddings = []
    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_out, spot_out = model(batch)

            image_out = image_out.cpu().detach().numpy()
            spot_out = spot_out.cpu().detach().numpy()

            image_embeddings.extend(image_out)
            spot_embeddings.extend(spot_out)

    return np.array(image_embeddings), np.array(spot_embeddings)


def run_inference_from_dataloader(
        model,
        dataloader_key,
        expression_key,
        dataloader_query=None,
        image_embeddings_query=None,
        top_k=50):
    image_embeddings_key, spot_embeddings_key = _run_inference_from_dataloader(model, dataloader_key)

    if image_embeddings_query is None:
        image_embeddings_query, _ = _run_inference_from_dataloader(model, dataloader_query)

    idx_match = find_matches(spot_embeddings_key, image_embeddings_query, top_k=top_k)

    matched_spot_embeddings_pred = np.zeros((idx_match.shape[0], spot_embeddings_key.shape[1]))
    matched_spot_expression_pred = np.zeros((idx_match.shape[0], expression_key.shape[1]))
    for i in range(idx_match.shape[0]):
        # matched_spot_embeddings_pred[i,:] = np.average(spot_embeddings_key[idx_match[i,:],:], axis=0)
        matched_spot_expression_pred[i, :] = np.average(expression_key[idx_match[i, :], :], axis=0)

    return matched_spot_expression_pred
