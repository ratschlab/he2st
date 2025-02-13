import numpy as np
import pyvips
import torch
from tqdm import tqdm

from src.Hist2ST.dataloader import calcADJ
from src.BLEEP.utils import run_inference_from_dataloader
from src.BLEEP.dataloader import BLEEPCustomDataLoader
from src.BLEEP.utils import run_inference_from_dataloader

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


def get_low_res_image(image_path, downsample_factor):
    image = pyvips.Image.new_from_file(image_path, access='sequential')
    image_low_res = image.resize(1 / downsample_factor)
    image_low_res_arr = np.ndarray(buffer=image_low_res.write_to_memory(),
                                   dtype=format_to_dtype[image_low_res.format],
                                   shape=[image_low_res.height, image_low_res.width, image_low_res.bands])
    return image_low_res_arr


def crop_tile(image, x_pixel, y_pixel, spot_diameter):
    x = x_pixel - int(spot_diameter // 2)
    y = y_pixel - int(spot_diameter // 2)
    spot = image.crop(y, x, spot_diameter, spot_diameter)
    main_tile = np.ndarray(buffer=spot.write_to_memory(),
                           dtype=format_to_dtype[spot.format],
                           shape=[spot.height, spot.width, spot.bands])
    main_tile = main_tile[:, :, :3]
    return main_tile


def compute_mini_tiles(image, n_tiles, super_resolution_mode=False):
    D = image.shape[0]  # assuming the image is a square, so width = height = D
    n = int(np.sqrt(n_tiles))  # number of squares along one dimension
    square_size = D // n
    D, n, square_size

    # List to hold the split images
    squares = []

    if not super_resolution_mode:
        # Loop to crop the image into n x n squares
        for i in range(n):
            for j in range(n):
                left = j * square_size
                right = left + square_size

                lower = i * square_size
                upper = lower + square_size

                # Crop the image
                crop = image[lower:upper, left:right, ]
                squares.append(crop)
    else:

        left = (square_size // n)
        right = left + square_size

        lower = (square_size // n)
        upper = lower + square_size

        crop = image[lower:upper, left:right, ]
        squares.append(crop)

    return squares

# Helper function to preprocess and convert to the correct dtype


def convert_dtype(data, dtype):
    return data.to(dtype)


def sklearn_predict_spatial_transcriptomics_from_image_path(image_path,
                                                            adata,
                                                            spot_diameter,
                                                            preprocess,
                                                            morphology_model,
                                                            model_expression,
                                                            device
                                                            ):
    image = pyvips.Image.new_from_file(image_path)
    counts = []
    use_bfloat16 = next(morphology_model.parameters()).dtype == torch.bfloat16
    # Set dtype based on the model's precision
    dtype = torch.bfloat16 if use_bfloat16 else torch.float
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            for _, spot in tqdm(adata.obs.iterrows(), total=len(adata.obs)):

                X = crop_tile(image, spot.x_pixel, spot.y_pixel, spot_diameter)
                X = preprocess(X).to(device)
                X = convert_dtype(X, dtype)
                X = morphology_model(X[None, ])

                X = X.detach().cpu().float().numpy()
                expr = model_expression.predict(X)
                counts.append(expr)
    counts = np.array(counts).squeeze()
    return counts


def HisToGene_predict_spatial_transcriptomics_from_image_path(image_path,
                                                              adata,
                                                              spot_diameter,
                                                              preprocess,
                                                              model_expression,
                                                              device
                                                              ):
    image = pyvips.Image.new_from_file(image_path)
    patches = []
    for _, spot in tqdm(adata.obs.iterrows(), total=len(adata.obs)):

        X = crop_tile(image, spot.x_pixel, spot.y_pixel, spot_diameter)
        X = preprocess(X)
        patches.append(X)

    patches = torch.tensor(np.array(patches)).to(device).float()
    coordinates = torch.tensor(adata.obs[["x_array", "y_array"]].values).to(device)
    patches = patches[None,]
    coordinates = coordinates[None,]

    counts = model_expression(patches, coordinates)
    counts = counts.detach().cpu().numpy()
    if counts.ndim > 2:
        counts = counts.squeeze()
    return counts


def Hist2ST_predict_spatial_transcriptomics_from_image_path(image_path,
                                                            adata,
                                                            spot_diameter,
                                                            preprocess,
                                                            model_expression,
                                                            device
                                                            ):
    image = pyvips.Image.new_from_file(image_path)
    patches = []
    for _, spot in tqdm(adata.obs.iterrows(), total=len(adata.obs)):

        X = crop_tile(image, spot.x_pixel, spot.y_pixel, spot_diameter)
        X = preprocess(X)
        patches.append(X)

    coordinates = adata.obs[["x_array", "y_array"]].values
    adjecency = calcADJ(coordinates)

    patches = torch.tensor(np.array(patches)).to(device).float()
    coordinates = torch.tensor(coordinates).to(device)
    adjecency = torch.tensor(adjecency).to(device)
    patches = patches[None,]
    coordinates = coordinates[None,]
    # adjecency = adjecency[None, ]
    counts, _, _ = model_expression(patches, coordinates, adjecency)
    counts = counts.detach().cpu().numpy()
    if counts.ndim > 2:
        counts = counts.squeeze()
    return counts


def THItoGene_predict_spatial_transcriptomics_from_image_path(image_path,
                                                              adata,
                                                              spot_diameter,
                                                              preprocess,
                                                              model_expression,
                                                              device
                                                              ):

    image = pyvips.Image.new_from_file(image_path)
    patches = []
    for _, spot in tqdm(adata.obs.iterrows(), total=len(adata.obs)):

        X = crop_tile(image, spot.x_pixel, spot.y_pixel, spot_diameter)
        X = preprocess(X)
        patches.append(X)

    coordinates = adata.obs[["x_array", "y_array"]].values
    adjecency = calcADJ(coordinates)

    patches = torch.tensor(np.array(patches)).to(device).float()
    coordinates = torch.tensor(coordinates).to(device)
    adjecency = torch.tensor(adjecency).to(device)
    patches = patches[None,]
    coordinates = coordinates[None,]
    # adjecency = adjecency[None, ]
    counts = model_expression(patches, coordinates, adjecency)
    counts = counts.detach().cpu().numpy()

    if counts.ndim > 2:
        counts = counts.squeeze()
    return counts


def BLEEP_predict_spatial_transcriptomics_from_image_path(image_path,
                                                          adata,
                                                          spot_diameter,
                                                          preprocess,
                                                          out_folder,
                                                          selected_genes_bool,
                                                          training_samples,
                                                          morphology_model,
                                                          image_feature_model,
                                                          model_expression,
                                                          top_k,
                                                          device,
                                                          ):

    # COMPUTE QUERY

    image = pyvips.Image.new_from_file(image_path)
    counts = []
    use_bfloat16 = next(morphology_model.parameters()).dtype == torch.bfloat16
    # Set dtype based on the model's precision
    dtype = torch.bfloat16 if use_bfloat16 else torch.float
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            for _, spot in tqdm(adata.obs.iterrows(), total=len(adata.obs)):

                X = crop_tile(image, spot.x_pixel, spot.y_pixel, spot_diameter)
                X = preprocess(X).to(device)
                X = convert_dtype(X, dtype)
                X = morphology_model(X[None, ])

                X = X[None, ].detach()
                item = {"image_features": X}
                expr = model_expression(item)
                expr = expr.detach().float().cpu().numpy()
                counts.append(expr)

    image_embeddings_query = np.array(counts).squeeze()

    # MATCH QUERY

    train_data_loader_custom = BLEEPCustomDataLoader(
        out_folder=out_folder,
        samples=training_samples,
        is_train=False,
        morphology_model_name=image_feature_model,
        genes_to_keep=selected_genes_bool)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_data_loader_custom,
                                                    batch_size=256,
                                                    num_workers=1,
                                                    shuffle=False)

    counts = run_inference_from_dataloader(model_expression,
                                           train_data_loader,
                                           train_data_loader_custom.transcriptomics_df.values,
                                           image_embeddings_query=image_embeddings_query,
                                           top_k=top_k)

    return counts
