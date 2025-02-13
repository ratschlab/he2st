from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import pyvips
import torch

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


def extract_morphology_embeddings(
        img_path,
        model,
        x_pixel,
        y_pixel,
        spot_diameter,
        device,
        preprocess,
        feature_dim,
        n_components=15,
        apply_pca=False):

    model.to(device)
    image = pyvips.Image.new_from_file(img_path)

    n_spots = len(x_pixel)
    morphology_representation = np.zeros([n_spots, feature_dim])
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            for i, (x, y) in tqdm(enumerate(zip(x_pixel, y_pixel)), total=n_spots):

                x = x - int(spot_diameter // 2)
                y = y - int(spot_diameter // 2)

                spot = image.crop(x, y, spot_diameter, spot_diameter)
                spot = np.ndarray(buffer=spot.write_to_memory(),
                                  dtype=format_to_dtype[spot.format],
                                  shape=[spot.height, spot.width, spot.bands])

                input_tensor = preprocess(spot)
                input_batch = input_tensor.unsqueeze(0)
                input_batch = input_batch.to(device)
                output = model(input_batch)

                output = output.detach().cpu().numpy().squeeze()

                morphology_representation[i, :,] = output

    if apply_pca:
        pca = PCA(n_components=n_components)
        pca.fit(morphology_representation)
        morphology_representation = pca.transform(morphology_representation)

    return morphology_representation
