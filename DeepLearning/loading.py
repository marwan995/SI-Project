import rasterio
import numpy as np


def load_image_and_mask(image_path, mask_path):
    """
    Load a satellite image and its corresponding mask.

    Parameters
    ----------
    image_path : str
        Path to the satellite image file (GeoTIFF).
    mask_path : str
        Path to the corresponding mask file (GeoTIFF).

    Returns
    -------
    image : ndarray
        The loaded image as a (H, W, 4) NumPy array, where the channels represent RGB and infrared.
    mask : ndarray
        The loaded mask as a (H, W) NumPy array.
    """
    with rasterio.open(image_path) as img_src:
        image = img_src.read()  # (4, H, W)

    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read()  # (1, H, W)

    image = np.transpose(image, (1, 2, 0))  # (H, W, 4)
    mask = mask[0, :, :]  # (H, W)

    return image, mask
