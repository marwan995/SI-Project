import rasterio
import numpy as np
import matplotlib.pyplot as plt
from const import IMAGES_PATH, MASKS_PATH


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


def per_band_minmax(img):
    """
    Normalize each band of the image independently to the range [0, 1].

    Parameters
    ----------
    img : ndarray
        Input image as a (H, W, C) NumPy array.

    Returns
    -------
    img : ndarray
        Normalized image with the same shape as the input.
    """
    img = img.astype(np.float32)
    for b in range(img.shape[2]):
        band = img[:, :, b]
        img[:, :, b] = (band - band.min()) / (band.max() - band.min() + 1e-6)
    return img


def plot_image_and_mask(image, mask):
    """
    Plot the RGB channels, infrared channel, and binary cloud mask of a satellite image.

    Parameters
    ----------
    image : ndarray
        Satellite image as a (H, W, 4) NumPy array.
    mask : ndarray
        Cloud mask as a (H, W) NumPy array.
    
    Returns
    -------
    None
    """
    # Ensure mask is binary (1 or 0)
    mask = (mask > 0).astype(int)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image[:, :, :3])  # show RGB
    plt.title("RGB Image")
    plt.subplot(1, 3, 2)
    plt.imshow(image[:, :, 3])  # show infrared
    plt.title("Infrared Image")
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")  # show binary mask
    plt.title("Cloud Mask")
    plt.show()


def read_and_plot(id):
    """
    Generate image and mask file paths based on an identifier.

    Parameters
    ----------
    id : str
        The identifier for the image and mask filenames (without extension).

    Returns
    -------
    image_path : str
        Full path to the satellite image file.
    mask_path : str
        Full path to the corresponding mask file.
    """
    image_path = f"{IMAGES_PATH}/{id}.tif"
    mask_path = f"{MASKS_PATH}/{id}.tif"
    return image_path, mask_path


def get_result_by_filename(results, target_filename):
    """
    Retrieve a result dictionary for a specific filename.

    Parameters
    ----------
    results : list of dict
        A list of result dictionaries, each containing a 'filename' key.
    target_filename : str
        The filename to search for.

    Returns
    -------
    dict or None
        The matching result dictionary if found, otherwise None.
    """
    for result in results:
        if result["filename"] == target_filename:
            return result
    return None  # If not found
