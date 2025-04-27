import rasterio
import numpy as np
import matplotlib.pyplot as plt
import torch
from const import IMAGES_PATH,MASKS_PATH

def load_image_and_mask(image_path, mask_path):
    with rasterio.open(image_path) as img_src:
        image = img_src.read()  # (4, H, W)

    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read()  # (1, H, W)

    image = np.transpose(image, (1, 2, 0))  # (H, W, 4)
    mask = mask[0, :, :]                    # (H, W)

    return image, mask

def per_band_minmax(img):
    img = img.astype(np.float32)
    for b in range(img.shape[2]):
        band = img[:, :, b]
        img[:, :, b] = (band - band.min()) / (band.max() - band.min() + 1e-6)
    return img

def plot_image_and_mask(image, mask):
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
    plt.imshow(mask, cmap='gray')  # show binary mask
    plt.title("Cloud Mask")
    plt.show()

def read_and_plot(id):
    image_path = f"{IMAGES_PATH}/{id}.tif"
    mask_path = f"{MASKS_PATH}/{id}.tif"
    return image_path, mask_path

def get_result_by_filename(results, target_filename):
    """Find and return the result dictionary for a specific filename"""
    for result in results:
        if result['filename'] == target_filename:
            return result
    return None  # If not found