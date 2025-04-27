import rasterio
import cv2
from const import IMAGES_TO_REMOVE
from utils import per_band_minmax
from tqdm import tqdm
import numpy as np
import pickle
import os


def load_image_and_mask(image_path, mask_path, target_size=(256, 256)):
    with rasterio.open(image_path) as img_src:
        image = img_src.read()  # (4, H, W)

    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read()  # (1, H, W)

    # Transpose image to (H, W, 4)
    image = np.transpose(image, (1, 2, 0))  # (H, W, 4)
    mask = mask[0, :, :]  # (H, W)

    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)  # For images
    mask = cv2.resize(
        mask, target_size, interpolation=cv2.INTER_NEAREST
    )  # For masks (preserve class labels)

    return image, mask


def load_full_dataset(image_dir, mask_dir, limit=None):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    filtered_pairs = [
        (img_name, mask_name)
        for img_name, mask_name in zip(image_files, mask_files)
        if img_name not in IMAGES_TO_REMOVE
    ]

    # Apply limit after filtering
    if limit:
        filtered_pairs = filtered_pairs[:limit]

    X = []
    Y = []

    for img_name, mask_name in tqdm(filtered_pairs, total=len(filtered_pairs)):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        image, mask = load_image_and_mask(img_path, mask_path)
        image = per_band_minmax(image)

        X.append(image)
        Y.append(mask.astype(np.uint8))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def load_model(model_path="random_forest_model.pkl"):
    """
    Load a pickled Random Forest model or Pipeline.

    Args:
        model_path (str): Path to the pickled model file (default: 'random_forest_model.pkl').

    Returns:
        The loaded model (e.g., RandomForestClassifier or Pipeline).
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        raise
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
