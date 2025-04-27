import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from const import IMAGES_PATH, MASKS_PATH
import pickle


def read_and_plot(id):
    """
    Construct file paths for image and mask files based on the given ID.

    Parameters
    ----------
    id : str or int
        Unique identifier used to construct the file paths.
        Will be converted to string and used in the filename pattern.

    Returns
    -------
    tuple
        A tuple containing two strings:
        - image_path : str
            Full path to the image file with pattern: {IMAGES_PATH}/{id}.tif
        - mask_path : str
            Full path to the mask file with pattern: {MASKS_PATH}/{id}.tif
    """
    image_path = f"{IMAGES_PATH}\\{id}.tif"
    mask_path = f"{MASKS_PATH}\\{id}.tif"
    return image_path, mask_path


def per_band_minmax(img):
    """
    Normalize each band of an image separately using min-max scaling.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array of shape (H, W, C) where:
        - H: Image height
        - W: Image width
        - C: Number of channels/bands

    Returns
    -------
    numpy.ndarray
        Normalized image array of same shape as input (H, W, C), with:
        - Each band independently scaled to [0, 1] range
        - dtype converted to float32
    """
    img = img.astype(np.float32)
    for b in range(img.shape[2]):
        band = img[:, :, b]
        img[:, :, b] = (band - band.min()) / (band.max() - band.min() + 1e-6)
    return img


def plot_image_and_mask(image, mask):
    """
    Display an RGB image, infrared band, and binary mask side by side.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array of shape (H, W, 4) where:
        - First 3 channels (image[:, :, :3]) are RGB bands
        - Last channel (image[:, :, 3]) is infrared band
    mask : numpy.ndarray
        Binary mask array of shape (H, W) where:
        - 0 represents background
        - 1 represents foreground/cloud

    Returns
    -------
    None
        Displays matplotlib figure with three subplots:
        - Left: RGB visualization
        - Middle: Infrared band
        - Right: Binary cloud mask
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


def image_mask_to_df(image, mask):
    """
    Convert an image and mask pair into a pandas DataFrame with spectral features.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array of shape (H, W, 4) containing:
        - Red channel (image[:, :, 0])
        - Green channel (image[:, :, 1])
        - Blue channel (image[:, :, 2])
        - Infrared channel (image[:, :, 3])
    mask : numpy.ndarray
        Binary mask array of shape (H, W) where:
        - 0 represents background
        - 1 represents foreground/class of interest

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - Original spectral bands (r, g, b, infrared)
        - Derived indices (ndsi, cloud_index)
        - Class labels from mask (class)
        Each row represents one pixel from the input
    """
    H, W, C = image.shape
    assert C == 4, "Image must have 4 bands (r, g, b, infrared)"
    assert mask.shape == (H, W), "Mask must match image spatial dimensions"

    # Extract bands
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    nir = image[:, :, 3]

    # Compute NDSI: (Green - NIR) / (Green + NIR)
    ndsi = (green - nir) / (green + nir + 1e-6)

    # Compute Cloud Index: (Blue + Green + Red) / NIR
    cloud_index = (blue + green + red) / (nir + 1e-6)

    # Flatten image and features
    flat_img = image.reshape(-1, 4)
    flat_ndsi = ndsi.flatten()
    flat_cloud_index = cloud_index.flatten()
    flat_mask = mask.flatten()

    # Create DataFrame
    df = pd.DataFrame(
        np.column_stack([flat_img, flat_ndsi[:, None], flat_cloud_index[:, None]]),
        columns=["r", "g", "b", "infrared", "ndsi", "cloud_index"],
    )
    df["class"] = flat_mask.astype(bool)

    return df


def image_mask_to_array(images):
    """
    Convert 4-channel images to 6-channel images by adding spectral indices.

    Parameters
    ----------
    images : numpy.ndarray
        Input image array of shape (N, H, W, 4) containing:
        - N: Number of images
        - H: Image height
        - W: Image width
        - 4 channels: [red, green, blue, nir]

    Returns
    -------
    numpy.ndarray
        Output array of shape (N, H, W, 6) containing:
        - Original 4 bands [red, green, blue, nir]
        - Added NDSI index (channel 4)
        - Added Cloud Index (channel 5)
    """
    # Validate inputs
    assert (
        len(images.shape) == 4 and images.shape[3] == 4
    ), "Images must have shape (N, H, W, 4)"

    N, H, W, _ = images.shape
    # Initialize output array for 6-channel images
    output_images = np.zeros((N, H, W, 6), dtype=images.dtype)

    # Process each image
    for i in range(N):
        image = images[i]  # Shape: (H, W, 4)

        # Extract bands
        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]
        nir = image[:, :, 3]

        # Compute NDSI: (Green - NIR) / (Green + NIR)
        ndsi = (green - nir) / (green + nir + 1e-6)

        # Compute Cloud Index: (Blue + Green + Red) / NIR
        cloud_index = (blue + green + red) / (nir + 1e-6)

        # Create 6-channel image by stacking original channels + NDSI + Cloud Index
        output_images[i] = np.stack([red, green, blue, nir, ndsi, cloud_index], axis=-1)

    return output_images


def build_training_dataframe(x_data, y_data):
    """
    Combine multiple image-mask pairs into a single training DataFrame.

    Parameters
    ----------
    x_data : numpy.ndarray
        Array of input images with shape (N, H, W, C) where:
        - N: Number of images
        - H: Image height
        - W: Image width
        - C: Number of channels
    y_data : numpy.ndarray
        Array of corresponding masks with shape (N, H, W)

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame containing pixel-wise features and labels from:
        - All input images (converted via image_mask_to_df)
        - All corresponding masks
        Shows progress bar during processing via tqdm
    """
    df_list = []

    for i in tqdm(range(len(x_data)), total=len(x_data), desc="Building DataFrame"):
        df = image_mask_to_df(x_data[i], y_data[i])
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    return full_df


def plot_image_mask_pred(model, image, mask, sample_idx=0):
    """
    Visualize an input image, ground truth mask, and model prediction.

    Parameters
    ----------
    model : object
        Trained model with predict() method that takes (N, C) input
    image : numpy.ndarray
        Input image array of shape (N, H, W, C) where C>=3
    mask : numpy.ndarray
        Ground truth mask array of shape (N, H, W)
    sample_idx : int, optional
        Index of sample to visualize (default: 0)

    Returns
    -------
    None
        Displays matplotlib figure with three subplots:
        - Left: RGB visualization of input image
        - Middle: Ground truth binary mask
        - Right: Model's predicted binary mask
    """
    # Validate inputs
    assert (
        image.ndim == 4 and image.shape[3] >= 3
    ), "Image must have shape (N, H, W, C) with C>=3"
    assert (
        mask.ndim == 3 and mask.shape[0] == image.shape[0]
    ), "Mask must have shape (N, H, W)"
    assert (
        sample_idx < image.shape[0]
    ), f"Sample index {sample_idx} out of range for {image.shape[0]} samples"

    # Extract the sample
    img = image[sample_idx]  # Shape: (H, W, C)
    tgt = mask[sample_idx]  # Shape: (H, W)

    # Predict mask using the model
    height, width, channels = img.shape
    img_flat = img.reshape(-1, channels)  # Shape: (H*W, C)
    pred_flat = model.predict(img_flat)  # Shape: (H*W,)
    prd = pred_flat.reshape(height, width)  # Shape: (H, W)

    # Ensure binary masks
    prd = (prd > 0).astype(np.uint8)  # Convert predictions to binary
    tgt = (tgt > 0).astype(np.uint8)  # Ensure target is binary

    # Ensure image is in [0, 1] for RGB visualization
    rgb_img = img[:, :, :3]  # Take first 3 channels (R, G, B)
    if rgb_img.max() > 1.0 or rgb_img.min() < 0.0:
        # Normalize to [0, 1] if not already
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot RGB image
    axes[0].imshow(rgb_img)
    axes[0].set_title("Input Image (RGB)")
    axes[0].axis("off")

    # Plot ground truth mask
    axes[1].imshow(tgt, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Plot predicted mask
    axes[2].imshow(prd, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    print(np.unique(tgt))


def image_mask_to_df_dask(image, mask):
    """
    Convert image-mask pair to a Dask DataFrame using image_mask_to_df.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array of shape (H, W, C)
    mask : numpy.ndarray
        Binary mask array of shape (H, W)

    Returns
    -------
    dask.dataframe.DataFrame
        Lazy-evaluated DataFrame containing pixel-wise features and labels
    """
    # Convert the image and mask to a DataFrame like before
    df = image_mask_to_df(image, mask)
    return df


@delayed
def process_image_pair(image, mask):
    """
    Process image-mask pairs lazily with Dask's delayed decorator.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array of shape (H, W, C)
    mask : numpy.ndarray
        Binary mask array of shape (H, W)

    Returns
    -------
    dask.Delayed
        Delayed object representing the future computation of
        converting the image-mask pair to a DataFrame
    """
    return image_mask_to_df_dask(image, mask)


def build_training_dataframe_dask(x_data, y_data):
    """
    Build training DataFrame using parallel processing with Dask.

    Parameters
    ----------
    x_data : numpy.ndarray
        Array of input images with shape (N, H, W, C)
    y_data : numpy.ndarray
        Array of corresponding masks with shape (N, H, W)

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame containing pixel-wise features from all images,
        computed in parallel using Dask's delayed execution
    """
    # Use Dask to process images in parallel
    delayed_dfs = [process_image_pair(x_data[i], y_data[i]) for i in range(len(x_data))]

    # Convert list of delayed objects into a Dask DataFrame
    dask_df = dd.from_delayed(delayed_dfs)

    # Compute the final DataFrame (this triggers the actual computation)
    full_df = dask_df.compute()

    return full_df


def plot_prediction(mask_pred, image, true_mask=None):
    """
    Visualize prediction results with optional ground truth comparison.

    Parameters
    ----------
    mask_pred : numpy.ndarray
        Predicted mask array of shape (H, W)
    image : numpy.ndarray
        Original input image array of shape (H, W, C)
    true_mask : numpy.ndarray, optional
        Ground truth mask array of shape (H, W) or (H, W, 1)

    Returns
    -------
    None
        Displays matplotlib figure with:
        - Original RGB image
        - Predicted mask
        - Ground truth mask (if provided)
    """
    # Prepare RGB image for display
    rgb_image = image[:, :, :3]
    rgb_norm = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    # Determine subplot count
    num_subplots = 3 if true_mask is not None else 2

    plt.figure(figsize=(5 * num_subplots, 5))

    # Original Image
    plt.subplot(1, num_subplots, 1)
    plt.title("Original Image")
    plt.imshow(rgb_norm)
    plt.axis("off")

    # Predicted Mask
    plt.subplot(1, num_subplots, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask_pred, cmap="gray")
    plt.axis("off")

    # True Mask (if provided)
    if true_mask is not None:
        # Remove singleton dim if needed
        if true_mask.ndim == 3 and true_mask.shape[2] == 1:
            true_mask = true_mask[:, :, 0]
        plt.subplot(1, num_subplots, 3)
        plt.title("Ground Truth Mask")
        plt.imshow(true_mask, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_model_and_print_params(model, save_path="random_forest_model.pkl"):
    """
    Save trained model to disk and print its parameters.

    Parameters
    ----------
    model : object
        Trained scikit-learn compatible model
    save_path : str, optional
        File path to save the model (default: "random_forest_model.pkl")

    Returns
    -------
    None
        Saves model to disk and prints its parameters to stdout
    """
    # Save the model using pickle
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")

    # Print model parameters
    print("\nModel Parameters:")
    # If model is a pipeline, iterate through steps
    for step_name, step_model in model.steps:
        print(f"\nParameters for {step_name}:")
        params = step_model.get_params()
        for param, value in params.items():
            print(f"  {param}: {value}")
