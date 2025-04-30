import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from encode_decode import rle_encode
from const import IMAGES_PATH, MASKS_PATH
from tqdm import tqdm


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

def create_rle_dataframe(data_loader):
    """
    Creates a pandas DataFrame from a DataLoader containing image IDs and RLE-encoded masks.
    
    Args:
        data_loader (DataLoader): PyTorch DataLoader yielding (image, mask, img_name) tuples.
    
    Returns:
        pd.DataFrame: DataFrame with columns 'id' (image filename without extension) and
                      'segmentation' (RLE-encoded mask string).
    """
    ids = []
    segmentations = []
    
    # Define resize transform to ensure 512x512 masks
    #resize = T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)
    
    # Wrap data_loader with tqdm for progress bar
    for _, mask, img_name in tqdm(data_loader, desc="Processing masks"):
        # Handle batched data
        for i in range(mask.size(0)):  # Iterate over batch dimension
            # Extract single mask and ensure it's binary
            single_mask = mask[i]  # Shape: (1, H, W)
            #single_mask = resize(single_mask)  # Resize to 512x512
            single_mask = single_mask.squeeze(0)  # Remove channel dimension: (512, 512)
            single_mask = (single_mask > 0.5).float()  # Ensure binary (0s and 1s)
            
            # Convert to NumPy for RLE encoding
            mask_np = single_mask.cpu().numpy().astype(np.uint8)
            
            # Encode mask as RLE
            rle_string = rle_encode(mask_np)
            
            # Extract image ID from filename (remove extension)
            img_id = os.path.splitext(img_name[i])[0]
            
            # Append to lists
            ids.append(img_id)
            segmentations.append(rle_string)
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'segmentation': segmentations
    })
    
    return df

def create_predicted_rle_dataframe(model, data_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Creates a pandas DataFrame with predicted mask RLEs from a model and DataLoader.
    
    Args:
        model (nn.Module): Trained PyTorch model (e.g., UNetPlusPlus) for segmentation.
        data_loader (DataLoader): DataLoader yielding (image, mask, img_name) tuples.
        device (str or torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
    
    Returns:
        pd.DataFrame: DataFrame with columns 'id' (image filename without extension) and
                      'segmentation' (RLE-encoded predicted mask string).
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to the specified device
    
    ids = []
    segmentations = []
    
    # Define resize transform to ensure 512x512 masks
    #resize = T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)
    
    with torch.no_grad():  # Disable gradient computation for inference
        # Wrap data_loader with tqdm for progress bar
        for image, _, img_name in tqdm(data_loader, desc="Generating predictions"):
            # Move images to device
            image = image.to(device)
            
            # Get model predictions
            pred = model(image)  # Shape: (batch_size, 1, H, W) or similar
            pred = torch.sigmoid(pred)  # Convert logits to probabilities
            pred = (pred > 0.5).float()  # Binarize predictions
            
            # Process each prediction in the batch
            for i in range(pred.size(0)):
                # Extract single predicted mask
                single_pred = pred[i]  # Shape: (1, H, W)
                #single_pred = resize(single_pred)  # Resize to 512x512
                single_pred = single_pred.squeeze(0)  # Shape: (512, 512)
                print(single_pred.shape)
                # Convert to NumPy for RLE encoding
                pred_np = single_pred.cpu().numpy().astype(np.uint8)
                
                # Encode predicted mask as RLE
                rle_string = rle_encode(pred_np)
                
                # Extract image ID from filename (remove extension)
                img_id = os.path.splitext(img_name[i])[0]
                
                # Append to lists
                ids.append(img_id)
                segmentations.append(rle_string)
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'segmentation': segmentations
    })
    
    return df