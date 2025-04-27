import numpy as np


def evaluate_dice_score(model, images, targets, batch_size=100000):
    """
    Evaluate the Dice similarity coefficient (F1 score) between model predictions and targets.

    Computes the Dice score in a memory-efficient way by processing large inputs in batches.
    The Dice score measures the overlap between predicted and target binary masks, with
    values ranging from 0 (no overlap) to 1 (perfect overlap).

    Parameters
    ----------
    model : object
        A trained model with a predict() method that takes input array of shape (N, C)
        and returns predicted class probabilities or logits of shape (N,)
    images : numpy.ndarray
        Input images array of shape (N, H, W, C) where:
        - N: Number of images
        - H: Height of images
        - W: Width of images
        - C: Number of channels
    targets : numpy.ndarray
        Ground truth binary masks array of shape (N, H, W)
    batch_size : int, optional
        Number of pixels to process at once (default=100000).
        Used to prevent memory issues with large inputs.

    Returns
    -------
    float
        Dice similarity coefficient between predictions and targets.
        Calculated as: (2 * intersection) / (union + epsilon)
    """
    N, H, W, C = images.shape
    total_pixels = N * H * W
    images_reshaped = images.reshape(-1, C)

    preds_flat = np.zeros(total_pixels, dtype=np.uint8)

    for start in range(0, total_pixels, batch_size):
        end = min(start + batch_size, total_pixels)
        preds_flat[start:end] = model.predict(images_reshaped[start:end])

    preds = preds_flat.reshape(N, H, W)

    # Ensure binary masks
    preds = (preds > 0).astype(np.uint8)
    targets = (targets > 0).astype(np.uint8)

    intersection = np.sum(preds * targets)
    union = np.sum(preds) + np.sum(targets)

    dice = (2.0 * intersection) / (union + 1e-6)
    return dice
