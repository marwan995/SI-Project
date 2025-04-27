import tqdm

def predict_mask(image, model):
    """
    Generate a pixel-wise mask prediction for an input image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array of shape (H, W, C) where:
        - H: Image height
        - W: Image width
        - C: Number of channels
    model : object
        Trained model with a predict() method that takes input array of shape (N, C)
        and returns predicted class probabilities or logits of shape (N,)

    Returns
    -------
    numpy.ndarray
        Predicted mask array of shape (H, W) with raw model outputs
    """
    # Flatten the image to (H*W, 4) for prediction
    h, w, c = image.shape
    flat_pixels = image.reshape(-1, c)

    # Predict class for each pixel
    preds = model.predict(flat_pixels)

    # Reshape prediction back to (H, W)
    mask_pred = preds.reshape(h, w)

    return mask_pred


def get_predicted_masks(images, model):
    """
    Generate binary masks for a collection of images using the provided model.

    Parameters
    ----------
    images : list or numpy.ndarray
        Collection of input images where each image has shape (H, W, C)
    model : object
        Trained model with a predict() method compatible with predict_mask()

    Returns
    -------
    list
        List of predicted binary masks (each of shape (H, W)) where:
        - 0 represents background class
        - 1 represents foreground class
    """
    pred_masks = []

    for img in tqdm(images, desc="Predicting masks"):
        # Predict the mask
        mask_pred = predict_mask(img, model)

        # Binarize the predicted mask if needed
        mask_pred_bin = (mask_pred > 0.5).astype(np.uint8)

        pred_masks.append(mask_pred_bin)

    return pred_masks