import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def dice_loss(pred, target, smooth=1e-6):
    """
    Compute the Dice loss between predicted and target tensors.

    The Dice loss is a measure of overlap between two samples, commonly used for
    image segmentation tasks. It ranges from 0 (perfect match) to 1 (no overlap).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted probabilities or logits of shape (N, C, H, W), where
        N is batch size, C is number of classes, H is height, and W is width.
    target : torch.Tensor
        Ground truth target values of shape (N, C, H, W), with binary values.
    smooth : float, optional
        Smoothing factor to avoid division by zero (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Computed Dice loss (scalar tensor).
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()

    return dice_loss


def calc_loss(pred, target, metrics, bce_weight=0.5):
    """
    Calculate combined BCE-Dice loss and update metrics dictionary.

    This function computes a weighted combination of Binary Cross-Entropy (BCE) loss
    and Dice loss, which is commonly used for image segmentation tasks. It also updates
    a metrics dictionary with the current loss values.

    Parameters
    ----------
    pred : torch.Tensor
        Raw prediction logits from the model of shape (N, C, H, W), where
        N is batch size, C is number of classes, H is height, and W is width.
    target : torch.Tensor
        Ground truth target values of shape (N, C, H, W), with binary values.
    metrics : dict
        Dictionary to accumulate loss metrics. Will be updated with:
        - 'bce': Binary Cross-Entropy loss
        - 'dice': Dice loss
        - 'loss': Combined loss
    bce_weight : float, optional
        Weight for BCE loss in the combined loss (default: 0.5).
        Dice weight will be (1 - bce_weight).

    Returns
    -------
    torch.Tensor
        The computed combined loss (scalar tensor).
    """
    # Binary cross-entropy loss
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # Apply sigmoid to pred for Dice and IoU losses
    pred_sigmoid = torch.sigmoid(pred)

    # Compute Dice and IoU losses
    dice = dice_loss(pred_sigmoid, target)

    # Combined loss
    loss = bce * bce_weight + dice * (1 - bce_weight)

    # Update metrics
    metrics["bce"] += bce.item()
    metrics["dice"] += dice.item()
    metrics["loss"] += loss.item()
    return loss


def evaluate_and_collect(model, dataloader, device):
    """
    Evaluate a segmentation model and collect results with IoU metrics.

    Runs inference on the provided dataloader, computes Intersection-over-Union (IoU)
    for each sample, and collects inputs, predictions, labels, IoU scores, and filenames.
    Results are sorted by IoU score (ascending).

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model to evaluate (should output logits).
    dataloader : torch.utils.data.DataLoader
        Dataloader yielding tuples of (inputs, labels, filenames).
        Expected shapes:
        - inputs: (B, C, H, W)
        - labels: (B, H, W) or (B, 1, H, W)
    device : torch.device
        Device to run evaluation on (e.g., 'cuda' or 'cpu').

    Returns
    -------
    list[dict]
        List of result dictionaries sorted by IoU (ascending), each containing:
        - 'input': Original input tensor (C, H, W) as float32 CPU tensor
        - 'label': Ground truth mask (H, W) as uint8 numpy array
        - 'pred': Binary prediction mask (H, W) as uint8 numpy array
        - 'iou': Computed IoU score (float)
        - 'filename': Corresponding filename (str)
    """
    model.eval()
    results = []

    def vectorized_iou(pred, target):
        # Input shapes: (B, 1, H, W) for both tensors
        pred = pred.bool()
        target = target.bool()
        intersection = (pred & target).float().sum(dim=(2, 3))
        union = (pred | target).float().sum(dim=(2, 3))
        return (intersection + 1e-6) / (union + 1e-6)  # (B, 1)

    with torch.no_grad():
        for inputs, labels, filenames in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # Ensure float type
            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5  # (B, 1, H, W)

            # Add channel dimension if missing (B, H, W) -> (B, 1, H, W)
            if labels.ndim == 3:
                labels = labels.unsqueeze(1)

            # Batch-wise IoU calculation
            batch_ious = vectorized_iou(preds, labels).squeeze(1)  # (B,)

            # Process each sample
            for i in range(inputs.size(0)):
                results.append(
                    {
                        "input": inputs[i].cpu().float(),
                        "label": labels[i].cpu().numpy().squeeze().astype(np.uint8),
                        "pred": preds[i].cpu().numpy().squeeze().astype(np.uint8),
                        "iou": batch_ious[i].item(),
                        "filename": filenames[i],
                    }
                )

    return sorted(results, key=lambda x: x["iou"])


def show_worst_predictions(results, k=5):
    """
    Visualize the k worst predictions based on IoU scores.

    Displays a grid of the worst performing predictions (lowest IoU) with input images,
    ground truth masks, predicted masks, and their IoU scores. Also prints unique values
    found in the ground truth and prediction masks for verification.

    Parameters
    ----------
    results : list[dict]
        List of result dictionaries (typically from evaluate_and_collect) containing:
        - 'input': Input tensor (C, H, W)
        - 'label': Ground truth mask (H, W) as numpy array
        - 'pred': Prediction mask (H, W) as numpy array
        - 'iou': IoU score (float)
        - 'filename': Original filename (str)
    k : int, optional
        Number of worst predictions to display (default: 5).

    Returns
    -------
    None
        Displays matplotlib figures and prints diagnostic information.
    """
    for i, result in enumerate(results[:k]):
        input_img = result["input"].permute(1, 2, 0).numpy()
        input_img = (input_img - input_img.min()) / (
            input_img.max() - input_img.min()
        )  # Normalize for display
        label_mask = result["label"]
        pred_mask = result["pred"]
        iou = result["iou"]
        filename = result["filename"]

        # Print unique values for true mask and predicted mask
        unique_true = np.unique(label_mask)
        unique_pred = np.unique(pred_mask)

        # Plot the images
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"{filename} | Worst-{i+1} | IoU: {iou:.4f}")

        plt.subplot(1, 3, 1)
        plt.imshow(input_img)
        plt.title("Input Image")

        plt.subplot(1, 3, 2)
        plt.imshow(label_mask, cmap="gray")
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Prediction")

        plt.show()
        print(f"Unique values for {filename}:")
        print(f"Ground Truth Mask (True): {unique_true}")
        print(f"Prediction Mask (Pred): {unique_pred}")


def evaluate_dice_score(model, dataloader, device):
    """
    Evaluate a segmentation model using Dice score metric.

    Computes the average Dice score (1 - Dice loss) across the entire dataset.
    The Dice score measures the similarity between predicted and ground truth masks,
    with 1.0 indicating perfect overlap and 0.0 indicating no overlap.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model to evaluate (should output logits).
    dataloader : torch.utils.data.DataLoader
        Dataloader yielding tuples of (inputs, targets, _).
        Expected shapes:
        - inputs: (B, C, H, W) where B is batch size, C is channels
        - targets: (B, H, W) or (B, 1, H, W) binary masks
    device : torch.device
        Device to run evaluation on ('cuda' or 'cpu').

    Returns
    -------
    float
        Average Dice score across all samples in the dataloader.
    """
    model.eval()
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        # Wrap dataloader with tqdm for progress bar
        for inputs, targets, _ in tqdm(
            dataloader, desc="Evaluating", total=len(dataloader)
        ):
            inputs = inputs.to(device)
            targets = targets.to(device)
            print(inputs.shape)
            # Forward pass
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Compute dice loss and convert to score
            dice = 1 - dice_loss(preds, targets).item()

            # Accumulate metrics
            batch_size = inputs.size(0)
            total_dice += dice * batch_size
            total_samples += batch_size

    # Compute average Dice score
    avg_dice = total_dice / total_samples
    print(f"Dice Score (Accuracy): {avg_dice:.4f}")
    return avg_dice
