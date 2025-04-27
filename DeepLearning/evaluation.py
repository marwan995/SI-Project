import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()

    return dice_loss


def calc_loss(pred, target, metrics, bce_weight=0.5):
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
