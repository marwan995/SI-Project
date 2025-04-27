import torch
import torch.nn as nn
import time
from evaluation import calc_loss
from collections import defaultdict


class ConvBlock(nn.Module):
    """
    A double convolutional block with batch normalization and ReLU activation.

    This block consists of two consecutive 3x3 convolutions, each followed by
    batch normalization and ReLU activation. Weights are initialized using
    Kaiming normal initialization for convolutions and constant initialization
    for batch norm layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.

    Attributes
    ----------
    double_conv : nn.Sequential
        The sequential container holding the double convolution layers with
        batch norm and activations.

    Methods
    -------
    forward(x)
        Forward pass of the block.
    _init_weights()
        Initializes weights for convolutional and batch norm layers.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Kaiming normal for conv layers and constants for batch norm.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.double_conv(x)


class UNetPlusPlus(nn.Module):
    """
    Implementation of UNet++ (Nested UNet) architecture for image segmentation.

    This architecture features nested and dense skip connections between encoder
    and decoder paths, improving gradient flow and feature propagation compared
    to standard UNet. Supports deep supervision for multi-level outputs.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default: 4).
    out_channels : int, optional
        Number of output channels/classes (default: 1).
    feature_channels : list[int], optional
        Number of channels at each encoder level (default: [32, 64, 128, 256, 512]).
    deep_supervision : bool, optional
        Whether to enable deep supervision (output at multiple decoder levels)
        (default: False).

    Attributes
    ----------
    encX_0 : ConvBlock
        Encoder blocks where X is the level (0-4).
    decX_Y : ConvBlock
        Decoder blocks where X is the starting level and Y is the nesting level.
    final* : nn.Conv2d
        Final 1x1 convolution layers for output.
    maxpool : nn.MaxPool2d
        Max pooling layer for downsampling.
    up : nn.Upsample
        Upsampling layer with bilinear interpolation.

    Methods
    -------
    forward(x)
        Forward pass through the network.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        feature_channels=[32, 64, 128, 256, 512],
        deep_supervision=False,
    ):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.enc0_0 = ConvBlock(in_channels, feature_channels[0])
        self.enc1_0 = ConvBlock(feature_channels[0], feature_channels[1])
        self.enc2_0 = ConvBlock(feature_channels[1], feature_channels[2])
        self.enc3_0 = ConvBlock(feature_channels[2], feature_channels[3])
        self.enc4_0 = ConvBlock(feature_channels[3], feature_channels[4])

        # Decoder blocks (nested skip connections)
        self.dec0_1 = ConvBlock(
            feature_channels[0] + feature_channels[1], feature_channels[0]
        )
        self.dec1_1 = ConvBlock(
            feature_channels[1] + feature_channels[2], feature_channels[1]
        )
        self.dec2_1 = ConvBlock(
            feature_channels[2] + feature_channels[3], feature_channels[2]
        )
        self.dec3_1 = ConvBlock(
            feature_channels[3] + feature_channels[4], feature_channels[3]
        )

        self.dec0_2 = ConvBlock(
            feature_channels[0] * 2 + feature_channels[1], feature_channels[0]
        )
        self.dec1_2 = ConvBlock(
            feature_channels[1] * 2 + feature_channels[2], feature_channels[1]
        )
        self.dec2_2 = ConvBlock(
            feature_channels[2] * 2 + feature_channels[3], feature_channels[2]
        )

        self.dec0_3 = ConvBlock(
            feature_channels[0] * 3 + feature_channels[1], feature_channels[0]
        )
        self.dec1_3 = ConvBlock(
            feature_channels[1] * 3 + feature_channels[2], feature_channels[1]
        )

        self.dec0_4 = ConvBlock(
            feature_channels[0] * 4 + feature_channels[1], feature_channels[0]
        )

        # Final convs for deep supervision
        if self.deep_supervision:
            self.final1 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        """
        Forward pass through the UNet++ network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            - If deep_supervision=False: Single output tensor of shape
              (batch_size, out_channels, height, width)
            - If deep_supervision=True: List of output tensors from multiple
              decoder levels (shapes may vary by level)
        """
        # Encoder
        x0_0 = self.enc0_0(x)
        x1_0 = self.enc1_0(self.maxpool(x0_0))
        x2_0 = self.enc2_0(self.maxpool(x1_0))
        x3_0 = self.enc3_0(self.maxpool(x2_0))
        x4_0 = self.enc4_0(self.maxpool(x3_0))

        # Nested Decoder
        x0_1 = self.dec0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.dec1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.dec2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.dec3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        x0_2 = self.dec0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.dec1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.dec2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        x0_3 = self.dec0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.dec1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        x0_4 = self.dec0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # Output
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [
                out1,
                out2,
                out3,
                out4,
            ]  # You can average these later or apply custom loss
        else:
            return self.final(x0_4)


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10
):
    """
    Train a segmentation model with validation and learning rate scheduling.

    Performs training and validation loops for the specified number of epochs,
    tracking BCE loss, Dice loss, and combined loss. Implements gradient clipping,
    model checkpointing, and learning rate scheduling.

    Parameters
    ----------
    model : torch.nn.Module
        The segmentation model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data yielding (inputs, masks, _).
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data yielding (inputs, masks, _).
    optimizer : torch.optim.Optimizer
        Optimizer for model parameter updates.
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler (e.g., ReduceLROnPlateau). Can be None.
    device : torch.device
        Device to train on ('cuda' or 'cpu').
    num_epochs : int, optional
        Number of training epochs (default: 10).

    Returns
    -------
    torch.nn.Module
        The trained model.
    """

    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        start_time = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_metrics = {
                "loss": 0.0,
                "bce": 0.0,
                "dice": 0.0,
            }
            samples_processed = 0
            dataloader = train_loader if phase == "train" else val_loader

            for inputs, masks, _ in dataloader:
                inputs = inputs.to(device)
                masks = masks.to(device).float()

                # Zero gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    outputs = model(inputs)

                    # Calculate loss and metrics
                    batch_metrics = defaultdict(float)
                    loss = calc_loss(outputs, masks, batch_metrics)

                    # Backward pass + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                # Update epoch metrics
                batch_size = inputs.size(0)
                samples_processed += batch_size
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key] * batch_size

            # Calculate epoch-level metrics
            epoch_loss = epoch_metrics["loss"] / samples_processed
            epoch_bce = epoch_metrics["bce"] / samples_processed
            epoch_dice = epoch_metrics["dice"] / samples_processed

            # Print metrics
            print(f"{phase.capitalize()} Metrics:")
            print(
                f"Loss: {epoch_loss:.4f} | BCE: {epoch_bce:.4f} | Dice: {epoch_dice:.4f}"
            )
            print()

            if phase == "val":
                if epoch_loss < best_loss:
                    print(
                        f"Validation loss improved from {best_loss:.4f} to {epoch_loss:.4f}"
                    )
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), "best_model.pth")

                if scheduler is not None:
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(epoch_loss)

        # Calculate epoch time
        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s\n")

    print(f"Training complete. Best validation loss: {best_loss:.4f}")
    return model
