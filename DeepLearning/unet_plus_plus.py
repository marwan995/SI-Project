import torch
import torch.nn as nn
import time
from evaluation import calc_loss
from collections import defaultdict


class ConvBlock(nn.Module):
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.double_conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, feature_channels=[32, 64, 128, 256, 512], deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.enc0_0 = ConvBlock(in_channels, feature_channels[0])
        self.enc1_0 = ConvBlock(feature_channels[0], feature_channels[1])
        self.enc2_0 = ConvBlock(feature_channels[1], feature_channels[2])
        self.enc3_0 = ConvBlock(feature_channels[2], feature_channels[3])
        self.enc4_0 = ConvBlock(feature_channels[3], feature_channels[4])

        # Decoder blocks (nested skip connections)
        self.dec0_1 = ConvBlock(feature_channels[0] + feature_channels[1], feature_channels[0])
        self.dec1_1 = ConvBlock(feature_channels[1] + feature_channels[2], feature_channels[1])
        self.dec2_1 = ConvBlock(feature_channels[2] + feature_channels[3], feature_channels[2])
        self.dec3_1 = ConvBlock(feature_channels[3] + feature_channels[4], feature_channels[3])

        self.dec0_2 = ConvBlock(feature_channels[0]*2 + feature_channels[1], feature_channels[0])
        self.dec1_2 = ConvBlock(feature_channels[1]*2 + feature_channels[2], feature_channels[1])
        self.dec2_2 = ConvBlock(feature_channels[2]*2 + feature_channels[3], feature_channels[2])

        self.dec0_3 = ConvBlock(feature_channels[0]*3 + feature_channels[1], feature_channels[0])
        self.dec1_3 = ConvBlock(feature_channels[1]*3 + feature_channels[2], feature_channels[1])

        self.dec0_4 = ConvBlock(feature_channels[0]*4 + feature_channels[1], feature_channels[0])

        # Final convs for deep supervision
        if self.deep_supervision:
            self.final1 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
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
            return [out1, out2, out3, out4]  # You can average these later or apply custom loss
        else:
            return self.final(x0_4)
        

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10):
    best_loss = float("inf")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        start_time = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_metrics = {
                'loss': 0.0,
                'bce': 0.0,
                'dice': 0.0,
            }
            samples_processed = 0
            dataloader = train_loader if phase == 'train' else val_loader

            for inputs, masks, _ in dataloader: 
                inputs = inputs.to(device)
                masks = masks.to(device).float()

                # Zero gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss and metrics
                    batch_metrics = defaultdict(float)
                    loss = calc_loss(outputs, masks, batch_metrics)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                # Update epoch metrics
                batch_size = inputs.size(0)
                samples_processed += batch_size
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key] * batch_size

            # Calculate epoch-level metrics
            epoch_loss = epoch_metrics['loss'] / samples_processed
            epoch_bce = epoch_metrics['bce'] / samples_processed
            epoch_dice = epoch_metrics['dice'] / samples_processed

            # Print metrics
            print(f'{phase.capitalize()} Metrics:')
            print(f'Loss: {epoch_loss:.4f} | BCE: {epoch_bce:.4f} | Dice: {epoch_dice:.4f}')
            print()

            if phase == 'val':
                if epoch_loss < best_loss:
                    print(f'Validation loss improved from {best_loss:.4f} to {epoch_loss:.4f}')
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_model.pth')
            
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)


        # Calculate epoch time
        epoch_time = time.time() - start_time
        print(f'Epoch time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s\n')

    print(f'Training complete. Best validation loss: {best_loss:.4f}')
    return model