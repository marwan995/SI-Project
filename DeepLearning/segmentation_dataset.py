from torch.utils.data import Dataset
import rasterio
from utils import per_band_minmax
import torch
import torchvision.transforms as T
from const import images_to_remove
import os 

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = [
            f for f in os.listdir(image_dir) if f.endswith(".tiff") or f.endswith(".tif") and not f in images_to_remove
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        with rasterio.open(img_path) as img_src:
            image = img_src.read([1, 2, 3, 4])  # Read bands 1-4
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)  # Read first band

        image = per_band_minmax(image)

        image = torch.from_numpy(image).float()  # (4, 512, 512)
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, 512, 512)
        mask = (mask > 0.5)  # Binary mask


        if self.transform:
            image = self.transform(image)
            mask = T.Resize(image.shape[1:])(mask)


        return image, mask,img_name