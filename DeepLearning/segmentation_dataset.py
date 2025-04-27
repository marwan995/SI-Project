from torch.utils.data import Dataset
import rasterio
from utils import per_band_minmax
import torch
import torchvision.transforms as T
from const import images_to_remove
import os


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading multi-band satellite imagery and masks for segmentation tasks.

    This dataset handles TIFF/TIF format images with corresponding mask files,
    applying optional transforms and automatically performing min-max normalization.

    Parameters
    ----------
    image_dir : str
        Directory containing input images (multi-band TIFF/TIF files).
    mask_dir : str
        Directory containing corresponding mask files (single-band TIFF/TIF files).
    transform : callable, optional
        Transformations to be applied to the input images. If provided, masks will be
        resized to match the transformed image dimensions (default: None).

    Attributes
    ----------
    image_filenames : list[str]
        List of valid image filenames (excluding those in images_to_remove).

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx)
        Returns the idx-th sample (image, mask, filename).
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = [
            f
            for f in os.listdir(image_dir)
            if f.endswith(".tiff") or f.endswith(".tif") and not f in images_to_remove
        ]

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Get the idx-th sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            Contains:
            - image : torch.Tensor
                4-channel normalized image tensor (4, H, W)
            - mask : torch.Tensor
                Binary mask tensor (1, H, W)
            - img_name : str
                Original filename of the sample
        """
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
        mask = mask > 0.5  # Binary mask

        if self.transform:
            image = self.transform(image)
            mask = T.Resize(image.shape[1:])(mask)

        return image, mask, img_name
