import os
import glob
import argparse
import numpy as np
import pandas as pd
import rasterio
import cv2
from PIL import Image
from tqdm import tqdm

from loading import load_model
from prediction import predict_mask
from encode_decode import rle_encode


def per_band_minmax(image):
    """
    Normalize each band of the image to [0, 1] range using min-max scaling.
    Works for both single image (H, W, C) and batched images (N, H, W, C).
    """
    if len(image.shape) == 3:  # Single image (H, W, C)
        normalized = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[-1]):
            band = image[..., c]
            normalized[..., c] = (band - band.min()) / (band.max() - band.min() + 1e-6)
    else:  # Assuming batched images (N, H, W, C)
        normalized = np.zeros_like(image, dtype=np.float32)
        for n in range(image.shape[0]):
            for c in range(image.shape[-1]):
                band = image[n, ..., c]
                normalized[n, ..., c] = (band - band.min()) / (band.max() - band.min() + 1e-6)
    return normalized


def preprocess_image(path, target_size=(256, 256)):
    """Load and preprocess a single image for ML model prediction"""
    with rasterio.open(path) as src:
        img = src.read()  # shape (4, H, W)
    
    # Transpose to (H, W, C) and resize
    img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize per band
    img = per_band_minmax(img)
    return img


def run_inference(input_folder, model_path, output_csv):
    """Run inference on all TIFF files in input folder using ML model"""
    # Load model
    model = load_model(model_path)
    
    records = []
    tif_paths = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    
    for path in tqdm(tif_paths, desc="Processing images"):
        file_id = os.path.splitext(os.path.basename(path))[0]
        
        # Preprocess image
        img = preprocess_image(path)
        
        # Predict mask
        mask = predict_mask(img, model)
        
        # Threshold to binary (if not already binary)
        mask = (mask > 0.5).astype(np.uint8)
        
        # RLE encode
        rle = rle_encode(mask)
        records.append((file_id, rle))
    
    # Save CSV
    df = pd.DataFrame(records, columns=["id", "rle"])
    df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")


def run_inference_with_mask(input_folder, model_path, output_csv):
    """Run inference and also save mask images"""
    # Load model
    model = load_model(model_path)
    
    # Create output folder for masks
    output_folder = os.path.join("output")
    os.makedirs(output_folder, exist_ok=True)
    
    records = []
    tif_paths = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    
    for path in tqdm(tif_paths, desc="Processing images"):
        file_id = os.path.splitext(os.path.basename(path))[0]
        
        # Preprocess image
        img = preprocess_image(path)
        
        # Predict mask
        mask = predict_mask(img, model)
        
        # Threshold to binary
        mask_bin = (mask > 0.5).astype(np.uint8)
        
        # RLE encode
        rle = rle_encode(mask_bin)
        records.append((file_id, rle))
        
        # Save mask as image
        output_path = os.path.join(output_folder, f"{file_id}_mask.png")
        mask_img = (mask_bin * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(output_path)
    
    # Save CSV
    df = pd.DataFrame(records, columns=["id", "rle"])
    df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")
    print(f"Masks saved to folder: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a folder of TIFFs using ML model and produce a submission CSV"
    )
    parser.add_argument(
        "--input_folder",
        "-i",
        required=True,
        help="Path to folder containing test .tif files",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        default="random_forest_model.pkl",
        help="Path to your saved model (.pkl)",
    )
    parser.add_argument(
        "--output_csv",
        "-o",
        default="predictions.csv",
        help="Where to write the submission CSV",
    )
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Flag to output the masks themselves",
    )

    args = parser.parse_args()

    if args.masks:
        run_inference_with_mask(
            input_folder=args.input_folder,
            model_path=args.model_path,
            output_csv=args.output_csv,
        )
    else:
        run_inference(
            input_folder=args.input_folder,
            model_path=args.model_path,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()