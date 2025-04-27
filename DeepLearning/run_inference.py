import os
import glob
import argparse

import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision.transforms as T

from unet_plus_plus import UNetPlusPlus, ConvBlock
from encode_decode import rle_encode


def load_model(model_path, device):
    model = UNetPlusPlus().to(device)

    state = torch.load(model_path, map_location=device)

    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


def preprocess_image(path, transform, device):
    # read 4-band image
    with rasterio.open(path) as src:
        img = src.read()  # shape (4, H, W)
    # convert to tensor (C, H, W)
    tensor = torch.from_numpy(img.astype(np.float32))
    # resize
    tensor = transform(tensor)
    # add batch dim
    return tensor.unsqueeze(0).to(device)


def run_inference(input_folder, model_path, output_csv, device):
    # build resize transform
    transform = T.Compose([T.Resize((256, 256))])
    # load model
    model = load_model(model_path, device)

    records = []
    tif_paths = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    for path in tif_paths:
        file_id = os.path.splitext(os.path.basename(path))[0]
        inp = preprocess_image(path, transform, device)
        with torch.no_grad():
            out = model(
                inp
            )  # tensor of shape (1, 1, 256, 256) or list if deep_supervision
            # if deep_supervision, take last output
            if isinstance(out, (list, tuple)):
                out = out[-1]
        # to CPU numpy
        mask = out.squeeze().cpu().numpy()
        # threshold to binary
        mask = (mask > 0.5).astype(np.uint8)
        # RLE encode
        rle = rle_encode(mask)
        records.append((file_id, rle))

    # save CSV
    df = pd.DataFrame(records, columns=["id", "rle"])
    df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")


def run_inference_with_mask(input_folder, model_path, output_csv, device):
    # build resize transform
    transform = T.Compose([T.Resize((256, 256))])
    # load model
    model = load_model(model_path, device)

    # create output folder for masks
    output_folder = os.path.join("output")
    os.makedirs(output_folder, exist_ok=True)

    records = []
    tif_paths = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    for path in tif_paths:
        file_id = os.path.splitext(os.path.basename(path))[0]
        inp = preprocess_image(path, transform, device)
        with torch.no_grad():
            out = model(
                inp
            )  # tensor of shape (1, 1, 256, 256) or list if deep_supervision
            # if deep_supervision, take last output
            if isinstance(out, (list, tuple)):
                out = out[-1]
        # to CPU numpy
        mask = out.squeeze().cpu().numpy()
        # threshold to binary
        mask_bin = (mask > 0.5).astype(np.uint8)

        # RLE encode
        rle = rle_encode(mask_bin)
        records.append((file_id, rle))

        # save mask as image
        output_path = os.path.join(output_folder, f"{file_id}_mask.png")
        # scale to 0-255 for saving
        mask_img = (mask_bin * 255).astype(np.uint8)
        from PIL import Image

        Image.fromarray(mask_img).save(output_path)

    # save CSV
    df = pd.DataFrame(records, columns=["id", "rle"])
    df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")
    print(f"Masks saved to folder: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a folder of TIFFs and produce a submission CSV"
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
        default="dl_9166.pth",
        help="Path to your saved model (.pth)",
    )
    parser.add_argument(
        "--output_csv",
        "-o",
        default="predictions.csv",
        help="Where to write the submission CSV",
    )
    parser.add_argument(
        "--device", "-d", default="cuda", help="Torch device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--masks", action="store_true", help="Flag to output the masks themselves"
    )

    args = parser.parse_args()

    if args.masks:
        run_inference_with_mask(
            input_folder=args.input_folder,
            model_path=args.model_path,
            output_csv=args.output_csv,
            device=torch.device(args.device),
        )
    else:
        run_inference(
            input_folder=args.input_folder,
            model_path=args.model_path,
            output_csv=args.output_csv,
            device=torch.device(args.device),
        )


if __name__ == "__main__":
    main()
