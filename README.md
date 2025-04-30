<div align="center">
    <h1 align='center'>☁️<i>AI-powered Cloud Masking</i>☁️</h1>
</div>

<details open="open">
<summary>
<h2 style="display:inline">📝 Table of Contents</h2>
</summary>

- [📑 Description](#description)
- [📦 Packages](#packages)
- [⛏️ Usage](#usage)
- [🔬 Approach](#approach)
- [⚙️ Model Specs](#model)

</details>

## Description <a name = "description"></a>

This project tackles the challenge of cloud obstruction in optical satellite imagery by building an AI system that detects and generates clouds masks. By generating cloud masks we can improving image clarity with other models to remove clouds, this can enhance the use of satellite data for applications like urban planning and environmental monitoring.

## Packages <a name = "packages"></a>

The following packages/modules were used for developing the DL model

- rasterio
- numpy
- matplotlib
- torch
- torchvision
- tqdm
- pandas
- seaborn
- scikit-learn
- dask
- pickle5
- pickle-mixin
- opencv-python
- pyarrow

### Note

Be careful of the installation of `torch` and `torchvision` as they need special installation commands to use GPU for running and training the model\
Use [Pytorch Docs](https://pytorch.org/get-started/locally/) for detailed installation instructions

## Usage <a name = "usage"></a>

1. Run `pip install -r requirements.txt` to install all the required packages
2. Go inside the `DeepLearning` directory
3. Run the inference script using `python ./run_inference.py -i <input_directory>` to get predictions.csv that has RLE encoded masks
   - The flag `--masks` can be used to get the raw masks

## Approach <a name = "approach"></a>

### Deep Learning

As we need to do a pixel wise classification we needed to use a model with an architecture like U-Net and we actually tried it and got a score of 75% which made us eager to find a better solution with a higher score.

We found papers talking about U-Net++ and it's usage in medical application with implies it's robustness as medical applications are very strict.

After using U-Net++ we got much better results but still the score wasn't up to our expectations so we investigated the misclassified images and to our surprise we found out that the training (ground truth) masks had some errors with some masks showing only zeros while the whole image has clouds.

At this point the model was showing very promising results as it could get good segmentation of the images with bad masks, so we used it to help detect the bad masks in the dataset and remove them from training as they're not representative to the test set and may lead to a drop in score.

### Machine Learning

We usually had good accuracy with random forest models so we used one here too with the same resizing of the DL model so input images were 256\*256 pixels.

The model had a pipeline with the first stage being a `PCA` stage then the `Random Forest` model itself.

The `Radom Forest` model had 100 trees and max depth of 15.

The model had accuracy on the test set of `87%` which was impressively close to the DL model.

## Model Specs <a name = "model"></a>

The model had the following specifications

- **Number of Parameters:** 9.16 M
- **Number of Operations:** 276.72 GOps
- **Size:** 35 MB
