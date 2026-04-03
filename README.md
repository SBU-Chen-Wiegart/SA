# README: Segment Anything

## Overview

This script performs automatic segmentation on **2D** grayscale TIFF image stacks using the **Segment Anything Model (SAM)**. The method generates a segmentation mask each 2d slice of X-ray image, and stacks the output as a new TIFF stack.

The workflow is designed for batch processing of image stacks and is especially useful when the object of interest occupies a roughly consistent area range across slices.

## Main Features

* Loads grayscale TIFF stacks from disk
* Converts each slice to RGB format for SAM input
* Optionally applies **Otsu thresholding** before segmentation
* Uses **SAM Automatic Mask Generator** to generate candidate masks
* Selects a mask based on a target area range
* Saves the segmented results as a TIFF stack
* Supports batch processing of multiple files

## Requirements

Install the required Python packages before running the script:

```python
pip install numpy opencv-python matplotlib scikit-image tqdm
```

You also need:

* `segment_anything`
* A valid SAM checkpoint file, such as `sam_vit_h_4b8939.pth` https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
* A CUDA-capable GPU if using `device = 'cuda'`

## Run
The code need to be run under python environment **test_SAM** in the new linux workstation

**Steps:**
```python
conda activate test_SAM
code
```

## Imported Packages

The script uses the following libraries:

* `segment_anything`
* `skimage`
* `numpy`
* `cv2`
* `matplotlib`
* `tqdm`

## Functions

### `NormalizeData(data)`

Normalizes the image intensities to the range `[0, 1]`.

**Input:**

* `data`: input image array

**Output:**

* normalized image array

### `seg_slice(img_gray, mask_generator, otsu_flag=False)`


**Input:**

* `img_gray`: 2D grayscale image
* `mask_generator`: initialized SAM mask generator
* `otsu_flag`: whether to apply Otsu thresholding before segmentation

**Output:**

* binary segmentation mask for the slice

### `get_from_masks(masks)`

Selects a segmentation mask whose area is within a predefined range.

## Input and Output

### Input

* One or more grayscale TIFF stacks
* SAM checkpoint file

### Output

* Segmented TIFF mask stack for each input file


