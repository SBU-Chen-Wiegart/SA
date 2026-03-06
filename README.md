# README: Segment Anything

## Overview

This script performs automatic segmentation on grayscale TIFF image stacks using the **Segment Anything Model (SAM)**. It reads one or more TIFF stacks, generates a segmentation mask for each slice, and saves the output as a new TIFF stack.

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
* A valid SAM checkpoint file, such as `sam_vit_h_4b8939.pth`
* A CUDA-capable GPU if using `device = 'cuda'`

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

Segments a single grayscale image slice.

**Steps:**

1. Normalize the grayscale image
2. Optionally apply Otsu thresholding
3. Convert grayscale image to RGB
4. Convert to 8-bit format
5. Generate candidate masks using SAM
6. Select one mask using `get_from_masks()`

**Input:**

* `img_gray`: 2D grayscale image
* `mask_generator`: initialized SAM mask generator
* `otsu_flag`: whether to apply Otsu thresholding before segmentation

**Output:**

* binary segmentation mask for the slice

### `get_from_masks(masks)`

Selects a segmentation mask whose area is within a predefined range.

**Current rule:**

* Select the first mask with area between **40000** and **50000** pixels

**Input:**

* `masks`: list of masks returned by SAM

**Output:**

* selected mask segmentation array

## Script Workflow

When the script is run directly:

1. A list of TIFF stack filenames is defined in `fn_list`
2. Each TIFF stack is loaded using `skimage.io.imread()`
3. The SAM model is loaded from the checkpoint file
4. A `SamAutomaticMaskGenerator` is created
5. Each slice from index `15` to the end is segmented
6. The resulting mask stack is saved as:

```python
original_filename_seg.tiff
```

For example:

```python
33303.tiff -> 33303_seg.tiff
```

## Input and Output

### Input

* One or more grayscale TIFF stacks
* SAM checkpoint file

### Output

* Segmented TIFF mask stack for each input file


