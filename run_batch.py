# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 02:21:26 2024

@author: 93969
"""

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import threshold_otsu
from tqdm import tqdm




def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def seg_slice(img_gray, mask_generator, otsu_flag=False):
    img_gray = NormalizeData(img_gray)
    
    # pre seg
    if otsu_flag:
        thresholds = threshold_otsu(img_gray)
        img_gray[img_gray<thresholds] = 0

    # Preprocess
    img_rgb = color.gray2rgb(img_gray)
    # img_rgb = img_rgb / np.max(img_rgb)
    img_rgb = (img_rgb * 255).astype(np.uint8)
    masks = mask_generator.generate(img_rgb)
    mask_slice = get_from_masks(masks)
    return mask_slice

def get_from_masks(masks):
    for i, m in enumerate(masks):
        area = m['area']
        if area> 40000 and area < 50000:
            break
    return masks[i]['segmentation'] 
 
if __name__=='__main__':

    fn_list = [r'E:/Research_Data/SAM/33303.tiff',
               r'E:/Research_Data/SAM/33308.tiff',
               r'E:/Research_Data/SAM/33310.tiff',
               r'E:/Research_Data/SAM/33313.tiff',
               r'E:/Research_Data/SAM/33318.tiff',
               r'E:/Research_Data/SAM/33323.tiff',
               r'E:/Research_Data/SAM/33325.tiff',
               r'E:/Research_Data/SAM/33333.tiff']
    for fn in fn_list:
        img_stack = io.imread(fn)
        # load model
        sam = sam_model_registry["default"](checkpoint="E:/Research_Data/SAM/sam_vit_h_4b8939.pth")
        device = 'cuda'
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        
        mask_stack = np.zeros(img_stack.shape)
        for i in tqdm(range(15, img_stack.shape[0])):
    
            mask = seg_slice(img_stack[i], mask_generator)
            mask_stack[i] = mask
            # plt.figure()
            # plt.imshow(mask)
        io.imsave(rf'{fn[:-5]}_seg.tiff', np.float32(mask_stack))


 
