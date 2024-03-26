from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping

import torch
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for idx, ann in enumerate(sorted_anns):
        # if idx < len(sorted_anns)-1:
        #     continue
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        
    ax.imshow(img)


import rasterio
IMAGES_PATH = 'c:/aifactory/train_img/'
MASKS_PATH = 'c:/aifactory/train_mask/'
MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

#band 이미지와 마스킹 이미지 비교
def show_band_images(image_path, mask_path):
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    axs = axs.ravel()
    
    for i in range(10):
        img = rasterio.open(image_path).read(i+1).astype(np.float32) / MAX_PIXEL_VALUE
        axs[i].imshow(img)
        axs[i].set_title(f'Band {i+1}')
        axs[i].axis('off')
    
    img = rasterio.open(mask_path).read(1).astype(np.float32) / MAX_PIXEL_VALUE
    axs[10].imshow(img)
    axs[10].set_title('Mask Image')
    axs[10].axis('off')
    axs[11].axis('off')
    plt.title('Band images compare Mask image')
    plt.tight_layout()
    plt.show() 

#밴드 조합 이미지 확인
def show_bands_image(image_path, band = (0,0,0)):
    img = rasterio.open(image_path).read(band).transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')  # 축 표시 없애기
    plt.title(f'Band {band} combine image')
    plt.show()
    return img

import cv2

BAND = (7,6,8)
# for i in range(0,3) :
#     #데이터 확인
#     show_band_images(IMAGES_PATH + f'train_img_{i}.tif', MASKS_PATH + f'train_mask_{i}.tif')
#     img = show_bands_image(IMAGES_PATH + f'train_img_{i}.tif', BAND) #사용
#     cv2.imwrite(f'./img_for_sam/img_{i}.png',img)

# image = cv2.imread('c:/study/python/train_img_3.jpg')
for i in range(10):
    image = rasterio.open(IMAGES_PATH + f'train_img_{i}.tif').read(BAND).transpose((1, 2, 0))
    # image = cv2.blur(image,(9,9))
    image = np.float32(image)/MAX_PIXEL_VALUE
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # 축 표시 없애기
    plt.title(f'Band {BAND} combine image')
    # plt.show()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.axis('off')
    # plt.show()

    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    sam_checkpoint = "c:/_data/SAM/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)

    print(len(masks))
    print(masks[0].keys())
    print(type(masks))
    for key, value in masks[0].items():
        print(f"======== {key} ========")
        print(value)
    print(len(masks[0]['segmentation']))

    plt.title('Origin')
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 