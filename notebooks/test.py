# IMPORT REQUIRED LIBRARIES
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.colors as mcolors
import pandas as pd
import json
import torch
import torch.nn.utils
from inference import run_inference, calculate_metrics, process_img

import sys
sys.path.append(os.path.abspath('segment-anything-2'))

from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# DEFINE DATA DIRECTORIES
images_a_dir = '../data/raw/AneRBC-I/Anemic_individuals/Original_images'
masks_a_dir = '../data/raw/AneRBC-I/Anemic_individuals/Binary_segmented'
images_h_dir = '../data/raw/AneRBC-I/Healthy_individuals/Original_images'
masks_h_dir = '../data/raw/AneRBC-I/Healthy_individuals/Binary_segmented'

# LOAD SAM2 MODEL
sam2_checkpoint = "../segment-anything-2/sam2_hiera_small.pt"
model_cfg = "C:/Users/saksh/OneDrive/Documents/Duke Academics/Spring 2025/Deep Learning/CV-Module-Project/segment-anything-2/sam2/configs/sam2/sam2_hiera_s.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)



def prepare_data():
    df = pd.DataFrame(
    {
        "ImageId": sorted(os.listdir(images_a_dir) + os.listdir(images_h_dir)),
        "MaskId": sorted(os.listdir(masks_a_dir) + os.listdir(masks_h_dir))
    }
)

    # Load the train.csv file
    train_df = df.copy()

    # 600 for fine-tuning, 100 for validation and 300 for testing
    train_df, test_df = train_test_split(train_df, test_size=0.3, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.142, random_state=42)

    # Prepare the training data list
    train_data = []
    for index, row in train_df.iterrows():
        image_name = row['ImageId']
        mask_name = row['MaskId']

        # Append image and corresponding mask paths
        if image_name in os.listdir(images_a_dir):
                images_dir = images_a_dir
                masks_dir = masks_a_dir
        else:
                images_dir = images_h_dir
                masks_dir = masks_h_dir

        train_data.append({
            "image": os.path.join(images_dir, image_name),
            "annotation": os.path.join(masks_dir, mask_name)
        })


    val_data = []
    for index, row in val_df.iterrows():
        image_name = row['ImageId']
        mask_name = row['MaskId']

        if image_name in os.listdir(images_a_dir):
                images_dir = images_a_dir
                masks_dir = masks_a_dir
        else:
                images_dir = images_h_dir
                masks_dir = masks_h_dir

        # Append image and corresponding mask paths
        val_data.append({
            "image": os.path.join(images_dir, image_name),
            "annotation": os.path.join(masks_dir, mask_name)
        })

    # Prepare the testing data list (if needed for inference or evaluation later)
    test_data = []
    for index, row in test_df.iterrows():
        image_name = row['ImageId']
        mask_name = row['MaskId']

        if image_name in os.listdir(images_a_dir):
                images_dir = images_a_dir
                masks_dir = masks_a_dir
        else:
                images_dir = images_h_dir
                masks_dir = masks_h_dir

        # Append image and corresponding mask paths
        test_data.append({
            "image": os.path.join(images_dir, image_name),
            "annotation": os.path.join(masks_dir, mask_name)
        })

    return train_data, val_data, test_data


train_data, val_data, test_data = prepare_data()

mean_pixel_accuracy = []
mean_iou_acc = []
mean_dc = []
mean_pre = []
mean_recall = []
mean_speci = []
mean_dl = []

for test_img in tqdm(test_data):
    image_path = test_img['image']
    mask_path = test_img['annotation']
    img = cv2.imread(image_path)[..., ::-1] 
    mask = cv2.cvtColor(cv2.imread(mask_path)[..., ::-1], cv2.COLOR_BGR2GRAY)
    mask = process_img(mask)
    mask = np.where(mask > 128, 255, 0).astype(np.uint8)

    image, seg_map = run_inference(img)

    pixel_acc, iou_score, dice_score, precision_score, recall_score, specificity_score, loss = calculate_metrics(mask, seg_map)
    mean_pixel_accuracy.append(pixel_acc)
    mean_iou_acc.append(iou_score)
    mean_dc.append(dice_score)
    mean_pre.append(precision_score)
    mean_recall.append(recall_score)
    mean_speci.append(specificity_score)
    mean_dl.append(loss)


print(f"Mean Pixel Accuracy for test data: {np.mean(mean_pixel_accuracy):.4f}")
print(f"Mean IoU for test data: {np.mean(mean_iou_acc):.4f}")
print(f"Mean Dice Coefficient for test data: {np.mean(mean_dc):.4f}")
print(f"Mean Precision for test data: {np.mean(mean_pre):.4f}")
print(f"Mean Recall for test data: {np.mean(mean_recall):.4f}")
print(f"Mean Specificity for test data: {np.mean(mean_speci):.4f}")
print(f"Mean Dice Loss for test data: {np.mean(mean_dl):.4f}")


# Mean Pixel Accuracy for test data: 0.9262
# Mean IoU for test data: 0.8865
# Mean Dice Coefficient for test data: 0.9334
# Mean Precision for test data: 0.9639
# Mean Recall for test data: 0.9062
# Mean Specificity for test data: 0.9554
# Mean Dice Loss for test data: 0.0666