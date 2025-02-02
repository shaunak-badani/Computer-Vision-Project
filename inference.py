import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.colors as mcolors
import pandas as pd
import streamlit as st


import torch
import torch.nn.utils

import sys
sys.path.append(os.path.abspath('segment-anything-2'))

from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "segment-anything-2/sam2_hiera_small.pt"
model_cfg = "C:/Users/saksh/OneDrive/Documents/Duke Academics/Spring 2025/Deep Learning/CV-Module-Project/segment-anything-2/sam2/configs/sam2/sam2_hiera_s.yaml"

# Load the fine-tuned model
sam2_fine_tuned = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor_fine_tuned = SAM2ImagePredictor(sam2_fine_tuned)

# Load the out of the box model
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"), apply_postprocessing=False)
sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2)

def process_img(img):  
    r = min(1024 / img.shape[1], 1024 / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    return img


def get_points(image_shape, num_points):  
    points = np.random.randint(0, [image_shape[1], image_shape[0]], size=(num_points, 1, 2))
    return points

def run_sam2_out_of_the_box(img):
    
    masks = sam2_mask_generator.generate(img)
    masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)

    print(masks[0])

    return 1 - masks[0]['segmentation']


def run_inference(img):
    # Load the selected image and mask
    image = process_img(img)

    # Generate random points for the input
    num_samples = 50  # Number of points per segment to sample
    input_points = get_points(image.shape, num_samples)

    # Load the fine-tuned model
    FINE_TUNED_MODEL_WEIGHTS = "models/fine_tuned_sam2_test_1_1000.torch"

    # Build net and load weights
    predictor_fine_tuned.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

    # Perform inference and predict masks
    with torch.no_grad():
        image = image.copy()
        predictor_fine_tuned.set_image(image)
        masks, scores, logits = predictor_fine_tuned.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Process the predicted masks and sort by scores
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    seg_map = 1 - sorted_masks[0]

    return image, seg_map


# Function to compute Pixel Accuracy
def pixel_accuracy(y_true, y_pred):
    correct_pixels = np.sum(y_true == y_pred)
    total_pixels = np.prod(y_true.shape)
    return correct_pixels / total_pixels

# Function to compute IoU (Intersection over Union)
def iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union if union != 0 else 0.0

# Function to compute Dice Coefficient
def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred)) if (np.sum(y_true) + np.sum(y_pred)) != 0 else 0.0

# Function to compute Specificity
def specificity(y_true, y_pred):
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    possible_negatives = np.sum(y_true == 0)
    return true_negatives / possible_negatives if possible_negatives != 0 else 0.0

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0.0

def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0.0

# Function to compute Dice Loss (1 - Dice Coefficient)
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Example: Calculate metrics for a given test case
def calculate_metrics(mask, seg_map):
    # Convert mask and segmentation map to binary (0 or 1)
    mask_binary = (mask > 0).astype(np.uint8)
    seg_map_binary = (seg_map > 0).astype(np.uint8)

    # Calculate each metric
    pixel_acc = pixel_accuracy(mask_binary, seg_map_binary)
    iou_score = iou(mask_binary, seg_map_binary)
    dice_score = dice_coef(mask_binary, seg_map_binary)
    precision_score = precision(mask_binary, seg_map_binary)
    recall_score = recall(mask_binary, seg_map_binary)
    specificity_score = specificity(mask_binary, seg_map_binary)
    loss = dice_loss(mask_binary, seg_map_binary)

    return pixel_acc, iou_score, dice_score, precision_score, recall_score, specificity_score, loss