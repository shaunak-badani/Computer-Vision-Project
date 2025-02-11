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

import sys
sys.path.append(os.path.abspath('segment-anything-2'))

from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# DEFINE DATA DIRECTORIES
images_a_dir = 'data/raw/AneRBC-I/Anemic_individuals/Original_images'
masks_a_dir = 'data/raw/AneRBC-I/Anemic_individuals/Binary_segmented'
images_h_dir = 'data/raw/AneRBC-I/Healthy_individuals/Original_images'
masks_h_dir = 'data/raw/AneRBC-I/Healthy_individuals/Binary_segmented'

# LOAD SAM2 MODEL
# sam2_checkpoint = "segment-anything-2/sam2_hiera_small.pt"
sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
# model_cfg = "C:/Users/saksh/OneDrive/Documents/Duke Academics/Spring 2025/Deep Learning/CV-Module-Project/segment-anything-2/sam2/configs/sam2/sam2_hiera_s.yaml"
model_cfg = "configs/sam2/sam2_hiera_s.yaml"

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


# TRAINING
def read_batch(data, visualize_data=False):
   # Select a random entry
   ent = data[np.random.randint(len(data))]

   # Get full paths
   Img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
   ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read annotation as grayscale

   if visualize_data:
       print(f"Image Chosen: {ent['image']}")

   if Img is None or ann_map is None:
       print(f"Error: Could not read image or mask from path {ent['image']} or {ent['annotation']}")
       return None, None, None, 0

   # Resize image and mask
   r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
   Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
   ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

   # Convert to binary mask (0 for background, 1 for masked region)
   binary_mask = np.where(ann_map == 255, 0, 1).astype(np.uint8)

   # Erode the binary mask to avoid boundary points
   eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

   # Get all coordinates inside the eroded mask (where the mask value is 1)
   coords = np.argwhere(eroded_mask > 0)

   # Select random points from the eroded mask
   num_points = 100  # Number of points to select

   # Randomly sample points from `coords` if there are enough points
   if len(coords) > num_points:
        points = coords[np.random.choice(len(coords), num_points, replace=False)]
   else:
        points = coords  # If fewer points exist, use all of them

   # Reformat points to (x, y) from (y, x)
   points = np.array([[p[1], p[0]] for p in points])

   if visualize_data:
        # Plotting the images and points
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(Img)
        plt.axis('off')

        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points in Different Colors
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=10, label=f'Point {i+1}')  # Corrected to plot y, x order

        # plt.legend()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

   binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (1024, 1024, 1)
   binary_mask = binary_mask.transpose((2, 0, 1))
   points = np.expand_dims(points, axis=1)

   # Return the image, binarized mask, points, and number of masks
   return Img, binary_mask, points, num_points


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice loss between y_true and y_pred.
    y_true: Ground truth mask (binary, 0 or 1)
    y_pred: Model output logits (before sigmoid)
    """
    y_pred = torch.sigmoid(y_pred)  # Convert logits to probabilities
    intersection = (y_true * y_pred).sum(dim=(1, 2))  # Sum over spatial dimensions
    union = y_true.sum(dim=(1, 2)) + y_pred.sum(dim=(1, 2))  # Sum over spatial dimensions

    dice = (2. * intersection + smooth) / (union + smooth)  # Dice coefficient
    return 1 - dice.mean()  # Dice loss


def validate_model(predictor, val_data):

    total_val_loss = 0
    total_val_iou = 0
    num_batches = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():  # No gradient updates
        for index in range(len(val_data)):
            image, mask, input_point, num_masks = read_batch(val_data, visualize_data=False)
            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])

            if gt_mask.shape[0] == 1 and prd_masks.shape[0] > 1:
                gt_mask = gt_mask.expand(prd_masks.shape[0], -1, -1)

            # Compute loss
            seg_bce_loss = criterion(prd_masks[:, 0], gt_mask)  
            seg_dice_loss = dice_loss(gt_mask, prd_masks[:, 0])  
            seg_loss = seg_bce_loss + seg_dice_loss  

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)

            total_val_loss += seg_loss.item()
            total_val_iou += np.mean(iou.cpu().detach().numpy())
            num_batches += 1

    avg_val_loss = total_val_loss / num_batches
    avg_val_iou = total_val_iou / num_batches

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")
    return avg_val_loss, avg_val_iou

def fine_tune_SAM2(train_data, val_data):
    # Configuring Hyperparameters
    # Train mask decoder.
    predictor.model.sam_mask_decoder.train(True)

    # Train prompt encoder.
    predictor.model.sam_prompt_encoder.train(True)

    # Configure optimizer.
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(), lr=0.0001, weight_decay=1e-4) #1e-5, weight_decay = 4e-5

    # Mix precision: more memory-efficient training strategy
    scaler = torch.cuda.amp.GradScaler()

    # No. of steps to train the model.
    NO_OF_STEPS = 5 # @param

    # Fine-tuned model name.
    FINE_TUNED_MODEL_NAME = "fine_tuned_sam2_600_100_300_tvt"

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
    accumulation_steps = 2  # Number of steps to accumulate gradients before updating

    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []

    mean_iou = 0

    for step in range(1, NO_OF_STEPS + 1):
        with torch.cuda.amp.autocast():
            image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False)
            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            if input_point.size == 0 or input_label.size == 0:
                continue
            
            image = image.copy()
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])

            # Compute loss
            criterion = torch.nn.BCEWithLogitsLoss()

            if gt_mask.shape[0] == 1 and prd_masks.shape[0] > 1:
                gt_mask = gt_mask.expand(prd_masks.shape[0], -1, -1)

            seg_bce_loss = criterion(prd_masks[:, 0], gt_mask)
            seg_dice_loss = dice_loss(gt_mask, prd_masks[:, 0])
            seg_loss = seg_bce_loss + seg_dice_loss

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            # Apply gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

            scheduler.step()

            if step % 500 == 0:
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".torch"
                torch.save(predictor.model.state_dict(), os.path.join('../models', FINE_TUNED_MODEL))

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            if step % 10 == 0:
                print(f"Step {step}:\t Accuracy (IoU) = {mean_iou:.4f}")
                train_losses.append(loss.item())  # Store full loss
                train_ious.append(mean_iou)       # Store running mean IoU

            # ---- Run Validation Every 100 Steps ----
            if step % 100 == 0:
                val_loss, val_iou = validate_model(predictor, val_data)
                val_losses.append(val_loss)
                val_ious.append(val_iou)

    return train_losses, train_ious, val_losses, val_ious

def main():
    train_data, val_data, test_data = prepare_data()
    train_losses, train_ious, val_losses, val_ious = fine_tune_SAM2(train_data, val_data)

    with open("../models/training_data.json", "w") as f:
        json.dump({"train_losses": train_losses, "train_ious": train_ious, 
                "val_losses": val_losses, "val_ious": val_ious}, f)
        
if __name__ == "__main__":
    main()



