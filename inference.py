import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
import joblib
from skimage.feature import hog
import joblib


import torch
import torch.nn.utils

import sys
sys.path.append(os.path.abspath('segment-anything-2'))

from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "segment-anything-2/sam2_hiera_small.pt"
model_cfg = "C:/Users/saksh/OneDrive/Documents/Duke Academics/Spring 2025/Deep Learning/CV-Module-Project/segment-anything-2/sam2/configs/sam2/sam2_hiera_s.yaml"

# Load the fine-tuned model
sam2_fine_tuned = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor_fine_tuned = SAM2ImagePredictor(sam2_fine_tuned)
FINE_TUNED_MODEL_WEIGHTS = "models/fine_tuned_sam2_400_training_1000.torch"
# Build net and load weights
predictor_fine_tuned.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2)

def process_img(img):  
    r = min(1024 / img.shape[1], 1024 / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    return img


def get_points(image_shape, num_points):  
    points = np.random.randint(0, [image_shape[1], image_shape[0]], size=(num_points, 1, 2))
    return points

def run_sam2_out_of_the_box_with_prompt(img):
    
    # Load the selected image and mask
    image = process_img(img)

    # Generate random points for the input
    num_samples = 50  # Number of points per segment to sample
    input_points = get_points(image.shape, num_samples)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(point_coords=input_points, point_labels=np.ones([input_points.shape[0], 1]))

    # Process the predicted masks and sort by scores
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    seg_map = 1 - sorted_masks[0]

    return seg_map



def run_inference(img):
    # Load the selected image and mask
    image = process_img(img)

    # Generate random points for the input
    num_samples = 50  # Number of points per segment to sample
    input_points = get_points(image.shape, num_samples)

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



def show_otsus_thresholding(img):
    """Show original, grayscale and HOG features"""

    # Process image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract HOG features
    _, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None
    )
    
    hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
    
    return hog_image, thresh


def predict_anemia_lgbm(img):
    """
    Predict anemia from a single RBC image using saved model
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        dict: Prediction results containing class and probabilities
    """
    try:
        # Load the classifier
        classifier = joblib.load('models/anemia_classifier.joblib')
        model = classifier['model']
        scaler = classifier['scaler']
        hog_params = classifier['hog_params']
        
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Extract HOG features using saved parameters
        hog_features = hog(
            gray,
            orientations=hog_params['orientations'],
            pixels_per_cell=hog_params['pixels_per_cell'],
            cells_per_block=hog_params['cells_per_block'],
            visualize=False,
            channel_axis=None
        )
        
        # Reshape and scale features
        hog_features = hog_features.reshape(1, -1)
        scaled_features = scaler.transform(hog_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        result = {
            'prediction': 'Anemic' if prediction == 1 else 'Healthy',
            'confidence': float(probabilities[prediction]),
        }
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
    

def extract_features(image, mask):
    """Extracts morphological, texture, and color features from a masked image."""

    binary_mask = np.where(mask == 255, 0, 1).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_RBC_AREA = 150.0
    filtered_contours = []

    for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > MIN_RBC_AREA:
                filtered_contours.append(contour)

    segmented = cv2.bitwise_and(image, image, mask=binary_mask)
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    # Morphological features
    props = regionprops(label(mask))
    rbc_count = len(filtered_contours)
    area, perimeter, eccentricity = 0, 0, 0
    if props:
        area = props[0].area
        perimeter = props[0].perimeter
        eccentricity = props[0].eccentricity

    # Texture features (GLCM) - Optimized by reducing gray levels
    gray = cv2.convertScaleAbs(gray, alpha=(255.0 / 64))  # Reduce gray levels to 64
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Color features (Histogram)
    hist = cv2.calcHist([segmented], [0], mask, [256], [0, 256])
    mean_intensity = np.mean(hist)
    mean_red = np.mean(segmented[:, :, 2])  # Red channel intensity
    std_red = np.std(segmented[:, :, 2])  # Variation in red intensity
    red_green_ratio = mean_red / (np.mean(segmented[:, :, 1]) + 1e-7)  # Ratio of Red to Green
    

    return [rbc_count, area, perimeter, eccentricity, contrast, correlation, energy, homogeneity, mean_intensity, mean_red, std_red, red_green_ratio]


def predict_anemia_dt(img, mask):
    try:
        # Load the classifier
        classifier = joblib.load('models/decision_tree_model.joblib')
        model = classifier['model']
        scaler = classifier['scaler']
        
        features = extract_features(img, mask)
        scaled_features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        result = {
            'prediction': 'Anemic' if prediction == 1 else 'Healthy',
            'confidence': float(probabilities[prediction]),
        }
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

