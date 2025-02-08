import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed

def extract_features(image_path, mask_path):
    """Extracts morphological, texture, and color features from a masked image."""
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = np.where(mask == 255, 0, 1).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmented = cv2.bitwise_and(image, image, mask=binary_mask)
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    # Morphological features
    props = regionprops(label(mask))
    rbc_count = len(contours)
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

# Paths to images and masks
base_dir = "data/raw/AneRBC-I"
image_dirs = {
    'healthy': os.path.join(base_dir, 'Healthy_individuals/Original_images'),
    'anemic': os.path.join(base_dir, 'Anemic_individuals/Original_images')
}
mask_dirs = {
    'healthy': os.path.join(base_dir, 'Healthy_individuals/Binary_Segmented'),
    'anemic': os.path.join(base_dir, 'Anemic_individuals/Binary_Segmented')
}
labels_file = os.path.join(base_dir, 'labels.csv')

# Load image file names and labels
labels_df = pd.read_csv(labels_file)

# Function to process a single row
def process_row(row):
    """Processes a single row from labels DataFrame."""
    filename = row['filename']
    label = row['label']

    if filename in os.listdir(image_dirs['healthy']):
        image_path = os.path.join(image_dirs['healthy'], filename)
        mask_path = os.path.join(mask_dirs['healthy'], filename)
    else:
        image_path = os.path.join(image_dirs['anemic'], filename)
        mask_path = os.path.join(mask_dirs['anemic'], filename)

    if os.path.exists(image_path) and os.path.exists(mask_path):
        features = extract_features(image_path, mask_path)
        if features:
            return features, label
    return None  # Skip if invalid

# Use joblib.Parallel for parallel feature extraction
num_cores = joblib.cpu_count()  # Get number of CPU cores
results = Parallel(n_jobs=num_cores)(
    delayed(process_row)(row) for _, row in tqdm(labels_df.iterrows(), total=len(labels_df))
)

# Filter out None values
results = [r for r in results if r is not None]
data, labels = zip(*results)  # Unzip into separate lists

# Convert to DataFrame
feature_names = ["rbc_count", "area", "perimeter", "eccentricity", "contrast", "correlation", "energy", "homogeneity", "mean_intensity", "mean_red", "std_red", "red_green_ratio"]
df = pd.DataFrame(data, columns=feature_names)
df.to_csv(os.path.join(base_dir, 'segmentation_features.csv'), index=False)
