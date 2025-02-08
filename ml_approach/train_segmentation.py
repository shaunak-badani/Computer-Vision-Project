import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load image and corresponding mask
def load_data(image_dir, mask_dir):
    images, masks = [], []
    
    for img_name in os.listdir(image_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is not None and mask is not None:
                images.append(img)
                masks.append(mask)

    return images, masks

# Extract features from an image (HOG + Color)
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False, channel_axis=None)

    # Compute color histograms (R, G, B channels)
    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()

    return np.hstack([hog_features, hist_r, hist_g, hist_b])

# Prepare dataset for pixel-wise classification
def prepare_dataset(images, masks):
    X, y = [], []
    
    for img, mask in zip(images, masks):
        h, w, _ = img.shape
        img_features = extract_features(img)

        # Flatten mask and repeat features for all pixels
        mask_flat = mask.flatten()
        X.append(img_features)
        y.append(mask_flat)
    
    return np.array(X), np.array(y).flatten()

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/rf_segmentation.pkl")
    print("Model saved successfully.")
    
    return model

# Perform segmentation on new images
def segment_image(model, image):
    features = extract_features(image)
    predictions = model.predict(features)
    
    segmented_mask = predictions.reshape(image.shape[:2])
    
    return segmented_mask

# Main execution
image_dir = "data/raw/AneRBC-I/Anemic_individuals/Original_images"
mask_dir = "data/raw/AneRBC-I/Anemic_individuals/Masks"

images, masks = load_data(image_dir, mask_dir)
X, y = prepare_dataset(images, masks)

model = train_model(X, y)

# Test segmentation on a new image
test_image = cv2.imread("data/raw/AneRBC-I/Anemic_individuals/Original_images/sample.png")
segmented_result = segment_image(model, test_image)

cv2.imshow("Segmented Output", segmented_result * 255)  # Save segmentation result
