import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_and_process_image(img_path):
    """Load image, apply Otsu's thresholding and extract HOG features"""
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    # Process image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract HOG features
    features = hog(thresh, orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), visualize=False, channel_axis=None)
    
    return features

def main():
    base_dir = "data/raw/AneRBC-I"
    healthy_dir = os.path.join(base_dir, 'Healthy_individuals/Original_images')
    anemic_dir = os.path.join(base_dir, 'Anemic_individuals/Original_images')
    

    # Create dataset
    features, labels = [], []
    
    # Process healthy images
    for img in os.listdir(healthy_dir):
        if img.endswith('.png'):
            feat = load_and_process_image(os.path.join(healthy_dir, img))
            if feat is not None:
                features.append(feat)
                labels.append(0)
    
    # Process anemic images
    for img in os.listdir(anemic_dir):
        if img.endswith('.png'):
            feat = load_and_process_image(os.path.join(anemic_dir, img))
            if feat is not None:
                features.append(feat)
                labels.append(1)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train and evaluate
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))

if __name__ == "__main__":
    main()