import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import kagglehub

def get_image_paths(base_dir):
    """Get paths for healthy and anemic images"""
    healthy_dir = os.path.join(base_dir, 'AneRBC_dataset/AneRBC-II/Healthy_individuals/Original_images')
    anemic_dir = os.path.join(base_dir, 'AneRBC_dataset/AneRBC-II/Anemic_individuals/Original_images')
    
    healthy_images = [(os.path.join(healthy_dir, img), 0) for img in os.listdir(healthy_dir) if img.endswith('.png')]
    anemic_images = [(os.path.join(anemic_dir, img), 1) for img in os.listdir(anemic_dir) if img.endswith('.png')]
    
    return healthy_images + anemic_images

def process_image(img_path):
    """Convert image to grayscale and normalize"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

def show_hog(img_path):
    """Show original, grayscale and HOG features"""
    # Get images
    original = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    gray = process_image(img_path)
    
    # Get HOG
    _, hog_img = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2), visualize=True, channel_axis=None)
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(original)
    ax2.imshow(gray, cmap='gray')
    ax3.imshow(hog_img, cmap='gray')
    plt.show()

def extract_features(img_paths_and_labels):
    """Extract HOG features from all images"""
    features, labels = [], []
    
    for i, (path, label) in enumerate(img_paths_and_labels):
        if i % 100 == 0:
            print(f'Processing image {i}/{len(img_paths_and_labels)}')
            
        img = process_image(path)
        if img is not None:
            feat = hog(img, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=False, channel_axis=None)
            features.append(feat)
            labels.append(label)
            
    return np.array(features), np.array(labels)

def main():
    # Load dataset
    dataset = kagglehub.dataset_download("jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images")
    base_dir = os.path.join(dataset, 'AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120')
    
    # Get image paths and show HOG features for samples
    image_paths = get_image_paths(base_dir)
    show_hog(image_paths[0][0])  # Show healthy sample
    show_hog(image_paths[-1][0])  # Show anemic sample
    
    # Extract features
    X, y = extract_features(image_paths)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train and evaluate
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

if __name__ == "__main__":
    main()