import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import kagglehub
import joblib
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def extract_hog_features(image):
    """
    Extract HOG features from an image
    """
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=None
    )
    return features

def visualize_processing_steps(image, save_path=None):
    """
    Visualize the processing steps: original, HOG features, and Otsu threshold
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    _, otsu = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    hog_features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None
    )
    
    hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(otsu, cmap='gray')
    plt.title("Otsu's Thresholding")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(hog_image, cmap='viridis')
    plt.title('HOG Features')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def preprocess_image(image_path, visualize=False):
    """
    Load and preprocess image with optional visualization
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        if visualize:
            viz_dir = 'visualizations'
            os.makedirs(viz_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            viz_path = os.path.join(viz_dir, f'{base_name}_processing_steps.png')
            visualize_processing_steps(gray, save_path=viz_path)
            print(f"Visualization saved to: {viz_path}")
        
        return gray
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def create_feature_dataset(image_paths, labels):
    """
    Create a dataset of HOG features from images
    """
    features_list = []
    processed_labels = []
    total_images = len(image_paths)
    
    first_visualization_done = False
    
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        try:
            if (idx + 1) % 100 == 0:
                print(f"Processing image {idx + 1}/{total_images}")
            
            visualize = not first_visualization_done
            img = preprocess_image(img_path, visualize=visualize)
            
            if img is not None:
                hog_features = extract_hog_features(img)
                features_list.append(hog_features)
                processed_labels.append(label)
                
                if visualize:
                    first_visualization_done = True
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    if len(features_list) == 0:
        raise ValueError("No features could be extracted from the images")
        
    return np.array(features_list), np.array(processed_labels)

def load_dataset(dataset_path):
    """
    Load images and create labels from the dataset structure
    """
    base_dir = os.path.join(dataset_path, 
                           'AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120', 
                           'AneRBC_dataset')
    
    healthy_path = os.path.join(base_dir, 'AneRBC-II', 'Healthy_individuals', 'Original_images')
    anemic_path = os.path.join(base_dir, 'AneRBC-II', 'Anemic_individuals', 'Original_images')
    
    print(f"Checking paths:")
    print(f"Healthy path: {healthy_path}")
    print(f"Healthy path exists: {os.path.exists(healthy_path)}")
    print(f"Anemic path: {anemic_path}")
    print(f"Anemic path exists: {os.path.exists(anemic_path)}")
    
    image_paths = []
    labels = []
    
    if os.path.exists(healthy_path):
        print(f"\nLoading healthy images from: {healthy_path}")
        for img_name in os.listdir(healthy_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(healthy_path, img_name))
                labels.append(0)
    
    if os.path.exists(anemic_path):
        print(f"Loading anemic images from: {anemic_path}")
        for img_name in os.listdir(anemic_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(anemic_path, img_name))
                labels.append(1)
    
    print(f"\nTotal images found: {len(image_paths)}")
    print(f"Healthy images: {labels.count(0)}")
    print(f"Anemic images: {labels.count(1)}")
    
    return image_paths, labels

def train_lightgbm(X_train, X_test, y_train, y_test):
    """
    Train and evaluate LightGBM model with correct parameters
    """
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and create visualizations
    """
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    feature_importance = pd.DataFrame({
        'feature': range(len(model.feature_importances_)),
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(20), feature_importance['importance'][:20])
    plt.title('Top 20 Most Important HOG Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()

def inference_pipeline(image_path):
    """
    Run inference on a single image using saved model
    """
    try:
        # Load saved model and scaler
        model = LGBMClassifier()
        model.load_model('anemia_model.txt')
        scaler = joblib.load('scaler.joblib')
        
        # Preprocess single image
        img = preprocess_image(image_path)
        if img is None:
            return None
            
        # Extract HOG features
        hog_features = extract_hog_features(img)
        
        # Reshape for single sample
        hog_features = hog_features.reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(hog_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        result = {
            'prediction': 'Anemic' if prediction == 1 else 'Healthy',
            'confidence': float(probability[prediction]),
            'probabilities': {
                'Healthy': float(probability[0]),
                'Anemic': float(probability[1])
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None

def main():
    np.random.seed(42)
    
    print("Getting dataset path...")
    dataset_path = kagglehub.dataset_download(
        "jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images"
    )
    print(f"Dataset path: {dataset_path}")
    
    print("\nLoading dataset...")
    image_paths, labels = load_dataset(dataset_path)
    
    if len(image_paths) == 0:
        raise ValueError("No images found. Please check the dataset structure.")
    
    print("\nExtracting HOG features...")
    X, y = create_feature_dataset(image_paths, labels)
    
    if len(X) == 0:
        raise ValueError("No features could be extracted. Check image processing.")
    
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining LightGBM model...")
    model = train_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("\nEvaluating model...")
    evaluate_model(model, X_test_scaled, y_test)
    
    print("\nSaving classifier...")
    classifier = {
        'model': model,
        'scaler': scaler,
        'hog_params': {
            'orientations': 9,
            'pixels_per_cell': (16, 16),
            'cells_per_block': (2, 2)
        }
    }
    joblib.dump(classifier, 'anemia_classifier.joblib')
    print("Model saved as anemia_classifier.joblib")

if __name__ == "__main__":
    main()
