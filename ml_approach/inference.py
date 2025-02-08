import cv2
import joblib
import numpy as np
from skimage.feature import hog

def predict_anemia(image_path):
    """
    Predict anemia from a single RBC image using saved model
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        dict: Prediction results containing class and probabilities
    """
    try:
        # Load the classifier
        classifier = joblib.load('anemia_classifier.joblib')
        model = classifier['model']
        scaler = classifier['scaler']
        hog_params = classifier['hog_params']
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
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
            'probabilities': {
                'Healthy': float(probabilities[0]),
                'Anemic': float(probabilities[1])
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None