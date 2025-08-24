import os
import pickle
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm
import config

class FeatureExtractor:
    def __init__(self):
        print("Loading InceptionV3 model...")
        base_model = InceptionV3(weights='imagenet')
        self.model = Model(base_model.input, base_model.layers[-2].output)
        print("Feature extractor loaded!")
    
    def extract_features(self):
        """Extract features from all images"""
        print("Extracting features from images...")
        
        if not os.path.exists(config.IMAGE_DIR):
            print(f"Image directory not found: {config.IMAGE_DIR}")
            return {}
        
        features = {}
        image_files = os.listdir(config.IMAGE_DIR)
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        print(f"Found {len(image_files)} images in {config.IMAGE_DIR}")
        
        if len(image_files) == 0:
            print("No image files found!")
            return {}
        
        for filename in tqdm(image_files, desc="Extracting features"):
            try:
                # Load and preprocess image
                image_path = os.path.join(config.IMAGE_DIR, filename)
                image = load_img(image_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                
                # Extract features
                feature = self.model.predict(image, verbose=0)
                image_id = os.path.splitext(filename)[0]  # Remove extension
                features[image_id] = feature.flatten()
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"Successfully extracted features for {len(features)} images")
        return features
    
    def save_features(self, features):
        """Save extracted features"""
        if not os.path.exists(config.PROCESSED_DIR):
            os.makedirs(config.PROCESSED_DIR)
            
        feature_path = os.path.join(config.PROCESSED_DIR, 'features.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {feature_path}")

# Run feature extraction
if __name__ == "__main__":
    extractor = FeatureExtractor()
    features = extractor.extract_features()
    
    if features:
        extractor.save_features(features)
        print("✅ Feature extraction completed successfully!")
    else:
        print("❌ No features extracted")
