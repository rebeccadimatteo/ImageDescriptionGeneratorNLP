from data_preparation.FeatureExtraction import FeatureExtractor
import os
import numpy as np


class ImageProcessor:
    def __init__(self, folder_path, model_weights='imagenet'):
        self.folder_path = folder_path
        self.feature_extractor = FeatureExtractor(model_weights)

    def process_images(self, save_path='../processed/features.npy'):
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        all_features = []

        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            img_name, features = self.feature_extractor.extract_features(image_path)
            all_features.append((img_name, features))

        dtype = [('image_name', 'U100'), ('features', np.float32, (len(all_features[0][1]),))]
        all_features_array = np.array(all_features, dtype=dtype)

        np.save(save_path, all_features_array)
        print(f"Features salvate in: {save_path}")
