import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input


class FeatureExtractor:
    def __init__(self, model_weights='imagenet'):
        self.model = VGG16(weights=model_weights, include_top=False)

    def extract_features(self, img_path, target_size=(224, 224)):
        img_name = os.path.basename(img_path)
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array)
        return img_name, features.flatten()
