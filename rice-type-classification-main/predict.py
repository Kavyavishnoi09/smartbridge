import os
from typing import List, Tuple

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def load_rice_model(model_path: str):
    """Load the trained Keras model if it exists, otherwise return None."""
    if os.path.exists(model_path):
        return load_model(model_path)
    return None


def predict_rice_type(model, img_path: str, class_names: List[str]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Preprocess image, run prediction, and return sorted class scores."""
    if model is None:
        raise FileNotFoundError("Model file not found.")

    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = round(float(np.max(prediction)) * 100, 2)

    scores = []
    for class_name, score in zip(class_names, prediction):
        scores.append((class_name, round(float(score) * 100, 2)))
    scores.sort(key=lambda item: item[1], reverse=True)

    return predicted_class, confidence, scores
