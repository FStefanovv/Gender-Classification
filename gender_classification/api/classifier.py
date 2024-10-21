import os

import numpy as np


from tensorflow.keras import models

from image_processing import process_image


class GenderClassifier:
    """Class for classifying the gender of the face

    Attributes:
        _model (Keras model instance): Trained model for gender classification
    """

    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "../../model/best_model.keras")
        self._model = models.load_model(model_path)

    def classify(self, images):
        """Classifies a list of faces

        Args:
            images (list of np.ndarray): list of face images to classify

        Returns:
            list of tuple: A list of tuples where each tuple contains:
                - class name (str)
                - label (int)
        """
        images_processed = []

        for img in images:
            processed = process_image(img)
            images_processed.append(processed)

        images_batched = np.stack(images_processed)

        predictions = self._model.predict(images_batched)

        results = []

        for pred in predictions:
            class_name = "Male" if pred > 0.5 else "Female"
            results.append((class_name, int(pred > 0.5)))

        return results
