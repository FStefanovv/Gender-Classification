"""
Image processing for the pretrained MobileNetV2
"""

import cv2
import numpy as np
import dlib


AVG_HEIGHT = 107
AVG_WIDTH = 83


def process_image(img):
    """
    Necessary processing for the MobileNetV2 CNN

    Args:
        img (np.ndarray): image to resize

    Returns:
        np.ndarray: resized image with pixel values rescaled to [-1, 1]
    """

    img_resized = cv2.resize(img, (AVG_WIDTH, AVG_HEIGHT))

    img_scaled = img_resized.astype(np.float32) / 127.5 - 1

    return img_scaled


def get_faces(image):
    original_height, original_width = image.shape[:2]

    scale_factor = 2

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    image_resized = cv2.resize(image, (new_width, new_height))

    detector = dlib.get_frontal_face_detector()

    rects = detector(image_resized)

    if len(rects) == 0:
        raise ValueError("No faces detected")

    cropped_faces = []
    scaled_rects = []

    for i, r in enumerate(rects):
        x1, y1 = max(0, r.left()), max(0, r.top())
        x2, y2 = max(0, r.right()), max(0, r.bottom())

        face = image_resized[y1:y2, x1:x2]

        cropped_faces.append(face)

        scaled_x1 = int(x1 / scale_factor)
        scaled_y1 = int(y1 / scale_factor)
        scaled_x2 = int(x2 / scale_factor)
        scaled_y2 = int(y2 / scale_factor)

        scaled_rects.append((scaled_x1, scaled_y1, scaled_x2, scaled_y2))

    return cropped_faces, scaled_rects
