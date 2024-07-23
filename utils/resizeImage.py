import cv2
import numpy as np


def resize_images(images, size):
    resized_images = []
    for img in images:
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        resized_images.append(img_resized)
    return np.array(resized_images)
