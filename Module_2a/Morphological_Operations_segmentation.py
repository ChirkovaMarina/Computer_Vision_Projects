import cv2
import numpy as np
from pathlib import Path

def segmentation(img):
    # Define the kernel.
    kernel = np.ones((15, 15), np.uint8)

    # Apply erosion.
    erosion = cv2.erode(img, kernel, iterations=1)

    # OTSU
    ret, thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((20, 20), np.uint8)

    # Apply the morphological gradient.
    gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

    gray = gradient

    # Find the rectangular contour.
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find external contours.
    external_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 15e6:  # Minimum area threshold.
            external_contours.append(cnt)

    # Create the binary image.
    binary = np.zeros_like(img)
    cv2.drawContours(binary, external_contours, -1, (255, 255, 255), cv2.FILLED)

    mask = gradient - binary

    mask[:, :650] = 0
    mask[:200, :] = 0
    mask[binary.shape[0] - 300:, :] = 0

    mask[:, binary.shape[1] - 1000:] = 0

    # Kernel for closing.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))

    # Apply closing.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


if __name__ == "__main__":
    image_path = (
        Path(__file__).resolve().parent.parent
        / 'data'
        / 'add_data'
        / 'camera2'
        / 'no-defects-14_40_18-13.12.2022.png'
    )
    img = cv2.imread(str(image_path), 0)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = segmentation(img)
    small = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    cv2.imshow("win", small)
    cv2.waitKey(0)
