from pathlib import Path

import cv2
import numpy as np


def main():
    image_path = Path(__file__).resolve().parent.parent / "data" / "road.png"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    for window_name, result in [
        ("Input", image),
        ("Opening", opening),
        ("Closing", closing),
        ("Gradient", gradient),
        ("Top Hat", tophat),
        ("Black Hat", blackhat),
    ]:
        cv2.imshow(window_name, result)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
