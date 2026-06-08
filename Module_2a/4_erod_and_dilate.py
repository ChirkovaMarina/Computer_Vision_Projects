from pathlib import Path

import cv2
import numpy as np


def main():
    image_path = Path(__file__).resolve().parent.parent / "data" / "road.png"
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    for window_name, result in [
        ("Input", img),
        ("Erosion", img_erosion),
        ("Dilation", img_dilation),
    ]:
        cv2.imshow(window_name, result)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
