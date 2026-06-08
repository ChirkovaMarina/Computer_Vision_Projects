from pathlib import Path

import cv2
import numpy as np


def main():
    image_path = Path(__file__).resolve().parent.parent / "data" / "road.png"
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresholds = [
        ("Threshold 50", cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]),
        ("Threshold 100", cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]),
        ("Threshold 150", cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]),
    ]

    # 1 = foreground, -1 = background, 0 = don't care.
    kernel = np.array(
        [
            [-1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
        ],
        dtype=np.int8,
    )

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    for window_name, thresholded in thresholds:
        hitmiss = cv2.morphologyEx(thresholded, cv2.MORPH_HITMISS, kernel)
        cv2.imshow(window_name, thresholded)
        cv2.waitKey(0)
        cv2.imshow(f"Hit-or-Miss {window_name}", hitmiss)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
