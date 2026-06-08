from pathlib import Path

import cv2


def main():
    image_path = Path(__file__).resolve().parent.parent / "data" / "road.png"
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = [
        ("Binary", cv2.THRESH_BINARY, 110),
        ("Binary Inv", cv2.THRESH_BINARY_INV, 110),
        ("Trunc", cv2.THRESH_TRUNC, 40),
        ("To Zero", cv2.THRESH_TOZERO, 40),
        ("To Zero Inv", cv2.THRESH_TOZERO_INV, 40),
    ]

    for window_name, threshold_type, threshold_value in variants:
        _, img_threshold = cv2.threshold(gray_image, threshold_value, 255, threshold_type)
        cv2.imshow(window_name, img_threshold)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
