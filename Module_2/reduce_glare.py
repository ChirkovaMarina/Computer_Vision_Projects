import math

import cv2
import numpy as np

VIDEO_PATH = '../data/processing/trm.179.003.avi'


class GlareReduction:
    """Reduce glare from bright light sources in video."""

    def __init__(self, path: str) -> None:
        self.path = path  # Video path to process.
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX  # Font for overlay labels.
        self.font_scale = 1.2  # Font scale for overlay labels.
        self.image = None  # Currently processed image.

    def put_text(self, image: np.ndarray, text: str) -> None:
        """Draw the label `text` on `image`."""

        text_size = cv2.getTextSize(text, self.font_face, self.font_scale, 2)[0]

        # Compute coordinates for text placement.
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = int(image.shape[0] * 0.9)

        cv2.putText(image, text, (text_x, text_y), self.font_face, self.font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    def _apply_polynomial_function(self, *args: float) -> None:
        """Apply a polynomial transform with coefficients `args` to `self.image`."""

        table = np.array([args[0] + args[1] * i + args[2] * (i ** 2) + args[3] * (i ** 3)
                          for i in np.arange(0, 256)], dtype='uint8')
        cv2.LUT(self.image, table, self.image)

    def _apply_gamma_correction(self, gamma=1.0) -> None:
        """Apply gamma correction with coefficient `gamma` to `self.image`."""

        table = np.array([((i / 255) ** gamma) * 255 for i in np.arange(0, 256)], dtype='uint8')
        cv2.LUT(self.image, table, self.image)

    def reduce_glare(self, image: np.ndarray) -> np.ndarray:
        """Reduce glare in `image`."""

        self.image = image.copy()

        # * FIRST POLYNOMIAL FUNCTION
        # Coefficients are selected so the function pushes pixel intensity
        # toward the threshold value of 100.
        # * Result: reduce glare and lower contrast.
        self._apply_polynomial_function(0, 1.657766, -0.009157128, 0.00002579473)

        # * GAMMA CORRECTION WITH `gamma` == 4 / 3 == 1.(3)
        # Value > 1 increases contrast and reduces brightness.
        self._apply_gamma_correction(4 / 3)

        # * SECOND POLYNOMIAL FUNCTION
        # Coefficients are selected so the function pushes pixel intensity
        # toward the threshold value of 160.
        # * Result: reduce glare and lower contrast.
        # This effect is weaker than the first polynomial function.
        self._apply_polynomial_function(-4.263256 * math.exp(-14), 1.546429, -0.005558036, 0.00001339286)

        # * GAMMA CORRECTION WITH `gamma` == 5 / 4 == 1.25
        # Value > 1 increases contrast and reduces brightness.
        # `gamma` is closer to 1, so the effect is milder than the first gamma correction.
        self._apply_gamma_correction(5 / 4)

        return self.image

    def run(self) -> None:
        """Process the video and display the result."""

        cap = cv2.VideoCapture(self.path)

        while cap.isOpened():
            frame_read, frame = cap.read()

            if not frame_read:
                cv2.destroyAllWindows()
                cap.release()
                break

            original_image = frame.copy()
            self.put_text(original_image, 'ORIGINAL IMAGE')

            processed_image = self.reduce_glare(frame)
            self.put_text(processed_image, 'REDUCED GLARE')

            result = np.hstack((original_image, processed_image))

            cv2.imshow('Reducing glare', result)
            cv2.waitKey(10)


if __name__ == '__main__':
    gr = GlareReduction(VIDEO_PATH)
    gr.run()
