import cv2
import numpy as np


# Build a glare mask in the HLS color space (https://en.wikipedia.org/wiki/HSL_and_HSV)
# img - input image in BGR
# horizon_line - horizon line in pixels
# lightness - default threshold is 90%; very bright objects are treated as glare
def get_mask_of_glares(img, horizon_line=0, lightness=0.9):
    low_threshold = int(255 * lightness)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Apply binary thresholding to the Lightness channel.
    _, mask = cv2.threshold(hls[:, :, 1], low_threshold, 255, cv2.THRESH_BINARY)
    # Remove the mask above the horizon line.
    if 0 < horizon_line <= img.shape[1]:
        mask[0:horizon_line, 0:mask.shape[1]] = 0
    # White pixels correspond to glare.
    return mask


# Draw detected glare on the image.
def draw_glares(img, mask, color=(0, 255, 0)):
    colored_background = np.zeros(img.shape, np.uint8)
    colored_background[::] = color
    bk = cv2.bitwise_or(colored_background, colored_background, mask=mask)
    mask = cv2.bitwise_not(mask)
    fg = cv2.bitwise_or(img, img, mask=mask)
    result = cv2.bitwise_or(fg, bk)
    return result


# Suppress pixel brightness. This is purely a visual effect.
def suppress_lightness(img, mask, quality=0.9):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    non_zero = cv2.findNonZero(mask)
    if non_zero is None:
        return img.copy()
    for idx in non_zero:
        y, x = idx[0][0], idx[0][1]
        new_value = int(hls[x, y, 1] * quality)
        hls[x, y, 1] = np.clip(new_value, 0, 255)
    result = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    return result
