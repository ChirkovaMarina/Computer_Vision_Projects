import cv2
from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import detection_flares as df

SHOW_SUPPRESS_LIGHTNESS = True
SHOW_SRC = True
SHOW_DRAW_MASK = True
SHOW_MASK = False

HARD_HORIZON_LINE_PX = 170

if __name__ == "__main__":
    # Place windows in fixed positions for easier comparison.
    cv2.namedWindow("Src")
    cv2.namedWindow("Mask")
    cv2.namedWindow("Draw_mask")
    cv2.namedWindow("Suppress_lightness")
    cv2.moveWindow("Src", 5, 5)
    cv2.moveWindow("Mask", 965, 5)
    cv2.moveWindow("Draw_mask", 5, 545)
    cv2.moveWindow("Suppress_lightness", 965, 545)
    video_path = MODULE_DIR.parent / 'data' / 'processing' / 'trm.174.007.avi'
    vid_capture = cv2.VideoCapture(str(video_path))
    if not vid_capture.isOpened():
        print("Failed to open the video file")
    else:
        file_count = 0
        while vid_capture.isOpened():
            ret, frame = vid_capture.read()
            if ret:
                # Build a glare mask for the current frame.
                mask = df.get_mask_of_glares(frame, HARD_HORIZON_LINE_PX)
                if SHOW_MASK:
                    cv2.imshow('Mask', mask)
                if SHOW_SRC:
                    cv2.imshow('Src', frame)
                if SHOW_DRAW_MASK:
                    res_dm = df.draw_glares(frame, mask)
                    cv2.imshow('Draw_mask', res_dm)
                if SHOW_SUPPRESS_LIGHTNESS:
                    res_sl = df.suppress_lightness(frame, mask, 0.94)  # Tune the third parameter if needed.
                    cv2.imshow('Suppress_lightness', res_sl)
                key = cv2.waitKey(20)
                if (key == ord('q')) or key == 27:
                    break
            else:
                break
    # Release the video capture handle.
    vid_capture.release()
    cv2.destroyAllWindows()
