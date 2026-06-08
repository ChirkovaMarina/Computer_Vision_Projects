import numpy as np
import cv2 as cv

drawing = False   # True while the mouse button is pressed.
mode = False   # If True, rectangles are drawn instead of circles; toggle with 'm'.
ix, iy = -1, -1

# Mouse event handler.
def draw_circle(event, x, y, flags,param):
    global ix, iy, drawing, mode
    if event == cv.EVENT_LBUTTONDOWN:  # User pressed the left mouse button.
        drawing = True
        ix, iy = x, y    # Update coordinates to the current position.
    elif event == cv.EVENT_MOUSEMOVE:    # User moved the mouse.
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:    # User released the left mouse button.
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x,  y), 5, (0, 0, 255), -1)

img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)  # Register the mouse callback for the selected window.
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k != 255:
        print(k)
    if k == ord('m'): # Switch drawing mode between circles and rectangles.
        mode = not mode
    elif k == 27 or k == 113: # 27 is Esc, 113 is 'q'; either exits the program.
        break
cv.destroyAllWindows()
