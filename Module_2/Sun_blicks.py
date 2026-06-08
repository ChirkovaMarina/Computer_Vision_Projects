import cv2
import numpy as np

# Remove sun glare with a sequence of filters.
def remove_glare(frame):
    # Convert the image to HSV.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split channels.
    h, s, v = cv2.split(hsv)

    # Apply adaptive brightness correction.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 15))
    v_clahe = clahe.apply(v)

    # Merge channels back into a single image.
    hsv_processed = cv2.merge((h, s, v_clahe))

    # Convert back to BGR.
    image_processed = cv2.cvtColor(hsv_processed, cv2.COLOR_HSV2BGR)

    return image_processed




# Open the video file.
video_path = "../data/processing/trm.168.090.avi"
cap = cv2.VideoCapture(video_path)

# Process each video frame.
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Apply the sun-glare removal function.
    frame_without_glare = remove_glare(frame)

    # Display the processed frame.
    cv2.imshow("Processed Video", frame_without_glare)

    # Stop processing when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
