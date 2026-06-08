import cv2

image = cv2.imread("./image.jpg")
height, width = image.shape[:2]  # The first two elements of the shape tuple.

start_row = int(height * 0.25)  # int converts float to int by rounding down.
start_col = int(width * 0.25)

end_row = int(height * 0.80)
end_col = int(width * 0.80)

cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow("original_image", image)
cv2.waitKey()

cv2.imshow("cropped", cropped)
cv2.waitKey()

# Flip the image horizontally.
flipped_img = cv2.flip(cropped, 1)

# Display the flipped image with imshow().
cv2.imshow("Flipped Image", flipped_img)
cv2.waitKey()

# Flip the image vertically and display it with imshow().
flipped_img = cv2.flip(cropped, 0)
cv2.imshow("Flipped Image", flipped_img)
cv2.waitKey()

# Flip the image vertically and horizontally, then display it with imshow().
flipped_img = cv2.flip(cropped, -1)
cv2.imshow("Flipped Image", flipped_img)
cv2.waitKey()
cv2.destroyAllWindows()
