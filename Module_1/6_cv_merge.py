import cv2

image = cv2.imread("./image.jpg")
b, g, r = cv2.split(image)  # Split into Blue, Green and Red channels.

cv2.imshow("original", image)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("blue", b)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("green", g)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("red", r)
cv2.waitKey()
cv2.destroyAllWindows()

merged = cv2.merge([b, g, r])  # Merge channels back together.
cv2.imshow("merged", merged)
cv2.waitKey()
cv2.destroyAllWindows()
