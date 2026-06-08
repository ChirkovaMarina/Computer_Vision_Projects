
import cv2 as cv
from pathlib import Path


def main():
    video_path = Path(__file__).resolve().parent.parent / "data" / "city" / "trm.169.007.avi"
    cap = cv.VideoCapture(str(video_path))
    res = True
    while res:
        res, im = cap.read()
        if res is False:
            break
        image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, [540, 480])
        fast = cv.FastFeatureDetector_create()
        kp = fast.detect(image, None)
        img2 = cv.drawKeypoints(image, kp, None, color=(255, 0, 255))
        cv.putText(img2, "Threshold: {}".format(fast.getThreshold()), (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(img2, "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()), (10, 200),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(img2, "neighborhood: {}".format(fast.getType()), (10, 300),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(img2, "Total Keypoints: {}".format(len(kp)), (10, 400),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
        fast.setNonmaxSuppression(0)
        kp = fast.detect(image, None)
        img3 = cv.drawKeypoints(image, kp, None, color=(255, 0, 0))
        cv.putText(img3, "Total Keypoints: {}".format(len(kp)), (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
        img4 = cv.hconcat((img2, img3))
        cv.imshow("fast", img4)
        key = cv.waitKey(10)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
