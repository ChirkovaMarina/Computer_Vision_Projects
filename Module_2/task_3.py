import cv2 as cv
import numpy as np

class Reader:
    """Video stream processing."""
    def initialize(self, path_to_video):
        self.cap = cv.VideoCapture(path_to_video)
        self.prevFrame = np.array([])

    def run(self):
        if not self.cap.isOpened():
            print("Failed to open the file.")
        else:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break
                self.on_frame(frame)
                cv.imshow('VideoPlayer', frame)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv.destroyAllWindows()

    def on_frame(self, frame):
        if self.prevFrame.size != 0:                            # If a previous frame exists in history...
            mask = cv.absdiff(frame, self.prevFrame)            # ...compute the frame difference.
            cv.normalize(mask, mask, 0, 120, cv.NORM_MINMAX)    # ...normalize brightness to reduce overexposed replacement areas.
            cv.addWeighted(frame, 1.0, mask, 1.0, 0.0, frame)   # ...overlay the normalized difference onto the frame.
        self.prevFrame = frame

if __name__ == '__main__':
    init_args = {
        'path_to_video': '../data/processing/trm.168.005.avi'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')