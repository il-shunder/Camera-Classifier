import cv2 as cv


class Camera:
    FRAME_WIDTH = 600
    FRAME_HEIGHT = 400

    def __init__(self):
        self.camera = cv.VideoCapture(0)

        if not self.camera.isOpened():
            raise ValueError("Unable to open camera!")

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()

            if ret:
                frame = cv.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return None
