import cv2 as cv
import numpy as np
import PIL.Image
import PIL.ImageTk
import tensorflow as tf
from keras import layers, losses, models


class Model:
    TEST_FRAME_PATH = "frame.jpg"
    TRAIN_FRAME_SIZE = (150, 150)

    def train(self, counters):
        class_list = np.array([])
        train_length = (sum(counters),)

        if train_length[0] > 0:
            self.create_model(len(counters))
            img_shape = cv.imread("0/frame0.jpg").shape
            train_shape = train_length + img_shape
            img_list = np.zeros(train_shape)
            img_list_index = 0

            for i in range(len(counters)):
                for j in range(counters[i]):
                    img = cv.imread(f"{i}/frame{j}.jpg")
                    img_list[img_list_index] = img
                    class_list = np.append(class_list, i)
                    img_list_index += 1

            self.model.fit(img_list, class_list, epochs=10, verbose=2)
            print("Model successfully trained!")
            return True
        return False

    def create_model(self, output_num):
        self.model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(output_num, activation="linear"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def predict(self, frame):
        frame = frame[1]
        cv.imwrite(self.TEST_FRAME_PATH, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        img = PIL.Image.open(self.TEST_FRAME_PATH)
        img.thumbnail(self.TRAIN_FRAME_SIZE, PIL.Image.Resampling.LANCZOS)
        img.save(self.TEST_FRAME_PATH)

        img = cv.imread(self.TEST_FRAME_PATH)
        prediction = self.model.predict(np.array([img]))

        return prediction
