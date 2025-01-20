import cv2 as cv
import numpy as np
import tensorflow as tf
from keras import datasets, layers, losses, models


class Model:
    def __init__(self):
        self.model = models.Sequential(
            [
                # layers.Input(shape=(150, 150, 1)),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(10, activation="linear"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self, counters):
        img_list = np.array([])
        class_list = np.array([])
        shape = 0
        reshape_size = 0

        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        for i in range(len(counters)):
            for j in range(counters[i]):
                img = cv.imread(f"{i}/frame{j}.jpg")
                if not reshape_size:
                    img_shape = img.shape
                    reshape_size = img_shape[0] * img_shape[1]
                # img = img.reshape(reshape_size)
                img_list = np.append(img_list, [img])
                class_list = np.append(class_list, i)
            shape += counters[i]

        # img_list = img_list.reshape(shape, reshape_size)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")
