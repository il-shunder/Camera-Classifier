import os
import time
import tkinter as tk
from tkinter import simpledialog

import cv2 as cv
import numpy as np
import PIL.Image
import PIL.ImageTk

import camera
import model


class App:
    SAVE_FRAME_DELAY = 0.1
    SAVE_FRAME_DURATION = 11
    DEFAULT_PREDICTED_TEXT = "Here the predicted class will be displayed"
    AUTO_PREDICT_ON_TEXT = "Auto Prediction: ON"
    AUTO_PREDICT_OFF_TEXT = "Auto Prediction: OFF"
    CLASSES_LABEL_1_TEXT = "Click on the below buttons to provide examples of each class."
    CLASSES_LABEL_2_TEXT = f"For the next {SAVE_FRAME_DURATION} seconds after clicking the button, the script will automatically save images every {SAVE_FRAME_DELAY} seconds."
    TRAIN_LABEL_TEXT = "Once you provide examples for all class, you will be able to train the model."
    PREDICT_LABEL_TEXT = "Once you train the model, you can predict and/or auto-predict classes"

    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window_title = window_title

        self.camera = camera.Camera()
        self.model = model.Model()

        self.train_allowed = False
        self.predict_allowed = False
        self.auto_predict = False

        self.classes = {}
        self.btns_for_classes = {}

        self.active_class = None
        self.start_time = None
        self.end_time = None

        self.init_gui()

        self.update_frame_delay = 15
        self.update_frame()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def __del__(self):
        self.remove_training_data()
        self.remove_test_frame()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.FRAME_WIDTH, height=self.camera.FRAME_HEIGHT)
        self.canvas.pack()

        self.classes_number = self.get_classes_number()
        self.counters = [0] * self.classes_number

        self.classes_label_1 = tk.Label(self.window, text=self.CLASSES_LABEL_1_TEXT)
        self.classes_label_1.config(font=("Arial", 15))
        self.classes_label_1.pack(anchor=tk.CENTER, expand=True)

        self.classes_label_2 = tk.Label(self.window, text=self.CLASSES_LABEL_2_TEXT)
        self.classes_label_2.config(font=("Arial", 15))
        self.classes_label_2.pack(anchor=tk.CENTER, expand=True)

        for i in range(self.classes_number):
            self.classes[i] = simpledialog.askstring(
                "Classname", f"Enter the name of the {i + 1} class:", parent=self.window
            )

        for i in range(self.classes_number):
            self.btns_for_classes[i] = tk.Button(
                self.window,
                text=self.classes[i],
                width=50,
                command=lambda index=i: self.activate_saving_for_class(index),
            )
            self.btns_for_classes[i].pack(anchor=tk.CENTER, expand=True)

        self.train_label = tk.Label(self.window, text=self.TRAIN_LABEL_TEXT)
        self.train_label.config(font=("Arial", 15))
        self.train_label.pack(anchor=tk.CENTER, expand=True, pady=(15, 0))

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=self.train)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)
        self.disable_element(self.btn_train)

        self.predict_label = tk.Label(self.window, text=self.PREDICT_LABEL_TEXT)
        self.predict_label.config(font=("Arial", 15))
        self.predict_label.pack(anchor=tk.CENTER, expand=True, pady=(15, 0))

        self.btn_toggle_auto_predict = tk.Button(
            self.window, text=self.AUTO_PREDICT_OFF_TEXT, width=50, command=self.auto_predict_toggle
        )
        self.btn_toggle_auto_predict.pack(anchor=tk.CENTER, expand=True)
        self.disable_element(self.btn_toggle_auto_predict)

        self.btn_predict = tk.Button(self.window, text="Predcit", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)
        self.disable_element(self.btn_predict)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True, pady=(15, 0))

        self.class_label = tk.Label(self.window, text=self.DEFAULT_PREDICTED_TEXT)
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def get_classes_number(self):
        classes_number = simpledialog.askstring("Number of Classes", "Enter the number of classes:", parent=self.window)

        if not classes_number.isnumeric():
            return self.get_classes_number()
        return int(classes_number)

    def activate_saving_for_class(self, index):
        if not os.path.isdir(str(index)):
            os.mkdir(str(index))

        self.disable_all_buttons()
        self.active_class = index
        self.start_time = time.time() + self.SAVE_FRAME_DELAY
        self.end_time = time.time() + self.SAVE_FRAME_DURATION

    def update_frame(self):
        if self.auto_predict:
            self.predict()

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            current_time = time.time()

            if self.active_class is not None and current_time > self.start_time and current_time < self.end_time:
                self.start_time = current_time + self.SAVE_FRAME_DELAY
                self.save_frame(frame)
            elif self.active_class is not None and current_time > self.end_time:
                self.active_class = None
                for i in range(self.classes_number):
                    self.enable_element(self.btns_for_classes[i])
                if not self.train_allowed:
                    self.train_allowed = all(x > 0 for x in self.counters)
                if self.train_allowed:
                    self.enable_element(self.btn_train)
                if self.predict_allowed:
                    self.enable_element(self.btn_toggle_auto_predict)
                    self.enable_element(self.btn_predict)
                self.enable_element(self.btn_reset)
        self.window.after(self.update_frame_delay, self.update_frame)

    def save_frame(self, frame):
        if self.active_class is not None:
            filepath = f"{self.active_class}/frame{self.counters[self.active_class]}.jpg"
            cv.imwrite(filepath, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            img = PIL.Image.open(filepath)
            img.thumbnail(self.model.TRAIN_FRAME_SIZE, PIL.Image.Resampling.LANCZOS)
            img.save(filepath)
            self.counters[self.active_class] += 1

    def train(self):
        self.disable_element(self.btn_train)
        model_trained = self.model.train(self.counters)
        if model_trained:
            self.predict_allowed = True
            self.enable_element(self.btn_toggle_auto_predict)
            self.enable_element(self.btn_predict)
        self.enable_element(self.btn_train)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

        if self.auto_predict:
            self.disable_all_buttons()
            self.enable_element(self.btn_toggle_auto_predict)
            self.btn_toggle_auto_predict.config(text=self.AUTO_PREDICT_ON_TEXT)
        else:
            self.enable_all_buttons()
            self.btn_toggle_auto_predict.config(text=self.AUTO_PREDICT_OFF_TEXT)

    def predict(self):
        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)
        index = np.argmax(prediction)
        self.class_label.config(text=self.classes[index])

    def reset(self):
        self.train_allowed = False
        self.predict_allowed = False
        self.auto_predict = False

        self.btn_toggle_auto_predict.config(text=self.AUTO_PREDICT_OFF_TEXT)
        self.counters = [0] * self.classes_number
        self.model = model.Model()
        self.class_label.config(text=self.DEFAULT_PREDICTED_TEXT)

        self.disable_element(self.btn_train)
        self.disable_element(self.btn_toggle_auto_predict)
        self.disable_element(self.btn_predict)
        self.remove_training_data()
        self.remove_test_frame()

    def disable_all_buttons(self):
        for i in range(self.classes_number):
            self.disable_element(self.btns_for_classes[i])
        self.disable_element(self.btn_train)
        self.disable_element(self.btn_toggle_auto_predict)
        self.disable_element(self.btn_predict)
        self.disable_element(self.btn_reset)

    def disable_element(self, element):
        element.config(state=tk.DISABLED)

    def enable_all_buttons(self):
        for i in range(self.classes_number):
            self.enable_element(self.btns_for_classes[i])
        self.enable_element(self.btn_train)
        self.enable_element(self.btn_toggle_auto_predict)
        self.enable_element(self.btn_predict)
        self.enable_element(self.btn_reset)

    def enable_element(self, element):
        element.config(state=tk.NORMAL)

    def remove_training_data(self):
        for dir in range(len(self.counters)):
            dir = str(dir)
            if os.path.isdir(dir):
                for file in os.listdir(dir):
                    file_path = os.path.join(dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(dir)

    def remove_test_frame(self):
        if os.path.isfile(self.model.TEST_FRAME_PATH):
            os.remove(self.model.TEST_FRAME_PATH)
