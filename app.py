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
    DEFAULT_PREDICTED_TEXT = "Here the predicted class will be displayed"
    AUTO_PREDICT_ON_TEXT = "Auto Prediction: ON"
    AUTO_PREDICT_OFF_TEXT = "Auto Prediction: OFF"

    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window_title = window_title

        self.camera = camera.Camera()
        self.model = model.Model()

        self.auto_predict = False

        self.classes = {}
        self.btns_for_classes = {}

        self.active_class = None
        self.start_time = None
        self.end_time = None
        self.save_frame_delay = 0.1
        self.save_frame_duration = 11

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

        self.btn_toggle_auto_predict = tk.Button(
            self.window, text=self.AUTO_PREDICT_OFF_TEXT, width=50, command=self.auto_predict_toggle
        )
        self.btn_toggle_auto_predict.pack(anchor=tk.CENTER, expand=True)

        self.classes_number = self.get_classes_number()
        self.counters = [0] * self.classes_number

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

        self.btn_train = tk.Button(
            self.window, text="Train Model", width=50, command=lambda: self.model.train(self.counters)
        )
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predcit", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text=self.DEFAULT_PREDICTED_TEXT)
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def get_classes_number(self):
        classes_number = simpledialog.askstring("Number of Classes", "Enter the number of classes:", parent=self.window)

        if not classes_number.isnumeric():
            return self.get_classes_number()
        return int(classes_number)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict
        if self.auto_predict:
            self.btn_toggle_auto_predict.config(text=self.AUTO_PREDICT_ON_TEXT)
        else:
            self.btn_toggle_auto_predict.config(text=self.AUTO_PREDICT_OFF_TEXT)

    def activate_saving_for_class(self, index):
        if not os.path.isdir(str(index)):
            os.mkdir(str(index))

        self.disable_class_buttons()
        self.active_class = index
        self.start_time = time.time() + self.save_frame_delay
        self.end_time = time.time() + self.save_frame_duration

    def update_frame(self):
        if self.auto_predict:
            self.predict()

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            current_time = time.time()

            if self.active_class is not None and current_time > self.start_time and current_time < self.end_time:
                self.start_time = current_time + self.save_frame_delay
                self.save_frame(frame)
            elif self.active_class is not None and current_time > self.end_time:
                self.active_class = None
                self.enable_class_buttons()

        self.window.after(self.update_frame_delay, self.update_frame)

    def save_frame(self, frame):
        if self.active_class is not None:
            filepath = f"{self.active_class}/frame{self.counters[self.active_class]}.jpg"
            cv.imwrite(filepath, cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
            img = PIL.Image.open(filepath)
            img.thumbnail(self.model.TRAIN_FRAME_SIZE, PIL.Image.Resampling.LANCZOS)
            img.save(filepath)

            self.counters[self.active_class] += 1

    def disable_class_buttons(self):
        for i in range(self.classes_number):
            self.btns_for_classes[i].config(state=tk.DISABLED)

    def enable_class_buttons(self):
        for i in range(self.classes_number):
            self.btns_for_classes[i].config(state=tk.NORMAL)

    def predict(self):
        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)
        index = np.argmax(prediction)
        self.class_label.config(text=self.classes[index])

    def reset(self):
        self.auto_predict = False
        self.btn_toggle_auto_predict.config(text=self.AUTO_PREDICT_OFF_TEXT)
        self.remove_training_data()
        self.remove_test_frame()
        self.counters = [0] * self.classes_number
        self.model = model.Model()
        self.class_label.config(text=self.DEFAULT_PREDICTED_TEXT)

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
