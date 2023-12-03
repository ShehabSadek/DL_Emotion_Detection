import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QDesktopWidget, QLabel, QPushButton, QVBoxLayout, QWidget, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer
import cv2
from tensorflow.keras import models
import numpy as np

class_names = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised',
}
colors = {
    'angry': (0, 0, 255),
    'disgusted': (0, 255, 0),
    'fearful': (255, 0, 255),
    'happy': (255, 255, 0),
    'neutral': (128, 128, 128),
    'sad': (0, 128, 255),
    'surprised': (0, 255, 255)
}
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
model = models.load_model("emotion_detection_model.keras")
model = models.load_model("emotion_detection_model_with_face_detection.keras")
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.capture = cv2.VideoCapture(0)
        self.timer_update_values = QTimer(self)
        self.timer_update_values.timeout.connect(self.update_values)
        self.timer_update_values.start(3000)
    def update_values(self):
        ret, frame = self.capture.read()
        emotion_percentages = self.predict_emotion_percentages(frame)
        self.update_display(emotion_percentages)
    def update_display(self, emotion_percentages):
            img = self.capture()
            emotion = self.predict_emotion(img)
            self.display_image_with_emotion(img, emotion)

            for i, (emotion, percentage) in enumerate(emotion_percentages.items()):
                label = QLabel(f"{emotion}: {percentage:.2%}", self)
                label.setAlignment(Qt.AlignTop)
                self.layout().addWidget(label, 2, i)
    def setup_ui(self):
        self.setWindowTitle("Emotion Detection")
        self.setWindowIcon(QIcon("path/to/icon.png"))

        layout = QGridLayout()

        self.video_label = QLabel(self)
        layout.addWidget(self.video_label, 0, 0, 1, 3)

        emotions = class_names.values()
        self.emotion_buttons = {}
        for i, emotion in enumerate(emotions):
            button = QPushButton(emotion, self)
            button.clicked.connect(lambda _, e=emotion: self.on_button_click(e))
            layout.addWidget(button, 1, i)

            self.emotion_buttons[emotion] = button

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(150)

    def update_video_feed(self):
        ret, frame = self.capture.read()
        if ret:
            emotion = self.predict_emotion(frame)
            self.display_image_with_emotion(frame, emotion)

    def predict_emotion(self, img):
        faces = face_cascade.detectMultiScale(img, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = img[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape((1, 48, 48, 3))
            emotion_prediction = model.predict(reshaped_face)
            emotion_label = class_names[np.argmax(emotion_prediction)]
            return emotion_label

    def display_image_with_emotion(self, img, emotion):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(img, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(img_rgb, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[emotion], 2)

        # mouths = mouth_cascade.detectMultiScale(img, minNeighbors=400)
        # for (x, y, w, h) in mouths:
        #     cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 255, 0), 2)

        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)
        self.video_label.setPixmap(pixmap)

        emotion_percentages = self.predict_emotion_percentages(img)

        for i, (emotion, percentage) in enumerate(emotion_percentages.items()):
            label = QLabel(f"{emotion}: {percentage:.2%}", self)
            label.setAlignment(Qt.AlignTop)
            self.layout().addWidget(label, 2, i)

    def predict_emotion_percentages(self, img):
        faces = face_cascade.detectMultiScale(img, minNeighbors=5)
        emotion_percentages = {emotion: 0.0 for emotion in class_names.values()}

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = img[y:y + h, x:x + w]
                resized_face = cv2.resize(face_roi, (48, 48))
                normalized_face = resized_face / 255.0
                reshaped_face = normalized_face.reshape((1, 48, 48, 3))
                emotion_prediction = model.predict(reshaped_face)
                emotion_probabilities = emotion_prediction[0]

                for i, emotion in enumerate(class_names.values()):
                    emotion_percentages[emotion] += emotion_probabilities[i]

            total_faces = len(faces)
            for emotion in emotion_percentages:
                emotion_percentages[emotion] /= total_faces

        return emotion_percentages



    def on_button_click(self, emotion):
        print(f"Button clicked: {emotion}")

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
