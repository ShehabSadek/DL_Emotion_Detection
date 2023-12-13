import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QApplication, QDesktopWidget, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QGridLayout,QProgressBar,
    QWidget, QTextEdit
)
from PyQt5.QtGui import QPixmap, QCursor, QIcon, QColor,QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2 as cv
from tensorflow.keras import models

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
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = models.load_model("emotion_detection_model.keras")
# model = models.load_model("emotion_detection_model_with_face_detection.keras")
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            faces = face_cascade.detectMultiScale(cv_img, minNeighbors=10)
            label=class_names[np.argmax(predict_emotion(cv_img,faces))]
            for (x, y, w, h) in faces:
                cv.rectangle(cv_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv.putText(cv_img,label,(x,y),cv.FONT_HERSHEY_DUPLEX,2,colors[label])
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion detection")
        self.setFixedHeight(600)
        self.setFixedWidth(900)
        self.setWindowIcon(QtGui.QIcon("./assets/logo.png"))
        MainWindow.instance = self
        pixmap = QPixmap("./assets/bg.jpg")

        background_label = QLabel(self)
        background_label.setPixmap(pixmap)
        background_label.resize(self.size())
        background_label.lower()
        self.emotions_preds=[]
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        #'angry','disgusted','fearful''happy''neutral''sad''surprised'
        self.emotion_labels = {
            'angry': QProgressBar(self),
            'disgusted': QProgressBar(self),
            'fearful': QProgressBar(self),
            'happy': QProgressBar(self),
            'neutral': QProgressBar(self),
            'sad': QProgressBar(self),
            'surprised': QProgressBar(self),
        }

        layout=QGridLayout()

        for row, (emotion, progress_bar) in enumerate(self.emotion_labels.items(), start=2):
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            label= QLabel(emotion)
            label.setStyleSheet("QLabel { color : black; }")
            layout.addWidget(label, row, 13, 2, 4)
            layout.addWidget(progress_bar, row, 15, 2, 4)

        layout.addWidget(self.image_label,0,0,12,12)
        # layout.addWidget(self.angry_label,1,13,1,4)
        # layout.addWidget(self.disgusted_label,2,13,1,4)
        # layout.addWidget(self.fearful_label,3,13,1,4)
        # layout.addWidget(self.happy_label,4,13,1,4)
        # layout.addWidget(self.neutral_label,5,13,1,4)
        # layout.addWidget(self.sad_label,6,13,1,4)
        # layout.addWidget(self.surprised_label,7,13,1,4)

        self.setLayout(layout)
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        self.update_bars()

    def update_bars(self):
        # if not np.any(self.emotions_preds):
        #     return
        for i,bar in enumerate(self.emotion_labels.values()):
            bar.setValue(int((self.emotions_preds.flatten()[i])*100))
            color = list(colors.values())[i]
            bar.setStyleSheet(f"QProgressBar {{ border: 1px solid black; }} QProgressBar::chunk {{ background-color: rgb({color[0]}, {color[1]}, {color[2]}); }}")
    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        label_size = self.image_label.size()
        scaled_image = convert_to_Qt_format.scaled(label_size, Qt.KeepAspectRatio)

        return QPixmap.fromImage(scaled_image)
    @classmethod
    def get_instance(cls):
        return cls.instance


def predict_emotion(img,faces):
    if not np.any(faces):
        main_window_instance = MainWindow.get_instance()
        emotion_prediction=np.zeros((1,7),np.int8)  
        main_window_instance.emotions_preds=emotion_prediction
        return emotion_prediction                     
    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]
        resized_face = cv.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape((1, 48, 48, 3))
        emotion_prediction = model.predict(reshaped_face)
        # emotion_label = class_names[np.argmax(emotion_prediction)]
        # print(class_names[np.argmax(emotion_prediction)])
        main_window_instance = MainWindow.get_instance()
        main_window_instance.emotions_preds=emotion_prediction
        return emotion_prediction
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

      
