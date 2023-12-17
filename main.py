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
import mediapipe as mp

class_names = {
    0: 'Angry',
    1: 'Disgusted',
    2: 'Fearful',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprised',
}
genders = {
    0: 'Man',
    1: 'Woman',
}
ages = {
    0: "18-20",
    1: "21-30",
    2: "31-40",
    3: "41-50",
    4: " 51-60"
}
colors = {
    'Angry': (0, 0, 255),
    'Disgusted': (0, 255, 0),
    'Fearful': (255, 0, 255),
    'Happy': (255, 255, 0),
    'Neutral': (128, 128, 128),
    'Sad': (0, 128, 255),
    'Surprised': (0, 255, 255)
}
# face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# hog_face_detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


mp_face_detection = mp.solutions.face_detection

# model = models.load_model("emotion_detection_model.keras")
model = models.load_model("emotion_detection_model_with_face_detection.keras")
gender_model = models.load_model("gender_detection_model.keras")
age_model = models.load_model("age_detection_model.keras")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
        self.gender_detection = 'man'
    def run(self):
        cap = cv.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            faces = self.detect_faces(cv_img)

            if faces is not None:
                label1, label2, label3= predict_emotion(cv_img, faces)
                label_e = class_names[np.argmax(label1)]
                label_g = genders[np.argmax(label2)]
                label_a = ages[np.argmax(label3)]
                print(label_e, label_g,label_a)
                for (x, y, w, h) in faces:
                    cv.rectangle(cv_img, (x, y), (x + w, y + h),color=colors[label_e],thickness=2)
                    cv.putText(cv_img, label_e, (x, y - 120), cv.FONT_HERSHEY_COMPLEX, fontScale=2, thickness=2,
                            color=(0,255,0))
                    cv.putText(cv_img, label_g, (x, y - 60), cv.FONT_HERSHEY_COMPLEX, fontScale=2, thickness=2,
                            color=(0,255,0))
                    cv.putText(cv_img, label_a, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, fontScale=2, thickness=2,
                            color=(0,255,0))

            if ret:
                self.change_pixmap_signal.emit(cv_img)

        cap.release()


    def detect_faces(self, image):
        try:
            if image is None or len(image) == 0:
                print("Error: Empty or invalid image in detect_faces")
                return None

            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            if rgb_image is None or len(rgb_image) == 0:
                print("Error: Empty or invalid RGB image in detect_faces")
                return None

            result = self.face_detection.process(rgb_image)

            faces = []
            if result.detections:
                for detection in result.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = rgb_image.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    faces.append(bbox)

            return faces

        except Exception as e:
            print(f"Error in detect_faces: {e}")
            return None




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
            'Angry': QProgressBar(self),
            'Disgusted': QProgressBar(self),
            'Fearful': QProgressBar(self),
            'Happy': QProgressBar(self),
            'Neutral': QProgressBar(self),
            'Sad': QProgressBar(self),
            'Surprised': QProgressBar(self),
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
        gender_prediction=np.zeros((1,2),np.int8)
        age_prediction=np.zeros((1,5),np.int8)

        main_window_instance.emotions_preds=emotion_prediction
        return emotion_prediction,gender_prediction,age_prediction          
    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]
        resized_face = cv.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape((1, 48, 48, 3))
        emotion_prediction = model.predict(reshaped_face)
        gender_prediction = gender_model.predict(reshaped_face)
        age_prediction = age_model.predict(reshaped_face)
        # emotion_label = class_names[np.argmax(emotion_prediction)]
        # print(class_names[np.argmax(emotion_prediction)])
        main_window_instance = MainWindow.get_instance()
        main_window_instance.emotions_preds=emotion_prediction
        return emotion_prediction,gender_prediction,age_prediction
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

      
