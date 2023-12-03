import sys
import os
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QApplication, QDesktopWidget, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QGridLayout,
    QWidget, QTextEdit
)
from PyQt5.QtGui import QPixmap, QCursor, QIcon, QColor,QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2 as cv
class_names = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised',
}
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion detection")
        self.setFixedHeight(600)
        self.setFixedWidth(900)
        self.setWindowIcon("")

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        #'angry','disgusted','fearful''happy''neutral''sad''surprised'
        self.angry_label = QLabel('angry')
        self.disgusted_label = QLabel('disgusted')
        self.fearful_label = QLabel('fearful')
        self.happy_label = QLabel('happy')
        self.neutral_label = QLabel('neutral')
        self.sad_label = QLabel('sad')
        self.surprised_label = QLabel('surprised')

        layout=QGridLayout()
        layout.addWidget(self.angry_label,0,0,12,12)
        layout.addWidget(self.disgusted_label,2,13,1,4)
        layout.addWidget(self.fearful_label,3,13,1,4)
        layout.addWidget(self.happy_label,4,13,1,4)
        layout.addWidget(self.neutral_label,5,13,1,4)
        layout.addWidget(self.sad_label,6,13,1,4)
        layout.addWidget(self.surprised_label,7,13,1,4)

        self.setLayout(layout)
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())