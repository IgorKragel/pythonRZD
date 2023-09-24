import os.path
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from tensorflow.python.estimator import keras
from keras.preprocessing import image
import numpy as np
from tensorflow import keras

class NeuralNetworkApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.frame_counter = 0  # Счетчик кадров для именования сохраненных файлов
        self.output_dir = "frames"  # Директория для сохранения кадров

        # Создаем директорию для сохранения кадров, если она не существует
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initUI(self):
        self.setWindowTitle("Neural Network Video Processing")
        self.setGeometry(100, 100, 800, 600)

        self.video_path = ""
        self.model = keras.models.load_model("first_model_weights.h5")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.result_label = QLabel(self)
        self.layout.addWidget(self.result_label)

        self.load_button = QPushButton("Выбрать видео")
        self.load_button.clicked.connect(self.loadVideo)
        self.layout.addWidget(self.load_button)

        self.process_button = QPushButton("Обработать")
        self.process_button.clicked.connect(self.processVideo)
        self.layout.addWidget(self.process_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.video_capture = None

    def loadVideo(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        if file_path:
            self.video_path = file_path

    def processVideo(self):
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.timer.start(30)  # Обновляем изображение каждые 30 миллисекунд

    def updateFrame(self):
        ret, frame = self.video_capture.read()

        if ret:
            frame = cv2.resize(frame, (1600, 900))  # Изменяем размер на 50x50 пикселей
            q_img = QImage(frame.data, 1600,900, QImage.Format_RGB888)
            # Обрабатываем изображение
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Переводим из BGR в RGB
            frame = cv2.resize(frame, (150, 150))  # Изменяем размер на 50x50 пикселей
            frame = np.expand_dims(frame, axis=0)  # Добавляем размерность батча
            """ self.saveFrame(frame)"""
            prediction = self.model.predict(frame)

            # Определяем класс изображения
            class_index = np.argmax(prediction)
            result_text = f"Класс: {class_index}, Вероятности: {prediction[0]}"
            self.result_label.setText(result_text)
            print(result_text)

            # Отображаем изображение в GUI
            h, w = frame.shape[:2]  # Получаем только высоту и ширину изображения
            ch = 3  # Задаем количество каналов (RGB)
            bytes_per_line = ch * w

            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
            QCoreApplication.processEvents()
        else:
            self.timer.stop()
            if self.video_capture.isOpened():
                self.video_capture.release()
            self.video_label.clear()
"""
    def saveFrame(self, frame):
        # Формируем имя файла для сохранения
        filename = os.path.join(self.output_dir, f"frame_{self.frame_counter:04d}.jpg")

        # Сохраняем изображение
        cv2.imwrite(filename, cv2.cvtColor(frame[0], cv2.COLOR_RGB2BGR))

        self.frame_counter += 1
        """
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeuralNetworkApp()
    window.show()
    sys.exit(app.exec_())
