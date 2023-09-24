import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from tensorflow import keras

# Путь к директории с изображениями
image_dir = 'C:/Users/IgorKragel/Videos/игорь/datasetnew/50x50/warning'

# Загрузите модель нейронной сети
pattern = keras.models.load_model("first_model_weights.h5")

# Список поддерживаемых форматов изображений (расширений файлов)
supported_image_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.gif']

# Открываем файл для записи результатов
with open("results.txt", "w") as f:
    # Проход по всем файлам в директории
    for filename in os.listdir(image_dir):
        # Полный путь к текущему файлу
        file_path = os.path.join(image_dir, filename)

        # Проверка, является ли файл изображением по расширению
        if any(filename.lower().endswith(ext) for ext in supported_image_formats):
            # Загрузка изображения
            img = tf.keras.utils.load_img(file_path, target_size=(50, 50))
            x = image.img_to_array(img)
            x = x.reshape(1, 50, 50, 3)

            # Предсказание
            prediction = pattern.predict(x)

            # Определение класса изображения
            class_index = np.argmax(prediction)

            # Вывод результата в консоль
            print(f"Файл: {filename}, Класс: {class_index}, Вероятности: {prediction}")

            # Запись результата в файл
            f.write(f"Файл: {filename}, Класс: {class_index}, Вероятности: {prediction}\n")
