from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow
import scipy
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# training / validation part ....


from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices())
# my output was => ['/device:CPU:0']
# good output must be => ['/device:CPU:0', '/device:GPU:0']




image_width, image_height = 150, 150 # Указываем разрешение для изображений к единому формату

directory_data_train= 'C:/Users/IgorKragel/Videos/игорь/datasetnew/dataset150x150/training' #Указываем путь к обучающей выборке train_data_dir
directory_data_validation= 'C:/Users/IgorKragel/Videos/игорь/datasetnew/dataset150x150/validation'  #Указываем путь к проверочной выборке validation_data_dir

# Сразу устанавливаем необходимые параметры

train_sample = 5396
validation_sample = 1165
epochs = 5
lot_size = 30  #batch_size
if K.image_data_format() != 'channels_first':
     input_shape = (image_width, image_height, 3)
else:
     input_shape = (3, image_width, image_height)
pattern = Sequential() # Создание модели

# Первый слой нейросети

pattern.add(Conv2D(32, (3, 3), input_shape=input_shape))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

# Второй слой нейросети

pattern.add(Conv2D(32, (3, 3)))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

# Третий слой нейросети

pattern.add(Conv2D(64, (3, 3)))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

#Aктивация, свертка, объединение, исключение

pattern.add(Flatten())
pattern.add(Dense(64))
pattern.add(Activation('relu'))
pattern.add(Dropout(0.5))
pattern.add(Dense(2))# число классов
pattern.add(Activation('softmax'))

#Cкомпилируем модель с выбранными параметрами. Также укажем метрику для оценки.

pattern.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Задаём параметры аугментации

train_datagen = ImageDataGenerator(
    rescale=1. / 255, # коэффициент масштабирования
    shear_range=0.2, # Интенсивность сдвига
    zoom_range=0.2, # Диапазон случайного увеличения
    horizontal_flip=True) # Произвольный поворот по горизонтали

test_datagen = ImageDataGenerator(rescale=1. / 255)

#Предобработка обучающей выборки
train_processing = train_datagen.flow_from_directory(
    directory_data_train,
    target_size=(image_width, image_height), # Размер изображений
    batch_size=lot_size, #Размер пакетов данных
    class_mode='categorical') # Режим класса

#Предобработка тестовой выборки

validation_processing= test_datagen.flow_from_directory(
    directory_data_validation,
    target_size=(image_width, image_height),
    batch_size=lot_size,
    class_mode='categorical')

pattern.fit_generator(
    train_processing, # Помещаем обучающую выборку
    steps_per_epoch=train_sample // lot_size, #количество итераций пакета до того, как период обучения считается завершенным
    epochs=epochs, # Указываем количество эпох
    validation_data=validation_processing, # Помещаем проверочную выборку
    validation_steps=validation_sample  // lot_size) # Количество итерации, но на проверочном пакете данных


pattern.save('first_model_weights.h5') #Сохранение модели

