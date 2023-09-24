from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import load_img
from keras.utils import img_to_array

import tensorflow
import scipy
import tensorflow as tf
from keras.preprocessing import image
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

pattern = keras.models.load_model("first_model_weights.h5")  # файл обученной нейронной сети

image_path = 'img.bmp'

imag = tf.keras.utils.load_img(image_path)



img = load_img(image_path, grayscale=False, color_mode='rgb', target_size=
(150, 150), interpolation='box')
x = img_to_array(img)
x = x.reshape(1, 150, 150, 3)
prediction = pattern.predict(x)
pattern.summary()
f = open("text.txt", "w")
print([i for i in prediction[0]])
class_index = np.argmax(prediction)
print(class_index)
print(prediction)
f.write(str([i for i in prediction[0]].index(max(prediction[0]))))