import logging
from telegram.ext import Updater, MessageHandler, Filters
from telegram.ext import CommandHandler
from telegram import ReplyKeyboardMarkup
from telegram import KeyboardButton
from telegram import ReplyKeyboardRemove
from random import randint

from keras.utils import load_img
from keras.utils import img_to_array

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

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
from PIL import Image
from keras.models import model_from_json
from keras import models

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG
)
logger = logging.getLogger(__name__)

TOKEN = '5814380638:AAFbJqHqcySEtAFWXlKv1ztp-j6gF1h0kTk'

new_model = keras.models.load_model("C:/Users/IgorKragel/PycharmProjects/Neuronka/first_model_weights.h5")

def start(update, context):
    update.message.reply_text("Привет!Я бот-классификатор, отправь мне фото кошки или собаки! ")

def Proverka(update,cotext):
    newFile = update.message.effective_attachment[0].get_file()
    newFile.download('img.bmp')
    x = newFile["file_path"]
    print(x)


    image_path = 'img.bmp'
    img = Image.open(image_path)
    # изменяем размер
    new_image = img.resize((150, 150))
    new_image.show()
    # сохранение картинки
    new_image.save('img.png')
    img_path = 'img.png'
    with tf.device('/CPU:0'):
        image = tf.keras.utils.load_img(image_path)

         # файл обученной нейронной сети
        img = load_img(img_path, grayscale=False, color_mode='rgb', target_size=
        (150, 150), interpolation='box')
        x = img_to_array(img)
        x = x.reshape(1, 150, 150, 3)
        prediction = new_model.predict(x)
        new_model.summary()
        f = open("text.txt", "w")
        print([i for i in prediction[0]])
        print(prediction)
    f.write(str([i for i in prediction[0]].index(max(prediction[0]))))
    f = open("text.txt", "r")
    opr = f.read()
    if (opr == '1'):
        update.message.reply_text("Это собака")
    if (opr == '0'):
        update.message.reply_text("Это кошка")

def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("Start", start))
    dp.add_handler(MessageHandler(Filters.photo, Proverka))


    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
