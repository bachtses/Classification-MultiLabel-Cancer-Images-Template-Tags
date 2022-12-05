import matplotlib.pyplot as plt
import numpy as np
import keras
import cv2
from keras.layers import Input
from keras.models import Model
import random
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf
import os.path
import pandas as pd
from tensorflow.keras.applications.densenet import DenseNet121

TESTING_IMAGES_FOLDER = "testing/"
WEIGHTS_FOLDER = "weights/"

# class_names = "Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax,Consolidation,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia"
class_names = "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural_Effusion"
n_classes = 5

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
img_input = Input(shape=input_shape)


print("** load model **")
base_model = DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet', input_tensor=img_input, pooling="avg")  # for RGB
x = base_model.output
predictions = Dense(n_classes, activation="sigmoid", name="predictions")(x)
model = Model(inputs=img_input, outputs=predictions)
# model.summary()


print("** load weights **")
model.load_weights(os.path.join(WEIGHTS_FOLDER, "best_weights_15559827687076797.h5"))
print(model.get_weights()[0])


# Data Read
for test_img_name in os.listdir(TESTING_IMAGES_FOLDER):
    img = cv2.imread(os.path.join(TESTING_IMAGES_FOLDER, test_img_name))
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    # img = np.mean(img, axis=2) convert to 1-dim gray
    # print(img)

    Xtest = img
    Xtest = Xtest.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # demonstrate prediction
    print("file: ", test_img_name)
    print(class_names)
    yhat = model.predict(Xtest, verbose=1)
    print(np.round(yhat, 4))
    max_pos = np.argmax(yhat, axis=1)
    class_list = class_names.split(",")
    print(max_pos)
    print(class_list[max_pos[0]])
    print("\n")
