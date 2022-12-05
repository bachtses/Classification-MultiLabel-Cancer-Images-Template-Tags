import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.metrics import AUC
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
import cv2
import os.path
import sys
import time
from numpy import array
from numpy import argmax
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflowjs as tfjs

TRAINING_IMAGES_FOLDER = "images/"
df = pd.read_csv("Data_Entry_Vectors.csv")

# print("\n")
# print("file: Data_Entry_Vectors.csv")
# print(df.head())
# print(df.tail())
# print(df.shape)
# print("\n")

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

batch = 32
epochs = 2

labels = df.columns.values.tolist()
labels.pop(0)
print("labels: ", labels)


print('*** data read ***')
n_classes = len(labels)

filenames_list = []
X = []
Y = []
for item in os.listdir(TRAINING_IMAGES_FOLDER):
    # print("file: ", item)
    img = cv2.imread(os.path.join(TRAINING_IMAGES_FOLDER, item))
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    # img = np.mean(img, axis=2)  # convert to 1-dim grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert to RGB
    # print(img)
    # plt.imshow(img)
    # plt.show()
    idx = df.index[df['image name'].str.match(item)][0]
    vector = []
    for i in labels:
        vector.append(df.loc[idx][i])
    img = img/255
    X.append(img)
    filenames_list.append(item)
    Y.append(vector)

print("X shape: ", np.shape(X))
print("Y shape: ", np.shape(Y))
print("filenames list shape: ", np.shape(filenames_list))

print("Number of samples in dataset: ", len(X))
print("\n")

X = np.array(X)
Y = np.array(Y)

start_time = time.time()

# create model
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
img_input = Input(shape=input_shape)
base_model = DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet', input_tensor=img_input, pooling="avg")  # for RGB
x = base_model.output
predictions = Dense(n_classes, activation="sigmoid", name="predictions")(x)
model = Model(inputs=img_input, outputs=predictions)
model.summary()

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])

# checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
# history = model.fit(X, Y, validation_split=0.1, batch_size=batch, epochs=epochs, verbose=1, callbacks=[checkpoint])
history = model.fit(X, Y, validation_split=0.2, batch_size=batch, epochs=epochs, verbose=1)


model.save('my_model.h5')
tfjs.converters.save_keras_model(model, 'models')

pd.DataFrame(model.history.history).plot()

elapsed_time = time.time() - start_time
elapsed_time = elapsed_time/60
elapsed_time = str(round(elapsed_time, 2))
print("Training duration : ", elapsed_time, "minutes")
