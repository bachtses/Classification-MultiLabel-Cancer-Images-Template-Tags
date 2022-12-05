import matplotlib.pyplot as plt
import numpy as np
import keras
import cv2
import random
from keras.models import load_model
import os.path
import pandas as pd
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint

df = pd.read_csv("output_CSV.csv")
TESTING_IMAGES_FOLDER = "testing/"

IMG_HEIGHT = 200
IMG_WIDTH = 200
IMG_CHANNELS = 1

labels = df.columns.values.tolist()
labels.pop(0)
print("labels: ", labels)


n_classes = len(labels)

X_test = []
Y_test = []

labels_array = []
predictions_array = []

# Data Read
for test_img_name in os.listdir(TESTING_IMAGES_FOLDER):
    img = cv2.imread(os.path.join(TESTING_IMAGES_FOLDER, test_img_name))
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
    img = np.mean(img, axis=2)  # convert to 1-dim grayscale
    # print("file: ", test_img_name)
    row = 0
    #  Data Read
    for k in range(len(df)):
        # image_name = df.iloc[row]['image name']
        if df.iloc[row]['image name'] == test_img_name:
            row_index = df[df['image name'] == test_img_name].index[0]
            real_label = (df.loc[[row_index]]).values.tolist()
            real_label = real_label[0]
            real_label.pop(0)

        row += 1

    X_test.append(img)
    Y_test.append(real_label)


print("labels: ", labels)

print("X shape: ", np.shape(X_test))
print("Y shape: ", np.shape(Y_test))

print("Number of samples in", TESTING_IMAGES_FOLDER, ":", len(X_test))
print("\n")

X_test = np.array(X_test)
Y_test = np.array(Y_test)

model = load_model('my_model.h5')

test_accu = model.evaluate(X_test, Y_test)
print('The testing accuracy is :', test_accu[1] * 100, '%')

