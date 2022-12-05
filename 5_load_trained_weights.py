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

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

labels = df.columns.values.tolist()
labels.pop(0)
print("labels: ", labels)


n_classes = len(labels)


X = []
Y = []
X = np.array(X)
Y = np.array(Y)
labels_array = []
predictions_array = []

# Data Read
for test_img_name in os.listdir(TESTING_IMAGES_FOLDER):
    img = cv2.imread(os.path.join(TESTING_IMAGES_FOLDER, test_img_name))
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.mean(img, axis=2) convert to 1-dim gray

    print("file: ", test_img_name)
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

    Xtest = img
    Xtest = Xtest.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


    # demonstrate prediction
    model = load_model('my_model.h5')
    # model.summary()


    yhat = model.predict(Xtest, verbose=0)

    print("labels: ", labels)
    results = []
    for i in range(len(yhat[0])):
        if yhat[0][i] >= 0.3:
            # yhat[0][i] = 1
            results.append(labels[i])
    print("Real label: ", real_label)
    print("AI model's prediction: ", np.round(yhat, 2))
    # print("prediction :", results)

    # MEDICAL REPORT
    print("\nMEDICAL REPORT RESULTS:")
    print("The X-Ray", test_img_name, "diagnosed with ", end='')
    flag = 0
    for i in results:
        if flag == 1:
            print(" and ", end='')
        flag = 1
        print(i, end='')
    print('.')
    print("\n\n\n")
    plt.imshow(img)
    plt.title(test_img_name)
    # plt.show()


