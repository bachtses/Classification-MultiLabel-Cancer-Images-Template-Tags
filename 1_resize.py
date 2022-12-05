import cv2
import os.path
import pandas as pd


# RESIZE THE IMAGES IN THE TRAINING FOLDER


TRAINING_IMAGES_FOLDER = "images/"
TESTING_IMAGES_FOLDER = "testing/"

IMG_WIDTH = 128
IMG_HEIGHT = 128

print("RESIZING ... ")

for item in os.listdir(TRAINING_IMAGES_FOLDER):
    # print("file: ", item)
    img = cv2.imread(os.path.join(TRAINING_IMAGES_FOLDER, item))
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite(os.path.join(TRAINING_IMAGES_FOLDER, item), img)


for item in os.listdir(TESTING_IMAGES_FOLDER):
    # print("file: ", item)
    img = cv2.imread(os.path.join(TESTING_IMAGES_FOLDER, item))
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite(os.path.join(TESTING_IMAGES_FOLDER, item), img)


