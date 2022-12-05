import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd


TRAINING_IMAGES_FOLDER = "images/"
TESTING_IMAGES_FOLDER = "testing/"
df = pd.read_csv("Data_Entry_Vectors.csv")


print("\n")
print("Data_Entry_Vectors.csv  :")
print(df.head())
print(df.tail())
print(df.shape)
print("\n")

labels = df.columns.values.tolist()
labels.pop(0)

stats = [0]*len(labels)
train_images_number = 0
#  Data Read
for item in os.listdir(TRAINING_IMAGES_FOLDER):
    # print("file: ", item)
    idx = df.index[df['image name'].str.match(item)][0]
    vector = df.loc[idx].values.tolist()
    vector.pop(0)
    for i in range(len(vector)):
        if vector[i] == 1:
            stats[i] += 1
    train_images_number += 1

print("labels : ", labels)
print("stats : ", stats)
print("\n")
print("FOLDER: ", TRAINING_IMAGES_FOLDER)
print("SIZE: ", train_images_number)


test_images_number = 0
#  Data Read
for item in os.listdir(TESTING_IMAGES_FOLDER):
    # print("file: ", item)
    test_images_number += 1
print("\n")
print("FOLDER: ", TESTING_IMAGES_FOLDER)
print("SIZE: ", test_images_number)


fig = plt.figure(figsize=(11, 6))
plt.bar(labels, stats, color='blue', width=0.3)
plt.xlabel("Cases")
plt.ylabel("Findings")
plt.title("Dataset Balance Graph")
plt.xticks(rotation=60)
plt.show()
