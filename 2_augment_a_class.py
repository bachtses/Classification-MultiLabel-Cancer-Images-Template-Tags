import os.path
import pandas as pd
import cv2
import csv

output_csv_name = "output_CSV.csv"
df = pd.read_csv("output_CSV.csv")
TRAINING_IMAGES_FOLDER = "images/"
TESTING_IMAGES_FOLDER = "testing/"

labels = df.columns.values.tolist()
labels.pop(0)
print("labels: ", labels)

CLASS_TO_BE_AUGMENTED = 3

new_rows_to_write = []
for item in os.listdir(TRAINING_IMAGES_FOLDER):

    idx = df.index[df['image name'].str.match(item)][0]
    vector = df.loc[idx].values.tolist()
    vector.pop(0)

    if vector[CLASS_TO_BE_AUGMENTED] == 1:
        new_name = 'augm' + item
        vector.insert(0, new_name)
        new_rows_to_write.append(vector)

        data = cv2.imread(os.path.join(TRAINING_IMAGES_FOLDER, item))
        data_RGB = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        flipHorizontal = cv2.flip(data_RGB, 1)

        cv2.imwrite(os.path.join(TRAINING_IMAGES_FOLDER, new_name), flipHorizontal)


with open(output_csv_name, 'a', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(new_rows_to_write)










