import os.path
import pandas as pd

output_csv_name = "output_CSV.csv"
df = pd.read_csv("output_CSV.csv")
TRAINING_IMAGES_FOLDER = "images/"


labels = df.columns.values.tolist()
labels.pop(0)

CLASS_TO_BE_REDUCED = 2    # from which class
number = 1500  # how many case to reduce


deleted_count = 0
for item in os.listdir(TRAINING_IMAGES_FOLDER):
    # print("file: ", item)
    idx = df.index[df['image name'].str.match(item)][0]
    vector = df.loc[idx].values.tolist()
    vector.pop(0)
    if vector[CLASS_TO_BE_REDUCED] == 1 and deleted_count < number:
        os.remove(os.path.join(TRAINING_IMAGES_FOLDER, item))
        deleted_count += 1

