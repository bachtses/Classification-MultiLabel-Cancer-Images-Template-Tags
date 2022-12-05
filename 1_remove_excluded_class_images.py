import os.path
import pandas as pd

# THIS CODE READS THE 'output_CSV.csv' AND THE MEDICAL IMAGES IN 'images' FOLDER
# AND REMOVE FROM TRAINING FOLDER ANY IMAGE THAT ITS NAME IS NOT IN THE CSV
# SO IT REMOVES THE IMAGES THAT BELONG TO EXCLUDED CLASSES
# i.e. no findings


output_csv_name = "Data_Entry_Vectors.csv"
df = pd.read_csv("Data_Entry_Vectors.csv")
IMAGES_FOLDER = "images/"
# IMAGES_FOLDER = "testing/"


print("REMOVING IMAGES THAT WERE CORRESPONDING TO EXCLUDED CLASSES ... ")
# DELETE IMAGES IN THE TRAINING FOLDER THAT ARE NOT IN OUTPUT CSV FILE
for item in os.listdir(IMAGES_FOLDER):
    # print("file: ", item)
    try:
        idx = df.index[df['image name'].str.match(item)][0]
    except:
        os.remove(os.path.join(IMAGES_FOLDER, item))

