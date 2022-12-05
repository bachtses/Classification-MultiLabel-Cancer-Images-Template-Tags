step 1) DOWNLOAD THE DATASET https://www.kaggle.com/nih-chest-xrays/data  (we need the images and the Data_Entry_2017.csv)
step 2) CREATE FOLDERS 'images' and 'testing' INSIDE THE PROJECT (and transfer the dowloaded images in 'images' folder)
step 3) RUN 0_multi_labels_CSV_creator.py  (it should export an 'output.csv' file that contains the mapping beetween images and deseases)
step 4) RUN 1_remove_excluded_class_images.py (it REMOVES from the 'images' folder the ones that belong to excluded classes)
step 5) RUN 3_statistics.py
step 6) Data balancing (data augmentation-reducing) (more .py files to be uploaded) SKIP THIS STEP FOR NOW
step 7) RUN 4_multi_label_classification_train.py (to start the training) 
