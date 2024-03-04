import os
from PIL import Image, ImageEnhance
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score

import matplotlib.pyplot as plt

training_path = "data/train"
testing_path = "data/test"


# A function "setup images" that only keeps 500 images from each folder inside the "data/train" folder
# and 100 images from each folder inside the "data/test" folder.
# The function should take no arguments and return nothing.
def setup_images():
    for folder in os.listdir(training_path):
        if folder == ".DS_Store":
            continue
        images = os.listdir(os.path.join(training_path, folder))
        for i in range(500, len(images)):
            os.remove(os.path.join(training_path, folder, images[i]))
    for folder in os.listdir(testing_path):
        if folder == ".DS_Store":
            continue
        images = os.listdir(os.path.join(testing_path, folder))
        # remove the images after 100 (ignore .DS_Store)
        # check if there is a .DS_Store file in the folder
        for i in range(100, len(images)):
            os.remove(os.path.join(testing_path, folder, images[i]))


# clean the data (standardize brightness, size, etc.)
def clean_data():
    all_folders = os.listdir(training_path)
    all_folders.extend(os.listdir(testing_path))
    for folder in all_folders:
        if folder == ".DS_Store":
            continue
        images = os.listdir(os.path.join(training_path, folder))
        for image in images:
            if image == ".DS_Store":
                continue
            # clean the image
            image_path = os.path.join(training_path, folder, image)
            image = Image.open(image_path)
            # resize the image
            image = image.resize((48, 48))
            # save the image
            image.save(image_path)


# Call the function
clean_data()
