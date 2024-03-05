import os
import random

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

training_path = "data/train"
testing_path = "data/test"
traits = ['focused', 'happy', 'neutral', 'surprise']


# A function "setup images" that only keeps 500 images from each folder inside the "data/train" folder
# and 100 images from each folder inside the "data/test" folder.
# The function should take no arguments and return nothing.
def setup_images():
    for folder in os.listdir(training_path):
        if folder == ".DS_Store":
            continue
        images = os.listdir(os.path.join(training_path, folder))
        if ".DS_Store" in images:
            os.remove(os.path.join(training_path, folder, ".DS_Store"))
        for i in range(500, len(images)):
            os.remove(os.path.join(training_path, folder, images[i]))
    for folder in os.listdir(testing_path):
        if folder == ".DS_Store":
            continue
        images = os.listdir(os.path.join(testing_path, folder))
        if ".DS_Store" in images:
            os.remove(os.path.join(testing_path, folder, ".DS_Store"))
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
            # # resize the image
            image = image.resize((48, 48))
            # save the image
            image.save(image_path, quality=95)


# A function "load_data" that loads the images from the "data/train" and "data/test" folders
# and returns the images as a list of numpy arrays and the labels as a list of integers.
def load_data(is_training=True):
    if is_training:
        path = training_path
    else:
        path = testing_path
    images = []
    labels = []
    for folder in os.listdir(path):
        if folder == ".DS_Store":
            continue
        for image in os.listdir(os.path.join(path, folder)):
            if image == ".DS_Store":
                continue
            image_path = os.path.join(path, folder, image)
            image = Image.open(image_path)
            image = np.array(image)
            images.append(image)
            labels.append(folder)
    return images, labels


def load_images_from_folder(folder, num_images=25):
    """
    Load a specific number of images randomly from a folder.

    :param folder: Path to the folder from which to load images.
    :param num_images: Number of images to load.
    :return: A list of PIL Image objects.
    """
    images = []
    file_names = os.listdir(folder)
    selected_files = random.sample(file_names, min(len(file_names), num_images))
    for filename in selected_files:
        if filename == ".DS_Store":
            continue
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def plot_class_distribution(classes, title):
    """
    Plot a bar graph showing the number of images in each class. This helps in understanding if any class is overrepresented or underrepresented.

    :param classes: List of class labels.
    :param title: Title of the plot.
    """
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(title)
    unique, counts = np.unique(classes, return_counts=True)
    axs.bar(unique, counts, color='k', alpha=0.7)
    axs.set_xlabel('Class')
    axs.set_ylabel('Number of Images')
    plt.show()


def plot_images(images, title, rows=5, cols=5):
    """
    Plot a list of images in a grid.

    :param images: List of PIL Image objects.
    :param title: Title of the plot.
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    """
    fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
    fig.suptitle(title)
    for i in range(rows * cols):
        ax = axs[i // cols, i % cols]
        ax.imshow(images[i], cmap="gray") if i < len(images) else ax.axis('off')
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_histogram(images, title):
    """
    plot a histogram showing the distribution of pixel intensities. This can provide insights into variations in
    lighting conditions among images.

    :param images: List of PIL Image objects.
    :param title: Title of the plot.
    """
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(title)
    axs.hist(np.array(images).ravel(), bins=256, range=(0, 256), density=True, color='k', alpha=0.7)
    axs.set_xlabel('Pixel Intensity')
    axs.set_ylabel('Frequency')
    plt.show()


train_images, train_labels = load_data()
test_images, test_labels = load_data(False)

print(f"Number of training images: {len(train_images)}")
print(f"Number of testing images: {len(test_images)}")

plot_class_distribution(train_labels, "Class Distribution in Training Data")
plot_class_distribution(test_labels, "Class Distribution in Testing Data")

for trait in traits:
    folder_path = os.path.join(training_path, trait)
    imgs = load_images_from_folder(folder_path)
    plot_images(imgs, f"Images for {trait}")
    plot_histogram(imgs, f"Histogram of pixel values for {trait}")
