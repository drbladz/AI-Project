import os
import json

label_data_path = "../label_data"
labels_path = "../labels"


def label(label_for, labels):
    image_map = {}
    for i, lbl in enumerate(labels):
        # load images from the lbl folder
        images = os.listdir(os.path.join(label_data_path, lbl))
        for image in images:
            if image == ".DS_Store":
                continue
            image_map[image] = lbl

    # save all labels to a json file
    with open(f'{labels_path}/{label_for}.json', 'w') as f:
        json.dump(image_map, f)


# label("gender", ["man", "woman", "other"])
# label("age", ["young", "old", "middle-aged"])
