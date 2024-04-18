import json
import os
import random

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.autograd import Variable


from util.models.main_model import MainModel

class_names = ['focused', 'happy', 'neutral', 'surprise']


def train(mdl, data_loader, optzer, crit):
    mdl.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in data_loader:
        inputs, labels = Variable(inputs), Variable(labels)

        optzer.zero_grad()  # Zero the parameter gradients
        outputs = mdl(inputs)  # Forward pass
        loss = crit(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optzer.step()  # Optimize

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    print(f'Training Loss: {epoch_loss:.4f}')


def load_attribute_labels(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def filter_dataset_by_attribute(dataset, attribute_labels):
    subset_indices = []
    for idx in range(len(dataset.samples)):
        img_path, _ = dataset.samples[idx]
        file_name = img_path.split('/')[-1]
        if file_name in attribute_labels:
            subset_indices.append(idx)
    return torch.utils.data.Subset(dataset, subset_indices)


def create_group_loaders(dataset, attribute_json):
    attribute_labels = load_attribute_labels(attribute_json)
    group_datasets = {
        label: filter_dataset_by_attribute(dataset, {k: v for k, v in attribute_labels.items() if v == label}) for label
        in set(attribute_labels.values())}
    return {label: DataLoader(dataset=group_datasets[label], batch_size=64, shuffle=True) for label in group_datasets}


def plot_confusion_matrix(cm, classnames):
    """
    Plots a confusion matrix using Seaborn's heatmap.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classnames, yticklabels=classnames)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def train_and_evaluate_all_levels(biased_datasets, test_loader):
    results = {}
    for level, biased_dataset in biased_datasets.items():
        print(f"Training on {level} dataset...")
        model = MainModel(4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Assuming existence of DataLoader
        biased_loader = DataLoader(dataset=biased_dataset, batch_size=64, shuffle=True)
        max_epochs = 30
        for epoch in range(max_epochs):
            print(f'Epoch {epoch + 1}/{max_epochs}')
            train(model, biased_loader, optimizer, criterion)
            print()
            print('-' * 50)
            print()

        if not os.path.exists('./util/generated_models'):
            os.makedirs('./util/generated_models')
        torch.save(model.state_dict(), f'./util/generated_models/main_model_{level}.pth')

        print(f"Evaluating on {level} dataset...")
        test_metrics = evaluate_model(model, test_loader)
        results[level] = test_metrics
    return results


def evaluate_model(mdl, tst_loader):
    mdl.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tst_loader:
            outputs = mdl(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print the label and image for the first 10 samples
    for i in range(10):
        print(f'Label: {class_names[all_labels[i]]}, Prediction: {class_names[all_preds[i]]}')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # plot_confusion_matrix(cm, class_names)

    # Other metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    return cm, accuracy, precision, recall, f1


def evaluate_groups(model, groups):
    results = {}
    for group_name, loader in groups.items():
        print(f"Evaluating {group_name} group...")
        _, accuracy, precision, recall, f1 = evaluate_model(model, loader)
        results[group_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    return results


def create_biased_split(dataset, attribute_labels, bias_info):
    group_indices = {label: [] for label in set(attribute_labels.values())}
    for idx, (img_path, _) in enumerate(dataset.samples):
        file_name = img_path.split('/')[-1]
        if file_name in attribute_labels:
            group = attribute_labels[file_name]
            group_indices[group].append(idx)

    biased_indices = []
    for group, indices in group_indices.items():
        keep_count = int(len(indices) * bias_info[group])
        biased_indices.extend(random.sample(indices, keep_count))

    return torch.utils.data.Subset(dataset, biased_indices)


def create_biased_datasets(dataset, attribute_json):
    attribute_labels = load_attribute_labels(attribute_json)
    bias_levels = {
        'Level 1': {'young': 0.43333, 'middle-aged': 0.28333, 'old': 0.28333},
        'Level 2': {'young': 0.53333, 'middle-aged': 0.23333, 'old': 0.23333},
        'Level 3': {'young': 0.73333, 'middle-aged': 0.13333, 'old': 0.13333}
    }
    return {level: create_biased_split(dataset, attribute_labels, biases) for level, biases in bias_levels.items()}


def evaluate_biased_models(with_train=False):
    model = MainModel(4)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.ImageFolder('data', transform=transform)
    train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.15, random_state=42)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    biased_datasets = create_biased_datasets(dataset, 'labels/age.json')
    if with_train:
        results = train_and_evaluate_all_levels(biased_datasets, test_loader)
        print("Bias Analysis Results:")
        for level, metrics in results.items():
            print(f"{level} - Accuracy: {metrics[1]}, Precision: {metrics[2]}, Recall: {metrics[3]}, F1-Score: {metrics[4]}")
    else:
        for level, biased_dataset in biased_datasets.items():
            print(f"Evaluating {level} dataset...")
            model.load_state_dict(torch.load(f'util/generated_models/main_model_{level}.pth'))
            model.eval()
            results = evaluate_groups(model, create_group_loaders(dataset, 'labels/age.json'))
            print(results)




def run_bias_analysis(level=None):
    # Load your main model
    main_model = MainModel(4)
    model_path = 'util/generated_models/main_model.pth' if level is None else f'util/generated_models/main_model_Level {level}.pth'

    main_model.load_state_dict(torch.load(model_path))
    main_model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.ImageFolder('data', transform=transform)
    age_groups = create_group_loaders(dataset, 'labels/age.json')
    gender_groups = create_group_loaders(dataset, 'labels/gender.json')

    age_results = evaluate_groups(main_model, age_groups)
    gender_results = evaluate_groups(main_model, gender_groups)

    # Populate Table 2 based on age_results and gender_results
    print("Age Results:")
    print(age_results)
    print("-" * 50)
    print("Gender Results:")
    print(gender_results)


if __name__ == '__main__':
    evaluate_biased_models()
