import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

from util.models.main_model import MainModel
from matplotlib import pyplot as plt
import seaborn as sns

from util.models.part3_model import ImprovedMainModel

class_names = ['focused', 'happy', 'neutral', 'surprise']


def plot_confusion_matrix(cm, classnames):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classnames, yticklabels=classnames)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_model(mdl, loader):
    mdl.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = mdl(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    for i in random.sample(range(0, len(all_labels)), 10):
        print(f'Label: {class_names[all_labels[i]]}, Prediction: {class_names[all_preds[i]]}')

    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    return cm, accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1


def get_loaders(fold_number):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.ImageFolder('../data', transform=transform)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        if fold == fold_number:
            train_subsampler = Subset(dataset, train_ids)
            test_subsampler = Subset(dataset, test_ids)

            train_loader = DataLoader(train_subsampler, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_subsampler, batch_size=64, shuffle=False)

            return train_loader, test_loader


def k_fold_cross_validation(model_type, k=10):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.ImageFolder('../data', transform=transform)
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    results = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}')
        train_subsampler = Subset(dataset, train_ids)
        test_subsampler = Subset(dataset, test_ids)

        train_loader = DataLoader(train_subsampler, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subsampler, batch_size=64, shuffle=False)

        if model_type == 'part_3':
            model = ImprovedMainModel(4)
        else:
            model = MainModel(4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        for epoch in range(30):  # Number of epochs can be adjusted
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        metrics = evaluate_model(model, test_loader)
        results.append(metrics)

        # Optionally, save the model
        if not os.path.exists('./generated_models'):
            os.makedirs('./generated_models')
        torch.save(model.state_dict(), f'./generated_models/{model_type}_fold_{fold + 1}.pth')

        print(f'Fold {fold + 1} results:', metrics)

    print("k fold ran successfully")


def calculate_averages(results):
    accuracy = np.mean([r[1] for r in results])
    precision = np.mean([r[2] for r in results])
    recall = np.mean([r[3] for r in results])
    f1 = np.mean([r[4] for r in results])
    micro_precision = np.mean([r[5] for r in results])
    micro_recall = np.mean([r[6] for r in results])
    micro_f1 = np.mean([r[7] for r in results])

    return accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1


def run_model_analysis_for_folds(fold_number, model_type='main_model'):
    _, test_loader = get_loaders(fold_number)

    if model_type == 'part_3':
        model = ImprovedMainModel(4)
    else:
        model = MainModel(4)
    model.eval()
    model.load_state_dict(torch.load(f'generated_models/{model_type}_fold_{fold_number + 1}.pth'))

    # Evaluate the model
    cm, accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1 = evaluate_model(model, test_loader)

    print(f'Fold {fold_number + 1} results:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'Micro Precision: {micro_precision}')
    print(f'Micro Recall: {micro_recall}')
    print(f'Micro F1: {micro_f1}')
    print(cm)
    # print "fold #, accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1"
    print(f'{fold_number + 1}, {accuracy}, {precision}, {recall}, {f1}, {micro_precision}, {micro_recall}, {micro_f1}')

    plot_confusion_matrix(cm, class_names)


if __name__ == '__main__':
    # k_fold_cross_validation('part_3')
    for i in range(10):
        run_model_analysis_for_folds(i, 'part_3')
    pass
