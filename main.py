import os
from datetime import time

import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
import seaborn as sns
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from util.models.main_model import MainModel
from util.models.variant_model_1 import VariantModel1
from util.models.variant_model_2 import VariantModel2

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


def validate(mdl, data_loader, crit):
    mdl.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # Inference mode, no need to compute gradients
        for inputs, labels in data_loader:
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = mdl(inputs)  # Forward pass
            loss = crit(outputs, labels)  # Compute loss

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')

    return epoch_loss


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

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    # Other metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    return cm, accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1


def get_loaders():
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.ImageFolder('../data', transform=transform)

    train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.15, random_state=42)
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.176, random_state=42) # 0.176 is approx 15% of 85%

    # Creating data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def run_model_creation(model_type):
    # Creating data loaders
    train_loader, val_loader, _ = get_loaders()

    max_epochs = 30
    if model_type == 'variant1':
        model = VariantModel1(4)
    elif model_type == 'variant2':
        model = VariantModel2(4)
    else:
        model = MainModel(4)

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer with a learning rate of 0.001
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}/{max_epochs}')
        train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        print()
        print('-' * 50)
        print()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Make sure the folder generated_models exists
            if not os.path.exists('./generated_models'):
                os.makedirs('./generated_models')

            torch.save(model.state_dict(), f'./generated_models/{model_type}.pth')


def run_model_analysis():
    # load model from file
    main_model = MainModel(4)
    variant_model1 = VariantModel1(4)
    variant_model2 = VariantModel2(4)
    main_model.load_state_dict(torch.load('./generated_models/main_model.pth'))
    variant_model1.load_state_dict(torch.load('./generated_models/variant1.pth'))
    variant_model2.load_state_dict(torch.load('./generated_models/variant2.pth'))

    # Get the test data loader
    _, _, test_loader = get_loaders()

    # Don't forget to set the models to evaluation mode
    main_model.eval()
    variant_model1.eval()
    variant_model2.eval()

    results_main = evaluate_model(main_model, test_loader)
    results_variant1 = evaluate_model(variant_model1, test_loader)
    results_variant2 = evaluate_model(variant_model2, test_loader)

    metrics_data = {
        'Model': ['Main Model', 'Variant 1', 'Variant 2'],
        'Accuracy': [results_main[1], results_variant1[1], results_variant2[1]],
        'Macro P': [results_main[2], results_variant1[2], results_variant2[2]],
        'Macro R': [results_main[3], results_variant1[3], results_variant2[3]],
        'Macro F': [results_main[4], results_variant1[4], results_variant2[4]],
        'Micro P': [results_main[5], results_variant1[5], results_variant2[5]],
        'Micro R': [results_main[6], results_variant1[6], results_variant2[6]],
        'Micro F': [results_main[7], results_variant1[7], results_variant2[7]]
    }

    pd.set_option('display.max_colwidth', None)
    df_metrics = pd.DataFrame(metrics_data)
    print(df_metrics)


if __name__ == '__main__':
    import time as t
    start_main = t.time()
    run_model_creation('main_model')
    end_main = t.time()
    print("*" * 50)
    print(f"Time taken to train main model: {end_main - start_main:.2f} seconds")
    print("*" * 50)

    start_main = t.time()
    run_model_creation('variant1')
    end_main = t.time()
    print("*" * 50)
    print(f"Time taken to train variant1 model: {end_main - start_main:.2f} seconds")
    print("*" * 50)

    start_main = t.time()
    run_model_creation('variant2')
    end_main = t.time()
    print("*" * 50)
    print(f"Time taken to train variant2 model: {end_main - start_main:.2f} seconds")
    print("*" * 50)

    run_model_analysis()
