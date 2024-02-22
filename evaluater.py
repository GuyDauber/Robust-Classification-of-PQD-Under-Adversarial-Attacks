import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_classification_performance(model, data_loader, device):
    """
    Evaluates performance of the PQD classifier according to the following metrics:
    accuracy, f1-score and confusion matrix

    Args:
        model: PQD classifier
        data_loader: inputs and labels
        device: cuda or cpu

    Returns:
        accuracy: the accuracy result for a given model and a dataset
        f1: f1 score result for a given model and a dataset
        cm: confusion matrix result for a given model and dataset
    """
    # Set up the model and evaluation parameters
    model.eval()
    y_true = torch.tensor([], dtype=torch.long, device=device)
    y_pred = torch.tensor([], dtype=torch.long, device=device)

    # Iterate over batches in the test set and make predictions
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, dim=1)
            y_true = torch.cat((y_true, labels))
            y_pred = torch.cat((y_pred, predictions))

    # Convert from torch to numpy and flatten the label and prediction arrays
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='macro')
    cm = confusion_matrix(y_true_np, y_pred_np)

    return float("{:.3f}".format(accuracy)), float("{:.3f}".format(f1)), cm


def calc_avg_robustness(org_signals, adv_signals):
    """
        Calculates average robustness of the perturbed dataset w.r.t the original dataset

        Args:
            org_signals: the original set of signals, of shape (batch_size, channels (3), input length (224x224))
            adv_signals: the modified dataset

        Returns:
            avg_robustness: the average robustness of the adversarial set, w.r.t the original set
    """
    diff_norm = torch.sqrt(torch.sum((adv_signals - org_signals) ** 2, axis=(1, 2, 3)))
    org_norm = torch.sqrt(torch.sum(org_signals ** 2, axis=(1, 2, 3)))
    avg_robustness = torch.sum(diff_norm / org_norm) / len(org_signals)

    return float("{:.3f}".format(avg_robustness))


def calc_mcr(org_loader, adv_loader, model, device):
    """
        Calculates misclassification rate of the perturbed dataset w.r.t the original dataset

        Args:
            org_loader: the original set of signals, with their labels
            adv_loader: the modified set of signals, with their labels
            model: model of the DL-based PQD classifier
            device: cuda or cpu

        Returns:
            mcr: the misclassification rate of the adversarial set, w.r.t the original set
    """
    model.eval()
    total_pred_change = 0
    total_signals = 0
    with torch.no_grad():
        for (org_data, adv_data) in zip(org_loader, adv_loader):
            org_inputs, org_labels = org_data
            org_inputs = org_inputs.to(device)
            org_outputs = model(org_inputs)
            _, org_predicted = torch.max(org_outputs.data, 1)

            adv_inputs, _ = adv_data
            adv_inputs = adv_inputs.to(device)
            adv_outputs = model(adv_inputs)
            _, adv_predicted = torch.max(adv_outputs.data, 1)

            total_signals += org_inputs.size(0)
            total_pred_change += (adv_predicted != org_predicted).sum().item()

    mcr = float("{:.3f}".format(total_pred_change / total_signals))

    return mcr
