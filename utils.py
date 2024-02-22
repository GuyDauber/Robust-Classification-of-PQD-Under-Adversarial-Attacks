import os
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def split_train_val_test(inputs, labels, train_split=0.8, val_split=0.25, seed=None, ratio_adv=0.05, ratio_retrain=0.5):
    """
    Splits dataset ready for supervised learning (inputs, labels) into train, validation and test datasets, each in the
    form of TensorDataset. Eventually, the percentages of train-validation-test datasets are 60%-20%-20%

    Args:
        inputs: PQD signals
        labels: labels of the PQD signals
        train_split: ratio of train (and validation) dataset out of the whole dataset
        val_split: ratio of validation dataset out of the train dataset
        seed: seed for splitting data to training and validation sets
        ratio_adv: dataset ratio of adversary
        ratio_retrain : dataset ration for adversarial retraining

    Returns:
        train_set: dataset which is allocated for the training process
        val_set: dataset which is allocated for the validation process
        inputs_test: inputs used for the test set
        labels_test: labels used for the test set
        test_set: dataset which is allocated for the final evaluation of the model
        train_set_adv: train dataset which is allocated for the adversarial training
        val_set_adv: validation dataset which is allocated for the adversarial training
        retrain_set: dataset which is allocated for the retrain process
    """
    # First, take the same inputs and labels for test-dataset
    size = len(inputs)
    train_val_size = int(train_split * size)
    inputs_test, labels_test = inputs[-(size - train_val_size):], labels[-(size - train_val_size):]
    test_set = TensorDataset(inputs_test, labels_test)

    # Second, randomly split the remaining inputs and labels into train-dataset and validation-dataset
    inputs_train_val, labels_train_val = inputs[0:train_val_size], labels[0:train_val_size]
    train_val_dataset = TensorDataset(inputs_train_val, labels_train_val)
    val_size = int(val_split * train_val_size)
    train_size = train_val_size - val_size
    train_set, val_set = random_split(train_val_dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(seed) if seed else None)

    # Optional: adversarial dataset for training a substitute model as part of black/grey-box threat model
    if seed is not None:
        random.seed(seed)  # Set the random seed for reproducibility

    # Randomly split the inputs and labels into train-dataset and validation-dataset for balanced adversary dataset
    train_val_size_adversary = int(len(inputs_train_val) * ratio_adv)  # Calculate the number of samples to select
    indices = balanced_indices_maker(labels, train_val_size_adversary)  # Randomly select balanced indices
    inputs_train_val_adversary, labels_train_val_adversary = inputs[indices], labels[indices]
    train_val_dataset_adversary = TensorDataset(inputs_train_val_adversary, labels_train_val_adversary)
    val_size_adversary = int(val_split * train_val_size_adversary)
    train_size_adversary = train_val_size_adversary - val_size_adversary
    train_set_adv, val_set_adv = random_split(train_val_dataset_adversary, [train_size_adversary, val_size_adversary],
                                              generator=torch.Generator().manual_seed(seed) if seed else None)

    # Make a balanced dataset for retraining process
    size_retrain = int(len(inputs_train_val) * ratio_retrain)  # Calculate the number of samples to select
    retrain_indices = balanced_indices_maker(labels, size_retrain)  # Randomly select balanced indices
    inputs_retrain, labels_retrain = inputs[retrain_indices], labels[retrain_indices]
    retrain_set = TensorDataset(inputs_retrain, labels_retrain)

    return train_set, val_set, inputs_test, labels_test, test_set, train_set_adv, val_set_adv, retrain_set


def balanced_indices_maker(cwt_labels, dataset_size, num_of_types=16):
    """
    Creates a set of indices from the labels, which includes all types of pqd signals.
    The indices are chosen both randomly and uniformly, such that each type contains same amount of indices.
    Args:
        cwt_labels: labels of the cwt signals
        dataset_size: target number for the size of the complete dataset of indices
        num_of_types: number of types of different pqd signals
    Returns:
        indices_list = set of indices for the desired distribution
    """

    type_size = int(dataset_size / num_of_types)
    indices_list = []

    for i in range(num_of_types):
        random_indices_i = np.random.choice(np.where(cwt_labels == i)[0], size=type_size, replace=False)
        random_indices_list_i = list(random_indices_i)
        indices_list += random_indices_list_i

    return indices_list


def signals_set_maker(pqd_path, cwt_signals, cwt_labels, num_of_samples, device, labels_order, train_splitter=0.8):
    """
    Creates a set of indices from the dataset, which includes all types of pqd signals
    Args:
        pqd_path: path of the mat file which contains the pqd signals
        cwt_signals: signals of pqd, after the cwt type transform
        cwt_labels: labels of the cwt signals
        num_of_samples: number of samples in each pqd signal
        device: cuda or cpu
        labels_order: list of the labels with their names
        train_splitter: splits the pqd dataset into train and test
    Returns:
        pqd_set: set of pqd signals, contains one from each type
        cwt_set: set of cwt signals, contains one from each type
        labels_set: set of the labels, contains one from each type
    """

    dataset = loadmat(pqd_path)

    signals, labels = [], []
    for i in range(0, 16 * 1000):
        signals.append(dataset['SignalsDataBase'][0][i]['signals'][0])
        labels.append(dataset['SignalsDataBase'][0][i]['labels'][0])

    signals = np.array(signals)
    labels = np.array(labels)
    labels = labels.reshape(-1, 1)

    size = len(signals)
    train_val_size = int(train_splitter * size)
    test_signals, test_labels = signals[-(size - train_val_size):], labels[-(size - train_val_size):]

    pqd_indices = np.empty(16, dtype=int)
    pqd_set = np.empty(shape=(16, num_of_samples), dtype='object')
    cwt_set = torch.tensor([], dtype=torch.long, device=device)
    labels_set = torch.tensor([], dtype=torch.long, device=device)

    for i, label in enumerate(labels_order):
        pqd_indices[i] = np.where(test_labels == label)[0][0]
        pqd_set[i] = test_signals[pqd_indices[i]]
        cwt_set = torch.cat((cwt_set, cwt_signals[pqd_indices[i]].unsqueeze(0)))
        labels_set = torch.cat((labels_set, cwt_labels[pqd_indices[i]].unsqueeze(0)))

    return pqd_set, cwt_set, labels_set


def performance_vec_loader(model, mode, attack_type):
    """
    Load performance metrics results vectors: accuracy, f1 score, misclassification rate, average robustness

    Args:
        model: PQD classifier
        mode: attack mode - black, gray, or white
        attack_type: type of adversarial attack - FGSM, BIM, PGD, or DeepFool

    Returns:
        acc_vec: accuracy vector as a function of epsilon, mode and attack dependent
        f1_vec: f1 score vector as a function of epsilon, mode and attack dependent
        mcr_vec: misclassification rate vector as a function of epsilon, mode and attack dependent
        avg_rob_vec: average robustness vector as a function of epsilon, mode and attack dependent
    """

    acc_path = "accuracies\\" + model + "_" + mode + "_" + attack_type + "_test_accuracy_" + ".pth"
    f1_path = "f1_scores\\" + model + "_" + mode + "_" + attack_type + "_test_f1_score_" + ".pth"
    mcr_path = "mis_class_rates\\" + model + "_" + mode + "_" + attack_type + "_test_misclassification_rate_" + ".pth"
    avg_rob_path = "average_robustness_scores\\" + model + "_" + mode + "_" + attack_type + \
                   "_test_average_robustness_" + ".pth"

    acc_vec = torch.load(os.path.join(os.getcwd(), acc_path))
    f1_vec = torch.load(os.path.join(os.getcwd(), f1_path))
    mcr_vec = torch.load(os.path.join(os.getcwd(), mcr_path))
    avg_rob_vec = torch.load(os.path.join(os.getcwd(), avg_rob_path))

    return acc_vec, f1_vec, mcr_vec, avg_rob_vec


class CustomLabelEncoder(LabelEncoder):
    def __init__(self, custom_order=None):
        self.custom_order = custom_order
        super().__init__()

    def fit(self, y):
        if self.custom_order is None:
            super().fit(y)
        else:
            self.classes_ = self.custom_order

        return self

    def transform(self, y):
        if self.custom_order is None:
            return super().transform(y)
        else:
            return [self.classes_.index(label) for label in y]

    def inverse_transform(self, y):
        if self.custom_order is None:
            return super().inverse_transform(y)
        else:
            return [self.classes_[idx] for idx in y]

    def fit_transform(self, y):
        return self.fit(y).transform(y)
