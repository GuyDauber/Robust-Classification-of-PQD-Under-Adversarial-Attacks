import torch
from time import time
import os


def train_model(model, name, train_loader, val_loader, optimizer, criterion, n_epochs, patience, device):
    """
    Training PQD classifier and saving loss and accuracy vectors of training and validation and the best model

    Args:
        model: DL-based PQD classifier
        name: string which indicates the architecture of the classifier
        train_loader: train dataset loader
        val_loader: validation dataset loader
        optimizer: optimization algorithm for training
        criterion: loss function
        n_epochs: max number of epochs
        patience: max number of bad validation steps for early stopping
        device: cuda or cpu
    """

    train_loss_vec, val_loss_vec = [], []
    train_accuracy_vec, val_accuracy_vec = [], []
    best_loss = torch.inf
    best_accuracy = 0.0
    bad_epochs = 0
    path_model = "models\\" + name + "_model_" + ".pth"
    path_train_loss = "losses\\" + name + "_train_loss_" + ".pth"
    path_val_loss = "losses\\" + name + "_val_loss_" + ".pth"
    path_train_accuracy = "accuracies\\" + name + "_train_accuracy_" + ".pth"
    path_val_accuracy = "accuracies\\" + name + "_val_accuracy_" + ".pth"

    print("\nTraining model " + name)
    start = time()

    for epoch in range(n_epochs):

        model.train()
        train_loss = 0.0

        for i, (inputs_train_batch, labels_train_batch) in enumerate(train_loader):
            inputs_train_batch, labels_train_batch = inputs_train_batch.to(device), labels_train_batch.to(device)
            # Clear gradients w.r.t parameters
            optimizer.zero_grad()
            # Make predictions
            predictions = model(inputs_train_batch)
            # Compute loss for back-propagation
            loss = criterion(predictions, labels_train_batch)
            # Compute gradients w.r.t parameters
            loss.backward()
            # Update parameters
            optimizer.step()
            # Compute loss and accuracy
            train_loss += loss.data.item()
        train_loss /= len(train_loader)
        train_accuracy = compute_accuracy(model, train_loader, device)
        train_loss_vec.append(train_loss)
        train_accuracy_vec.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs_val_batch, labels_val_batch) in enumerate(val_loader):
                inputs_val_batch, labels_val_batch = inputs_val_batch.to(device), labels_val_batch.to(device)
                # Make predictions
                predictions = model(inputs_val_batch)
                # Compute loss
                loss = criterion(predictions, labels_val_batch)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_accuracy = compute_accuracy(model, val_loader, device)
            val_loss_vec.append(val_loss)
            val_accuracy_vec.append(val_accuracy)

        if val_loss < best_loss:
            print("------------------------------saving good model------------------------------")
            best_loss = val_loss
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(os.getcwd(), path_model))
            bad_epochs = 0
        else:
            bad_epochs = bad_epochs + 1
            if bad_epochs == patience:
                print("Early stopping")
                break

        print(f"Epoch: {epoch+1} | Bad epochs: {bad_epochs} | "
              f"Training accuracy = {train_accuracy * 100:.2f}% | Validation accuracy = {val_accuracy * 100:.2f}%")

    end = time()
    print(f"Training time = {end - start:.0f} seconds")
    print(f"Best loss = {best_loss:.2f} | Best accuracy = {best_accuracy * 100:.2f}%")

    # Save loss and accuracy vectors for plots
    torch.save(train_loss_vec, os.path.join(os.getcwd(), path_train_loss))
    torch.save(train_accuracy_vec, os.path.join(os.getcwd(), path_train_accuracy))
    torch.save(val_loss_vec, os.path.join(os.getcwd(), path_val_loss))
    torch.save(train_accuracy_vec, os.path.join(os.getcwd(), path_val_accuracy))


def compute_accuracy(model, dataloader, device):
    """
    Compute accuracy of the classification model over a certain dataset, used during training process.

    Args:
        model: DL-based PQD classifier
        dataloader: inputs and labels
        device: cuda or cpu

    Returns:
        accuracy: accuracy of the model over the dataset
    """
    model.eval()
    total_correct = 0
    total_inputs = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_inputs += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    model_accuracy = total_correct / total_inputs

    return model_accuracy
