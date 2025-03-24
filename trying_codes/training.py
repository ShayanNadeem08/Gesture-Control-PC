
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch.optim as optim
from preprocessing import convert_batch_grey

# Function to train the model for one epoch
def train_epoch(model, train_loader, train_idx_list, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for idx, (inputs, targets) in enumerate(train_loader):
        print(inputs.size(), targets.size())
        if idx not in train_idx_list:
            continue

        inputs = convert_batch_grey(inputs)
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        #loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    return train_loss, train_acc

# Function to evaluate the model
def validate(model, valid_loader, valid_idx_list, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(valid_loader):
            if idx not in valid_idx_list:
                continue

            inputs = convert_batch_grey(inputs)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Save predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    valid_loss = running_loss / total
    valid_acc = 100. * correct / total

    return valid_loss, valid_acc, all_preds, all_targets

# Function to do KFold cross-validation training
def kfold_train_and_validate(model, train_loader, device, num_epochs, num_kfold_splits, batch_size):
    """
    Train and evaluate the 3D-CNN model using k-fold cross-validation
    with stratification across subjects.

    Args:
        X: Sequences of images tensor
        y: Labels tensor
        subject_ids: Subject IDs for cross-validation
        model_type: Type of model to use

    Returns:
        results: Dictionary containing evaluation results
        best_model_state: State dict of the best performing model
    """ 

    # Initialize KFold cross-validator
    print("Using",num_kfold_splits,"fold cross validation")
    kf = KFold(n_splits=num_kfold_splits, shuffle=True, random_state=42)

    # Initialize lists to store results
    subject_accuracies = []
    all_y_true = []
    all_y_pred = []

    # Track the best model across all folds
    best_overall_acc = 0
    final_best_model_state = None

    # Iterate through folds
    for fold, (train_idx, valid_idx) in enumerate(kf.split(range(len(train_loader))), 1):
        print(f"Fold {fold}/{num_kfold_splits}:")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs
        )

        # Variables for early stopping
        best_valid_acc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            train_loss, train_acc, valid_loss, valid_acc = np.random.rand(4)
            
            # Train for one epoch
            train_loss, train_acc = train_epoch(model, train_loader, train_idx, criterion, optimizer, device)

            # Evaluate on test set
            valid_loss, valid_acc, _, _ = validate(model, train_loader, valid_idx, criterion, device)

            # Print progress
            print(f'  Epoch {epoch+1}/{num_epochs}: '
                  f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'  Valid Loss: {valid_loss:.4f}, Test Acc: {valid_acc:.2f}%')

            # Check for improvement
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model for final evaluation
        model.load_state_dict(best_model_state)

        # Final evaluation
        _, final_valid_acc, y_pred, y_true = validate(model, valid_loader, criterion, device)

        # Store results
        subject_accuracies.append(final_valid_acc / 100.0)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold} Final Validation Accuracy: {final_valid_acc:.2f}%")

        # Keep track of the best model across all folds
        if final_valid_acc > best_overall_acc:
            best_overall_acc = final_valid_acc
            final_best_model_state = best_model_state.copy()

    # Calculate overall metrics
    overall_accuracy = np.mean(subject_accuracies)
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)

    # Store results
    results = {
        'subject_accuracies': subject_accuracies,
        'overall_accuracy': overall_accuracy,
        'confusion_matrix': conf_matrix,
        'y_true': all_y_true,
        'y_pred': all_y_pred
    }

    return results, final_best_model_state, model

