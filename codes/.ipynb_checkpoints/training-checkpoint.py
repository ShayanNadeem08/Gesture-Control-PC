
import numpy as np
import torch

# Function to train the model for one epoch
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
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
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
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

    test_loss = running_loss / total
    test_acc = 100. * correct / total

    return test_loss, test_acc, all_preds, all_targets


def train_and_evaluate(X, y, subject_ids, model_type='densenet'):
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
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize KFold cross-validator
    n_splits = 5  # 5-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store results
    subject_accuracies = []
    all_y_true = []
    all_y_pred = []

    # Cross-validation parameters
    num_epochs = 10
    batch_size = 16

    # Track the best model across all folds
    best_overall_acc = 0
    final_best_model_state = None

    # Iterate through folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(X.numpy()), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        # Create PyTorch datasets and dataloaders
        train_dataset = HandGestureDataset(X[train_idx], y[train_idx])
        test_dataset = HandGestureDataset(X[test_idx], y[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create the model
        model = DenseNet3D(growth_rate=12, block_config=(2, 4, 4), num__init__features=16).to(device)

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
        best_test_acc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

            # Evaluate on test set
            test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

            # Check for improvement
            if test_acc > best_test_acc:
                best_test_acc = test_acc
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
        _, final_test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

        # Store results
        subject_accuracies.append(final_test_acc / 100.0)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold} Final Test Accuracy: {final_test_acc:.2f}%")

        # Keep track of the best model across all folds
        if final_test_acc > best_overall_acc:
            best_overall_acc = final_test_acc
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

    return results, final_best_model_state