import matplotlib.pyplot as plt

# Function to visualize results
def visualize_results(results, model_name='densenet'):
    """
    Visualize the cross-validation results.

    Args:
        results: Dictionary containing evaluation results
        model_name: Name of the model
    """
    # Plot subject accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results['subject_accuracies'])), results['subject_accuracies'])
    plt.axhline(y=results['overall_accuracy'], color='r', linestyle='-',
                label=f"Overall: {results['overall_accuracy']:.4f}")
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title(f'Leave-One-Subject-Out Cross-Validation Results ({model_name})')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f'subject_accuracies_{model_name}.png')
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    gesture_names = ['Down', 'Left', 'Right', 'Up']
    conf_matrix = results['confusion_matrix']
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=gesture_names, yticklabels=gesture_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
