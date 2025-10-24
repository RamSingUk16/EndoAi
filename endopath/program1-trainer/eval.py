"""
eval.py

Evaluates a trained PathoPulse model on the test set and generates:
- Confusion matrices for each head
- Precision/Recall/F1 scores
- ROC curves and AUC scores
- Summary plots and metrics in PDF format
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from split_dataset import load_split_manifest
from train import create_dataset
from constants import CLASS_ORDER, NE_SUBTYPES, EH_SUBTYPES


def masked_sparse_categorical_crossentropy(y_true: tf.Tensor,
                                         y_pred: tf.Tensor,
                                         mask_value: int = -1) -> tf.Tensor:
    """Compute loss only on samples where y_true != mask_value.

    Duplicate of the training-time custom loss so saved models referencing
    it can be loaded here via custom_objects.
    """
    # Create mask where target is not mask_value
    mask = tf.not_equal(y_true, mask_value)
    mask = tf.reshape(mask, [-1])

    # Reshape tensors
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

    # Get the valid targets and predictions
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # If no valid samples, return 0
    if tf.equal(tf.size(y_true_masked), 0):
        return tf.constant(0.0, dtype=tf.float32)

    # Compute loss only on valid samples
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(loss)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm, labels, title, figsize=(8, 6)):
    """Plot a confusion matrix with pretty formatting."""
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()


def plot_roc_curves(y_true, y_pred, labels, title):
    """Plot ROC curves for multi-class classification."""
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC curve and AUC for each class
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()


def calculate_metrics(y_true, y_pred, labels):
    """Calculate precision, recall, and F1 score for each class."""
    # Convert probabilities to class predictions
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Calculate precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_classes, 
        y_pred_classes, 
        average=None,
        labels=range(len(labels))
    )
    
    # Create metrics dictionary
    metrics = {
        'confusion_matrix': cm,
        'class_metrics': {
            label: {
                'precision': p,
                'recall': r,
                'f1': f,
                'support': s
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        'overall': {
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1': np.mean(f1)
        }
    }
    
    return metrics


def evaluate_model(model_path):
    """Run full evaluation on a trained model."""
    # Load configuration
    config = load_config()
    
    # Load test set
    split_path = os.path.join(config['paths']['split_output'], 'split_manifest.csv')
    _, test_df = load_split_manifest(split_path)
    
    # Create test dataset
    test_ds = create_dataset(
        test_df,
        config['paths']['dataset_root'],
        config['model']['input_size'],
        batch_size=32,
        training=False,
        augment=False
    )
    
    # Load model (include custom loss used during training)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy
        }
    )
    
    # Get predictions
    predictions = model.predict(test_ds)
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    # Extract true labels
    y_true_main = np.array(tf.concat([batch[1]['main_output'] 
                                    for batch in test_ds], axis=0))
    y_true_ne = np.array(tf.concat([batch[1]['ne_output'] 
                                   for batch in test_ds], axis=0))
    y_true_eh = np.array(tf.concat([batch[1]['eh_output'] 
                                   for batch in test_ds], axis=0))
    
    # Calculate metrics for each head
    metrics = {
        'main': calculate_metrics(y_true_main, predictions[0], CLASS_ORDER),
        'ne': calculate_metrics(y_true_ne, predictions[1], NE_SUBTYPES),
        'eh': calculate_metrics(y_true_eh, predictions[2], EH_SUBTYPES)
    }
    
    # Create output directory
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join('eval_results', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrices
    plot_confusion_matrix(
        metrics['main']['confusion_matrix'],
        CLASS_ORDER,
        'Main Classes Confusion Matrix'
    )
    plt.savefig(os.path.join(output_dir, 'main_confusion.png'))
    plt.close()
    
    plot_confusion_matrix(
        metrics['ne']['confusion_matrix'],
        NE_SUBTYPES,
        'NE Subtypes Confusion Matrix'
    )
    plt.savefig(os.path.join(output_dir, 'ne_confusion.png'))
    plt.close()
    
    plot_confusion_matrix(
        metrics['eh']['confusion_matrix'],
        EH_SUBTYPES,
        'EH Subtypes Confusion Matrix'
    )
    plt.savefig(os.path.join(output_dir, 'eh_confusion.png'))
    plt.close()
    
    # Plot ROC curves
    plot_roc_curves(y_true_main, predictions[0], CLASS_ORDER, 
                   'Main Classes ROC Curves')
    plt.savefig(os.path.join(output_dir, 'main_roc.png'))
    plt.close()
    
    plot_roc_curves(y_true_ne, predictions[1], NE_SUBTYPES, 
                   'NE Subtypes ROC Curves')
    plt.savefig(os.path.join(output_dir, 'ne_roc.png'))
    plt.close()
    
    plot_roc_curves(y_true_eh, predictions[2], EH_SUBTYPES, 
                   'EH Subtypes ROC Curves')
    plt.savefig(os.path.join(output_dir, 'eh_roc.png'))
    plt.close()
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    return metrics


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == '__main__':
    # Get latest model file
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) 
                  if f.endswith('.h5')]
    if not model_files:
        raise ValueError("No model files found in models/ directory")
    
    latest_model = max(model_files, 
                      key=lambda f: os.path.getctime(os.path.join(models_dir, f)))
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"Evaluating model: {latest_model}")
    metrics = evaluate_model(model_path)
    
    # Print summary metrics
    print("\nEvaluation Results:")
    for head in ['main', 'ne', 'eh']:
        print(f"\n{head.upper()} Classification:")
        print(f"Overall Metrics:")
        for metric, value in metrics[head]['overall'].items():
            print(f"  {metric}: {value:.3f}")