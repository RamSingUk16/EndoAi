"""
Analyze model confidence and prediction distributions
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from constants import CLASS_ORDER, NE_SUBTYPES, EH_SUBTYPES

def load_data(split_path):
    """Load validation data."""
    from train import gather_dataset
    file_paths, main_labels, ne_labels, eh_labels = gather_dataset(split_path)
    return file_paths, main_labels, ne_labels, eh_labels

def predict_batch(model, images):
    """Get model predictions with confidence scores."""
    predictions = model.predict(images, verbose=0)
    # Model returns a list of outputs in the same order as the model's outputs
    return {
        'main': predictions[0],
        'ne': predictions[1],
        'eh': predictions[2]
    }

def analyze_confidence(model, dataset, output_dir):
    """Analyze prediction confidence and generate visualizations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Collect predictions and true labels
    all_preds = {
        'main': [], 'ne': [], 'eh': []
    }
    all_labels = {
        'main': [], 'ne': [], 'eh': []
    }
    all_confidences = {
        'main': [], 'ne': [], 'eh': []
    }
    
    for images, labels in dataset:
        preds = predict_batch(model, images)
        
        # Store predictions and confidences
        for head in ['main', 'ne', 'eh']:
            pred_probs = preds[head]
            true_labels = labels[f'{head}_output'].numpy()
            
            pred_classes = np.argmax(pred_probs, axis=1)
            confidences = np.max(pred_probs, axis=1)
            
            all_preds[head].extend(pred_classes)
            all_labels[head].extend(true_labels)
            all_confidences[head].extend(confidences)
    
    # Convert to arrays
    for head in ['main', 'ne', 'eh']:
        all_preds[head] = np.array(all_preds[head])
        all_labels[head] = np.array(all_labels[head])
        all_confidences[head] = np.array(all_confidences[head])
    
    # Generate visualizations and analysis
    heads = {
        'main': ('Main Classification', CLASS_ORDER),
        'ne': ('NE Subtype', NE_SUBTYPES),
        'eh': ('EH Subtype', EH_SUBTYPES)
    }
    
    report = {
        'confidence_stats': {},
        'accuracy_by_confidence': {},
        'class_distribution': {},
        'confusion_matrices': {}
    }
    
    for head, (title, classes) in heads.items():
        # Confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_confidences[head], bins=20, alpha=0.7)
        plt.title(f'Prediction Confidence Distribution - {title}')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(f'{output_dir}/{head}_confidence_dist.png')
        plt.close()
        
        # Confidence stats
        report['confidence_stats'][head] = {
            'mean': float(np.mean(all_confidences[head])),
            'median': float(np.median(all_confidences[head])),
            'std': float(np.std(all_confidences[head]))
        }
        
        # Accuracy by confidence threshold
        thresholds = np.linspace(0, 1, 20)
        accuracies = []
        for thresh in thresholds:
            mask = all_confidences[head] >= thresh
            if np.sum(mask) > 0:
                acc = np.mean(all_preds[head][mask] == all_labels[head][mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies)
        plt.title(f'Accuracy vs Confidence Threshold - {title}')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(f'{output_dir}/{head}_accuracy_by_confidence.png')
        plt.close()
        
        report['accuracy_by_confidence'][head] = {
            'thresholds': thresholds.tolist(),
            'accuracies': accuracies
        }
        
        # Class distribution and confusion matrix
        if head in ['ne', 'eh']:
            # Filter out -1 labels (not applicable)
            mask = all_labels[head] != -1
            filtered_preds = all_preds[head][mask]
            filtered_labels = all_labels[head][mask]
        else:
            filtered_preds = all_preds[head]
            filtered_labels = all_labels[head]
        
        # Confusion matrix
        cm = confusion_matrix(filtered_labels, filtered_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{output_dir}/{head}_confusion_matrix.png')
        plt.close()
        
        report['confusion_matrices'][head] = cm.tolist()
        
        # Class distribution
        pred_dist = np.bincount(filtered_preds, minlength=len(classes))
        true_dist = np.bincount(filtered_labels, minlength=len(classes))
        
        report['class_distribution'][head] = {
            'predicted': pred_dist.tolist(),
            'true': true_dist.tolist()
        }
    
    # Save report
    with open(f'{output_dir}/confidence_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def load_model_and_create_dataset(model_path, test_data_path, batch_size=32):
    """Load model and create dataset for evaluation."""
    from train import masked_sparse_categorical_crossentropy
    custom_objects = {
        'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    file_paths, main_labels, ne_labels, eh_labels = load_data(test_data_path)
    
    # Create dataset
    from train import build_dataset
    dataset = build_dataset(
        file_paths, main_labels, ne_labels, eh_labels,
        batch_size=batch_size, shuffle=False, augment_layer=None
    )
    
    return model, dataset

def print_report_summary(report):
    """Print a summary of the confidence analysis report."""
    print("\nConfidence Analysis Summary")
    print("==========================")
    
    heads = {'main': 'Main Task', 'ne': 'NE Subtype', 'eh': 'EH Subtype'}
    
    for head, name in heads.items():
        print(f"\n{name}:")
        stats = report['confidence_stats'][head]
        print(f"Mean confidence: {stats['mean']:.3f}")
        print(f"Median confidence: {stats['median']:.3f}")
        print(f"Std deviation: {stats['std']:.3f}")
        
        # Get accuracy at different confidence thresholds
        thresholds = report['accuracy_by_confidence'][head]['thresholds']
        accuracies = report['accuracy_by_confidence'][head]['accuracies']
        high_conf_acc = accuracies[-5]  # Accuracy at ~80% confidence threshold
        print(f"Accuracy at high confidence (80%): {high_conf_acc:.3f}")
        
        # Class distribution analysis
        dist = report['class_distribution'][head]
        total_true = sum(dist['true'])
        total_pred = sum(dist['predicted'])
        
        print("\nClass Distribution (True/Predicted):")
        classes = CLASS_ORDER if head == 'main' else (NE_SUBTYPES if head == 'ne' else EH_SUBTYPES)
        for i, cls in enumerate(classes):
            true_pct = dist['true'][i] / total_true * 100
            pred_pct = dist['predicted'][i] / total_pred * 100
            print(f"{cls}: {true_pct:.1f}% / {pred_pct:.1f}%")

def main():
    model_path = 'models/model_v1.0.0.h5'
    test_data_path = 'dataset_split/test'
    output_dir = 'confidence_analysis'
    
    print(f"Loading model from {model_path}")
    model, dataset = load_model_and_create_dataset(model_path, test_data_path)
    
    print("Running confidence analysis...")
    report = analyze_confidence(model, dataset, output_dir)
    
    print("\nAnalysis complete!")
    print(f"Visualizations saved to {output_dir}/")
    
    print_report_summary(report)

if __name__ == '__main__':
    main()