"""
Analyze training history from model metadata and generate visualizations.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_training_history(metadata_path):
    """Load training history from metadata JSON file."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['training']['history']

def plot_metrics(history, output_dir):
    """Plot training metrics."""
    # Set style and create directory for plots
    plt.style.use('default')  # Use default style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    
    # Create directory for plots
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'loss_curves.png')
    plt.close()
    
    # Plot accuracies for each head
    heads = ['main_output', 'ne_output', 'eh_output']
    plt.figure(figsize=(12, 6))
    for head in heads:
        plt.plot(history[f'{head}_accuracy'], label=f'{head} (Train)')
        plt.plot(history[f'val_{head}_accuracy'], label=f'{head} (Val)')
    plt.title('Model Accuracy by Output Head')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'accuracy_curves.png')
    plt.close()

    # Plot individual head losses
    plt.figure(figsize=(12, 6))
    for head in heads:
        plt.plot(history[f'{head}_loss'], label=f'{head} (Train)')
        plt.plot(history[f'val_{head}_loss'], label=f'{head} (Val)')
    plt.title('Loss by Output Head')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'head_losses.png')
    plt.close()

def analyze_convergence(history):
    """Analyze training convergence and performance."""
    print("\nTraining Analysis:")
    print("-----------------")
    
    # Find best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    print(f"Best validation loss achieved at epoch {best_epoch}")
    
    # Final metrics
    final_metrics = {
        'Main Accuracy': history['val_main_output_accuracy'][-1],
        'NE Subtype Accuracy': history['val_ne_output_accuracy'][-1],
        'EH Subtype Accuracy': history['val_eh_output_accuracy'][-1]
    }
    print("\nFinal Validation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value*100:.2f}%")
    
    # Analyze convergence
    train_loss = history['loss']
    val_loss = history['val_loss']
    
    # Check if training was still improving
    final_window = 5
    if len(train_loss) >= final_window:
        recent_improvement = (np.mean(val_loss[-final_window:]) < 
                            np.mean(val_loss[-2*final_window:-final_window]))
        print(f"\nTraining was{' still' if recent_improvement else ' not'} improving "
              f"in final {final_window} epochs")
    
    # Check for overfitting
    best_train = min(train_loss)
    best_val = min(val_loss)
    gap = best_train / best_val if best_val > 0 else float('inf')
    
    if gap > 1.3:  # More than 30% gap
        print("\nPotential overfitting detected - validation loss significantly higher than training loss")
        print("Consider:")
        print("- Increasing regularization (dropout, weight decay)")
        print("- Using data augmentation")
        print("- Reducing model capacity")
    
    # Learning rate analysis
    if len(train_loss) >= 3:
        loss_smoothness = np.mean(np.abs(np.diff(train_loss[-10:])))
        if loss_smoothness > 0.1:  # High variance in recent loss
            print("\nUnstable training detected - consider reducing learning rate")
    
    print("\nRecommendations:")
    print("----------------")
    if final_metrics['Main Accuracy'] < 0.5:
        print("- Model performance is below 50% accuracy, consider:")
        print("  * Increasing model capacity")
        print("  * Adjusting class weights for better balance")
        print("  * Adding more training data or augmentation")
    
    if best_epoch == len(train_loss):
        print("- Training might benefit from running for more epochs")
    
    if best_epoch < len(train_loss) / 2:
        print("- Early convergence detected, try:")
        print("  * Higher learning rate")
        print("  * Adjusting learning rate schedule")
        print("  * Different optimizer settings")

def main():
    metadata_path = 'models/model_v1.0.0_metadata.json'
    output_dir = 'training_analysis'
    
    print(f"Loading training history from {metadata_path}")
    history = load_training_history(metadata_path)
    
    print("Generating visualization plots...")
    plot_metrics(history, output_dir)
    print(f"Plots saved to {output_dir}/")
    
    analyze_convergence(history)

if __name__ == '__main__':
    main()