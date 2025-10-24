"""
check_model_metrics.py

Parse the metadata JSON to extract and display model performance metrics.
Helps evaluate if the model confidence is sufficient for production use.
"""

import json
import os
from pathlib import Path


def display_metrics(metadata_path):
    """Load and display key training metrics from metadata JSON."""
    
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        print("Training may still be in progress.")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("=" * 80)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 80)
    
    # Model info
    print(f"\nModel Version: {metadata.get('version', 'N/A')}")
    print(f"Created: {metadata.get('created_at', 'N/A')}")
    print(f"Model Path: {metadata.get('model_path', 'N/A')}")
    
    # Dataset info
    dataset = metadata.get('dataset', {})
    train_counts = dataset.get('train_counts', {})
    test_counts = dataset.get('test_counts', {})
    
    print(f"\n{'DATASET DISTRIBUTION':-^80}")
    print(f"Training samples: {train_counts.get('total', 'N/A')}")
    print(f"Test samples: {test_counts.get('total', 'N/A')}")
    
    print("\nClass distribution (training):")
    for cls, count in train_counts.get('main', {}).items():
        print(f"  {cls}: {count}")
    
    # Training history
    training = metadata.get('training', {})
    history = training.get('history', {})
    
    if history:
        final_epoch = training.get('final_epoch', len(history.get('loss', [])))
        best_val_loss = training.get('best_val_loss', 'N/A')
        
        print(f"\n{'TRAINING RESULTS':-^80}")
        print(f"Total epochs: {final_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}" if isinstance(best_val_loss, float) else f"Best validation loss: {best_val_loss}")
        
        # Final epoch metrics
        print(f"\nFinal epoch metrics:")
        for key in sorted(history.keys()):
            if history[key]:
                final_value = history[key][-1]
                print(f"  {key}: {final_value:.4f}" if isinstance(final_value, (int, float)) else f"  {key}: {final_value}")
        
        # Main output accuracy (most important for confidence)
        if 'main_output_accuracy' in history and history['main_output_accuracy']:
            train_acc = history['main_output_accuracy'][-1]
            val_acc = history.get('val_main_output_accuracy', [None])[-1]
            
            print(f"\n{'CONFIDENCE ASSESSMENT':-^80}")
            print(f"Main classification accuracy (training): {train_acc*100:.2f}%")
            if val_acc:
                print(f"Main classification accuracy (validation): {val_acc*100:.2f}%")
                
                # Production readiness assessment
                print(f"\n{'PRODUCTION READINESS':-^80}")
                if val_acc >= 0.90:
                    print("✓ EXCELLENT - Model shows high confidence (≥90% validation accuracy)")
                    print("  Recommended for production use.")
                elif val_acc >= 0.80:
                    print("✓ GOOD - Model shows acceptable confidence (80-90% validation accuracy)")
                    print("  Suitable for production with monitoring.")
                elif val_acc >= 0.70:
                    print("⚠ MODERATE - Model shows moderate confidence (70-80% validation accuracy)")
                    print("  Consider additional training or data augmentation.")
                else:
                    print("✗ LOW - Model confidence is below recommended threshold (<70%)")
                    print("  Requires significant improvement before production use.")
                    print("  Recommendations:")
                    print("  - Increase training epochs")
                    print("  - Add more training data")
                    print("  - Review data quality and class balance")
                    print("  - Tune hyperparameters")
    
    print("=" * 80)
    return metadata


def main():
    # Look for the latest metadata file
    models_dir = Path("models")
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    metadata_files = list(models_dir.glob("*_metadata.json"))
    
    if not metadata_files:
        print("No metadata files found. Training may not have completed yet.")
        return
    
    # Get the most recent metadata file
    latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Reading metrics from: {latest_metadata}\n")
    display_metrics(str(latest_metadata))


if __name__ == "__main__":
    main()
