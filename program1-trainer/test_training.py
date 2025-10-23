"""Test script to verify training pipeline with a small subset of data."""
import os
import yaml
import tensorflow as tf
from train import create_dataset, create_callbacks
from model import build_model
from split_dataset import load_split_manifest

def test_training_pipeline():
    """Run a quick test of the training pipeline with minimal data."""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load split manifest
    split_path = os.path.join(config['paths']['split_output'], 'split_manifest.csv')
    train_df, val_df = load_split_manifest(split_path)
    
    # Take small subset for quick test
    train_subset = train_df.sample(n=min(32, len(train_df)), random_state=42)
    val_subset = val_df.sample(n=min(8, len(val_df)), random_state=42)
    
    # Create minimal datasets
    train_ds = create_dataset(
        train_subset, 
        config['paths']['dataset_root'],
        config['model']['input_size'],
        batch_size=4,
        augment=False,  # Disable augmentation for test
        config=config
    )
    val_ds = create_dataset(
        val_subset,
        config['paths']['dataset_root'],
        config['model']['input_size'],
        batch_size=4,
        training=False,
        config=config
    )
    
    # Build model
    model = build_model(config)
    
    # Initial compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['training']['initial_learning_rate']),
        loss={
            'main_output': tf.keras.losses.CategoricalCrossentropy(),
            'ne_output': tf.keras.losses.CategoricalCrossentropy(),
            'eh_output': tf.keras.losses.CategoricalCrossentropy()
        },
        metrics={'main_output': 'accuracy'}
    )
    
    # Create callbacks with shorter patience
    test_config = config.copy()
    test_config['training']['early_stopping_patience'] = 2
    callbacks = create_callbacks(test_config, "test_model")
    
    # Run quick training
    print("Testing initial training phase...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        callbacks=callbacks
    )
    
    # Test fine-tuning phase
    print("\nTesting fine-tuning phase...")
    model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(test_config['training']['fine_tune_learning_rate']),
        loss={
            'main_output': tf.keras.losses.CategoricalCrossentropy(),
            'ne_output': tf.keras.losses.CategoricalCrossentropy(),
            'eh_output': tf.keras.losses.CategoricalCrossentropy()
        },
        metrics={'main_output': 'accuracy'}
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        callbacks=callbacks
    )
    
    print("\nTraining pipeline test completed successfully!")

if __name__ == '__main__':
    test_training_pipeline()