"""
train.py

Implements training pipeline for PathoPulse trainer with:
- tf.data input pipeline (224x224, ImageNet norm, configurable augmentations)
- masked losses for subtype heads
- class weights computed from training distribution
- two-phase training (frozen/unfrozen) with EarlyStopping
"""

import os
import json
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path
from constants import VERSION, IMAGENET_MEAN, IMAGENET_STD, CLASS_ORDER, NE_SUBTYPES, EH_SUBTYPES

IMAGE_SIZE = (224, 224)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def gather_dataset(split_root: str):
    """Walk split_root (dataset_split/train or dataset_split/test) and return lists.

    Returns: file_paths, main_labels, ne_sub_labels, eh_sub_labels
    For samples where a subtype is not applicable, subtype label is -1.
    """
    file_paths = []
    main_labels = []
    ne_sub = []
    eh_sub = []

    for class_name in CLASS_ORDER:
        class_path = os.path.join(split_root, class_name)
        if not os.path.exists(class_path):
            continue

        if class_name == 'NE':
            for i, subtype in enumerate(NE_SUBTYPES):
                subdir = os.path.join(class_path, subtype)
                if not os.path.exists(subdir):
                    continue
                for fname in os.listdir(subdir):
                    if not fname.upper().endswith('.JPG'):
                        continue
                    file_paths.append(os.path.join(subdir, fname))
                    main_labels.append(CLASS_ORDER.index('NE'))
                    ne_sub.append(i)
                    eh_sub.append(-1)
        elif class_name == 'EH':
            for i, subtype in enumerate(EH_SUBTYPES):
                subdir = os.path.join(class_path, subtype)
                if not os.path.exists(subdir):
                    continue
                for fname in os.listdir(subdir):
                    if not fname.upper().endswith('.JPG'):
                        continue
                    file_paths.append(os.path.join(subdir, fname))
                    main_labels.append(CLASS_ORDER.index('EH'))
                    ne_sub.append(-1)
                    eh_sub.append(i)
        else:
            # EP and EA (no subtypes)
            for fname in os.listdir(class_path):
                if not fname.upper().endswith('.JPG'):
                    continue
                file_paths.append(os.path.join(class_path, fname))
                main_labels.append(CLASS_ORDER.index(class_name))
                ne_sub.append(-1)
                eh_sub.append(-1)

    return file_paths, main_labels, ne_sub, eh_sub


def create_dataset(df, data_root, input_size, batch_size=32, training=True, augment=True, config=None):
    """Create a tf.data.Dataset from a dataframe of samples.
    
    Args:
        df: DataFrame with file_paths and labels
        data_root: Root directory containing images
        input_size: Model input size [height, width, channels]
        batch_size: Batch size for training
        training: Whether this is training set (enables shuffling)
        augment: Whether to apply data augmentation
        config: Configuration dictionary for augmentations
    """
    def preprocess(image_path, main_label, ne_label, eh_label):
        # Read and normalize image
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
        except tf.errors.NotFoundError:
            # Use a placeholder black image if file is not found
            img = tf.zeros(input_size, dtype=tf.float32)
        
        img = tf.image.resize(img, input_size[:2])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        
        # Convert labels to one-hot
        main_label = tf.one_hot(main_label, len(CLASS_ORDER))
        ne_label = tf.one_hot(ne_label, len(NE_SUBTYPES))
        eh_label = tf.one_hot(eh_label, len(EH_SUBTYPES))
        
        return img, {
            'main_output': main_label,
            'ne_output': ne_label,
            'eh_output': eh_label
        }
    
    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((
        df['file_path'],
        df['main_label'],
        df['ne_subtype'],
        df['eh_subtype']
    ))
    
    if training:
        ds = ds.shuffle(buffer_size=len(df))
    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training and augment:
        augmentation = make_augmentation_layer(config)
        if augmentation:
            ds = ds.map(lambda x, y: (augmentation(x), y),
                       num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_callbacks(config, model_name="model"):
    """Create training callbacks for model fitting.
    
    Args:
        config: Configuration dictionary with training parameters
        model_name: Base name for the model files
        
    Returns:
        List of callbacks for model.fit()
    """
    callbacks = []
    
    # Create models directory if it doesn't exist
    models_dir = config.get('paths', {}).get('models_output', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Model checkpoint to save best model
    checkpoint_path = os.path.join(models_dir, f"{model_name}.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping to prevent overfitting
    patience = config.get('training', {}).get('early_stopping_patience', 5)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,  # configurable
        restore_best_weights=True,
        mode='min',
        verbose=1,
        min_delta=1e-4
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=max(1, patience - 2),  # tie to early stopping for fast runs
        min_lr=1e-6,
        mode='min',
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard logging (disabled to avoid serialization issues in some TF versions)
    # log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    # tensorboard = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir,
    #     histogram_freq=1,
    #     write_graph=True
    # )
    # callbacks.append(tensorboard)
    
    return callbacks

def make_augmentation_layer(config):
    """Create a sequential layer for data augmentation based on config."""
    if not config.get('augmentation', {}).get('enabled', True):
        return None

    rot = config['augmentation'].get('rotation_range', 0) / 180.0
    zoom = config['augmentation'].get('zoom_range', 0.0)
    flip_h = config['augmentation'].get('horizontal_flip', True)
    flip_v = config['augmentation'].get('vertical_flip', False)
    contrast_factor = 0.1

    layers = []
    if flip_h or flip_v:
        # RandomFlip supports horizontal, vertical or both
        if flip_h and flip_v:
            layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
        elif flip_h:
            layers.append(tf.keras.layers.RandomFlip("horizontal"))
        else:
            layers.append(tf.keras.layers.RandomFlip("vertical"))

    if rot > 0:
        layers.append(tf.keras.layers.RandomRotation(rot))
    if zoom > 0:
        layers.append(tf.keras.layers.RandomZoom(height_factor=(-zoom, 0.0), width_factor=(-zoom, 0.0)))
    layers.append(tf.keras.layers.RandomContrast(contrast_factor))

    return tf.keras.Sequential(layers)


def preprocess_image(image_bytes, augment_layer=None, training=False):
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, IMAGE_SIZE)

    if training and augment_layer is not None:
        image = augment_layer(image)

    # ImageNet normalisation
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def compute_sample_weights(main_labels, ne_labels, eh_labels, class_weights):
    """Compute sample weights based on class weights for each head."""
    main_weights = np.array([class_weights['main'][int(l)] for l in main_labels])
    
    # For subtypes, mask out irrelevant samples (-1) by setting weight to 0
    ne_mask = ne_labels != -1
    ne_weights = np.zeros_like(ne_labels, dtype=np.float32)
    ne_weights[ne_mask] = [class_weights['ne_sub'][int(l)] for l in ne_labels[ne_mask]]
    
    eh_mask = eh_labels != -1
    eh_weights = np.zeros_like(eh_labels, dtype=np.float32)
    eh_weights[eh_mask] = [class_weights['eh_sub'][int(l)] for l in eh_labels[eh_mask]]
    
    # Combine weights - main weight always applies, subtype weights only when relevant
    sample_weights = main_weights + 0.5 * (ne_weights + eh_weights)  # 0.5 matches loss_weights
    return sample_weights

def build_dataset(file_paths, main_labels, ne_labels, eh_labels, batch_size=32, 
                 shuffle=True, augment_layer=None, class_weights=None):
    """Build tf.data.Dataset with optional sample weights for class balancing."""
    
    # Convert labels to numpy arrays for easier handling
    main_labels = np.array(main_labels)
    ne_labels = np.array(ne_labels)
    eh_labels = np.array(eh_labels)
    
    # Compute sample weights if class weights provided
    if class_weights:
        sample_weights = compute_sample_weights(main_labels, ne_labels, eh_labels, class_weights)
        ds = tf.data.Dataset.from_tensor_slices(
            ((file_paths, main_labels, ne_labels, eh_labels), sample_weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((file_paths, main_labels, ne_labels, eh_labels))

    AUTOTUNE = tf.data.AUTOTUNE

    if class_weights:
        def _load_with_weights(inputs, weights):
            path, main, ne, eh = inputs
            image_bytes = tf.io.read_file(path)
            img = preprocess_image(image_bytes, augment_layer, training=shuffle)
            return img, {'main_output': main, 'ne_output': ne, 'eh_output': eh}, weights
        
        if shuffle:
            ds = ds.shuffle(buffer_size=1000, seed=42)
        ds = ds.map(_load_with_weights, num_parallel_calls=AUTOTUNE)
    else:
        def _load(path, main, ne, eh):
            image_bytes = tf.io.read_file(path)
            img = preprocess_image(image_bytes, augment_layer, training=shuffle)
            return img, {'main_output': main, 'ne_output': ne, 'eh_output': eh}
        
        if shuffle:
            ds = ds.shuffle(buffer_size=1000, seed=42)
        ds = ds.map(lambda p, m, n, e: _load(p, m, n, e), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights."""
    unique, counts = np.unique(labels, return_counts=True)
    total = np.sum(counts)
    weights = {int(label): total / (len(unique) * count) 
              for label, count in zip(unique, counts)}
    return weights

def masked_sparse_categorical_crossentropy(y_true: tf.Tensor,
                                         y_pred: tf.Tensor,
                                         mask_value: int = -1) -> tf.Tensor:
    """Compute loss only on samples where y_true != mask_value."""
    # Create mask where target is not mask_value
    mask = tf.not_equal(y_true, mask_value)
    mask = tf.reshape(mask, [-1])
    
    # Reshape tensors
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    
    # Get the valid targets and predictions
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    
    def zero(): 
        return tf.constant(0.0, dtype=tf.float32)

    def compute():
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
        return tf.reduce_mean(loss)

    return tf.cond(tf.equal(tf.size(y_true_masked), 0), zero, compute)

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_metadata(config: dict,
                 model_path: str,
                 class_weights: Dict[str, Dict[int, float]],
                 train_counts: Dict[str, int],
                 test_counts: Dict[str, int],
                 history: Dict[str, List[float]]) -> None:
    """Save model metadata, including dataset stats and training history."""
    metadata = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "model_path": model_path,
        "config": config,
        "dataset": {
            "train_counts": train_counts,
            "test_counts": test_counts,
            "class_weights": class_weights
        },
        "training": {
            "history": history,
            "final_epoch": len(history["loss"]),
            "best_val_loss": min(history["val_loss"])
        }
    }
    
    metadata_path = model_path.replace(".h5", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)

def train_model(model: tf.keras.Model,
                train_ds: tf.data.Dataset,
                val_ds: tf.data.Dataset,
                class_weights: Dict[str, Dict[int, float]],
                config: dict) -> Tuple[tf.keras.Model, Dict[str, List[float]]]:
    """Train the model using the specified configuration."""
    
    initial_epochs = config["training"]["initial_epochs"]
    fine_tune_epochs = config["training"]["fine_tune_epochs"]
    initial_lr = config["training"]["initial_learning_rate"]
    fine_tune_lr = config["training"]["fine_tune_learning_rate"]
    
    # Compile model with masked losses and class weights
    losses = {
        "main_output": tf.keras.losses.SparseCategoricalCrossentropy(),
        "ne_output": masked_sparse_categorical_crossentropy,
        "eh_output": masked_sparse_categorical_crossentropy
    }
    
    loss_weights = {
        "main_output": 1.0,
        "ne_output": 0.5,  # Less weight on subtypes
        "eh_output": 0.5
    }
    
    metrics = {
        "main_output": "accuracy",
        "ne_output": "accuracy",
        "eh_output": "accuracy"
    }
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(initial_lr),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    # Get callbacks using the common create_callbacks function
    callbacks = create_callbacks(config, model_name=f"model_{VERSION}")
    
    print("Initial training phase (backbone frozen)...")
    # Pass class_weights for main head only (subtypes use masking)
    # Optional fast mode to limit steps per epoch for quick smoke tests
    fast_cfg = config.get('training', {})
    steps_per_epoch = fast_cfg.get('steps_per_epoch')
    validation_steps = fast_cfg.get('validation_steps')

    fit_kwargs = {}
    if steps_per_epoch:
        fit_kwargs['steps_per_epoch'] = steps_per_epoch
    if validation_steps:
        fit_kwargs['validation_steps'] = validation_steps

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        **fit_kwargs
    )
    
    # Fine-tuning phase
    print("\nFine-tuning phase (top half unfrozen)...")
    from model import unfreeze_top_half
    
    unfreeze_top_half(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=initial_epochs,
        callbacks=callbacks,
        **fit_kwargs
    )
    
    # Combine histories
    combined_history = {}
    for k in history.history.keys():
        combined_history[k] = (
            history.history[k] +
            fine_tune_history.history[k]
        )
    
    return model, combined_history

def main():
    config = load_config()
    split_root = config['paths'].get('split_output', 'dataset_split')
    train_root = os.path.join(split_root, 'train')
    test_root = os.path.join(split_root, 'test')
    
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    print(f"Loading train dataset from: {train_root}")
    train_files, train_main, train_ne, train_eh = gather_dataset(train_root)
    print(f"Found {len(train_files)} training samples")
    
    print(f"Loading test dataset from: {test_root}")
    test_files, test_main, test_ne, test_eh = gather_dataset(test_root)
    print(f"Found {len(test_files)} test samples")
    
    print("\nComputing class weights...")
    class_weights = {
        "main": compute_class_weights(train_main),
        "ne_sub": compute_class_weights([x for x in train_ne if x != -1]),
        "eh_sub": compute_class_weights([x for x in train_eh if x != -1])
    }
    print("Class weights:", class_weights)

    aug_layer = make_augmentation_layer(config)
    
    train_ds = build_dataset(train_files, train_main, train_ne, train_eh,
                           batch_size=batch_size, shuffle=True,
                           augment_layer=aug_layer, class_weights=class_weights)
    test_ds = build_dataset(test_files, test_main, test_ne, test_eh,
                         batch_size=batch_size, shuffle=False,
                         augment_layer=None)

    # Dataset statistics for metadata
    train_counts = {
        "total": len(train_files),
        "main": dict(zip(CLASS_ORDER,
                        [sum(1 for x in train_main if x == i)
                         for i in range(len(CLASS_ORDER))]))
    }
    test_counts = {
        "total": len(test_files),
        "main": dict(zip(CLASS_ORDER,
                        [sum(1 for x in test_main if x == i)
                         for i in range(len(CLASS_ORDER))]))
    }

    # Build model
    from model import build_model
    print("\nBuilding and training model...")
    model = build_model(config)

    # Ensure models directory and model path are known
    models_dir = config.get('paths', {}).get('models_output', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"model_{VERSION}.h5")

    # Optionally save an initial model (untrained) so backend can load immediately
    if config.get('training', {}).get('save_initial_model', False):
        try:
            model.save(model_path)
            print(f"Saved initial (untrained) model to: {model_path}")
        except Exception as e:
            print(f"Warning: could not save initial model: {e}")

    # Train model
    model, history = train_model(model, train_ds, test_ds,
                             class_weights, config)

    # Save metadata
    # Ensure a model file exists even if callbacks did not trigger
    try:
        if not os.path.exists(model_path):
            model.save(model_path)
            print(f"Forced save model to: {model_path}")
    except Exception as e:
        print(f"Warning: could not save trained model: {e}")

    save_metadata(config, model_path, class_weights,
               train_counts, test_counts, history)

    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {model_path.replace('.h5', '_metadata.json')}")


if __name__ == '__main__':
    main()
