"""
train.py

Implements a tf.data pipeline for the PathoPulse trainer proof-of-concept.

Features:
- Reads images from dataset_split/{train,test}
- Resizes to 224x224 and normalizes to ImageNet stats
- Configurable augmentations (RandomFlip, RandomRotation, RandomZoom, RandomContrast)
- Produces datasets batched for training and testing
- Smoke-test: prints batch shapes and class distributions for one batch
"""

import os
import json
import yaml
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


CLASS_ORDER = ['NE', 'EP', 'EH', 'EA']
NE_SUBTYPES = ['Follicular', 'Luteal', 'Menstrual']
EH_SUBTYPES = ['Simple', 'Complex']


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


def make_augmentation_layer(config):
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


def build_dataset(file_paths, main_labels, ne_labels, eh_labels, batch_size=32, shuffle=True, augment_layer=None):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, main_labels, ne_labels, eh_labels))

    AUTOTUNE = tf.data.AUTOTUNE

    def _load(path, main, ne, eh):
        image_bytes = tf.io.read_file(path)
        # Use tf.py_function to call preprocess_image which uses tf ops; here we avoid py_function to keep graph
        img = preprocess_image(image_bytes, augment_layer, training=shuffle)
        return img, {'main': main, 'ne_sub': ne, 'eh_sub': eh}

    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=42)

    ds = ds.map(lambda p, m, n, e: _load(p, m, n, e), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


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

    aug_layer = make_augmentation_layer(config)

    train_ds = build_dataset(train_files, train_main, train_ne, train_eh, batch_size=batch_size, shuffle=True, augment_layer=aug_layer)
    test_ds = build_dataset(test_files, test_main, test_ne, test_eh, batch_size=batch_size, shuffle=False, augment_layer=None)

    # Smoke-test: iterate one batch from train and print shapes / distributions
    for batch in train_ds.take(1):
        imgs, labels = batch
        print(f"Image batch shape: {imgs.shape}")
        main = labels['main'].numpy()
        ne = labels['ne_sub'].numpy()
        eh = labels['eh_sub'].numpy()
        unique, counts = np.unique(main, return_counts=True)
        dist = dict(zip([CLASS_ORDER[int(u)] for u in unique], counts.tolist()))
        print(f"Main class distribution in batch: {dist}")
        print(f"NE subtype sample (first 10): {ne[:10]}")
        print(f"EH subtype sample (first 10): {eh[:10]}")


if __name__ == '__main__':
    main()
