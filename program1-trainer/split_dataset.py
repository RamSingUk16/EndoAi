"""
split_dataset.py - Simplified dataset splitter
"""

import os
import shutil
import random
import json
import yaml
import pandas as pd
from pathlib import Path
import constants


def load_split_manifest(manifest_path):
    """Load and parse the split manifest CSV file.
    
    Returns:
        train_df, val_df: DataFrames containing file paths and labels
    """
    df = pd.read_csv(manifest_path)
    manifest_dir = os.path.dirname(manifest_path)
    
    # Fix file paths to use dataset_split directory
    df['file_path'] = df.apply(
        lambda x: os.path.join(manifest_dir, x['split'], x['class'], 
                             x['subtype'] if pd.notna(x['subtype']) else '',
                             os.path.basename(x['file_path'])), axis=1)
    
    # Create label mappings
    class_to_idx = {cls: idx for idx, cls in enumerate(constants.CLASS_ORDER)}
    ne_to_idx = {sub: idx for idx, sub in enumerate(constants.NE_SUBTYPES)}
    eh_to_idx = {sub: idx for idx, sub in enumerate(constants.EH_SUBTYPES)}
    
    # Extract main and subtype labels
    df['main_label'] = df['class'].apply(lambda x: class_to_idx[x])
    df['ne_subtype'] = df.apply(lambda x: ne_to_idx[x['subtype']] if x['class'] == 'NE' else -1, axis=1)
    df['eh_subtype'] = df.apply(lambda x: eh_to_idx[x['subtype']] if x['class'] == 'EH' else -1, axis=1)
    
    # Split into train/test
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'test'].copy()
    
    return train_df, val_df

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def is_valid_filename(filename):
    return not any(c in filename for c in ['²', '⌐', 'ú'])

def list_images(base_dir):
    data = {
        'NE': {'Follicular': [], 'Luteal': [], 'Menstrual': []},
        'EH': {'Simple': [], 'Complex': []},
        'EP': {'': []},
        'EA': {'': []}
    }
    
    for class_name in data.keys():
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
            
        if class_name in ['NE', 'EH']:  # Classes with subtypes
            for subtype in data[class_name].keys():
                subtype_dir = os.path.join(class_dir, subtype)
                if not os.path.exists(subtype_dir):
                    print(f"Warning: {subtype_dir} not found")
                    continue
                
                files = [f for f in os.listdir(subtype_dir) 
                        if f.upper().endswith('.JPG') and is_valid_filename(f)]
                files = [os.path.join(subtype_dir, f) for f in files]
                data[class_name][subtype].extend(files)
                print(f"Found {len(files)} valid files in {class_name}/{subtype}")
        else:  # Classes without subtypes
            files = [f for f in os.listdir(class_dir) 
                    if f.upper().endswith('.JPG') and is_valid_filename(f)]
            files = [os.path.join(class_dir, f) for f in files]
            data[class_name][''].extend(files)
            print(f"Found {len(files)} valid files in {class_name}")
    
    return data

def create_split_dirs(output_dir):
    for split in ['train', 'test']:
        # Create split directories
        for class_name in ['NE', 'EH', 'EP', 'EA']:
            if class_name in ['NE', 'EH']:
                subtypes = ['Follicular', 'Luteal', 'Menstrual'] if class_name == 'NE' else ['Simple', 'Complex']
                for subtype in subtypes:
                    os.makedirs(os.path.join(output_dir, split, class_name, subtype), exist_ok=True)
            else:
                os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

def split_and_copy(image_data, output_dir, test_ratio=0.2):
    split_info = {'train': {}, 'test': {}}
    manifest_data = []
    
    for class_name, subtypes in image_data.items():
        split_info['train'][class_name] = {}
        split_info['test'][class_name] = {}
        
        for subtype, files in subtypes.items():
            if not files:
                print(f"Warning: No files for {class_name}/{subtype}")
                continue
                
            # Shuffle and split
            random.shuffle(files)
            split_idx = int(len(files) * (1 - test_ratio))
            train_files = files[:split_idx]
            test_files = files[split_idx:]
            
            # Record counts
            split_info['train'][class_name][subtype] = len(train_files)
            split_info['test'][class_name][subtype] = len(test_files)
            
            # Copy files and create manifest entries
            dst_base = class_name if subtype == '' else os.path.join(class_name, subtype)
            
            print(f"Copying {len(train_files)} training and {len(test_files)} test files for {class_name}/{subtype}")
            
            # Handle training files
            for src in train_files:
                dst = os.path.join(output_dir, 'train', dst_base, os.path.basename(src))
                shutil.copy2(src, dst)
                manifest_data.append({
                    'file_path': os.path.join('train', dst_base, os.path.basename(src)),
                    'class': class_name,
                    'subtype': subtype if subtype else '',
                    'split': 'train'
                })
            
            # Handle test files
            for src in test_files:
                dst = os.path.join(output_dir, 'test', dst_base, os.path.basename(src))
                shutil.copy2(src, dst)
                manifest_data.append({
                    'file_path': os.path.join('test', dst_base, os.path.basename(src)),
                    'class': class_name,
                    'subtype': subtype if subtype else '',
                    'split': 'test'
                })
    
    # Save split info
    with open(os.path.join(output_dir, 'split_info.json'), 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)
        
    # Save manifest
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(os.path.join(output_dir, 'split_manifest.csv'), index=False)

def main():
    # Load configuration
    config = load_config()
    data_dir = config['paths']['dataset_root']
    output_dir = config['paths']['split_output']
    test_size = config['split']['test_size']
    random_seed = config['split']['random_seed']
    
    print(f"\nStarting dataset split:")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Test size: {test_size}")
    print(f"Random seed: {random_seed}\n")
    
    # Set random seed
    random.seed(random_seed)
    
    # Get list of image files
    image_data = list_images(data_dir)
    
    # Create output directories
    create_split_dirs(output_dir)
    
    # Split dataset and copy files
    split_and_copy(image_data, output_dir, test_size)
    
    print("\nDataset split complete!")
    print("Check split_info.json for details.")

if __name__ == "__main__":
    main()