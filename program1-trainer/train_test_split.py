"""
train_test_split.py

This script performs an 80/20 split of the endometrial tissue image dataset,
preserving the hierarchical structure of NE subtypes and EH subtypes.
"""

import os
import shutil
import random
import json
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

def clean_filename(filename: str) -> bool:
    """Check if filename contains problematic characters."""
    problematic = ['²', '⌐', 'ú']
    return not any(c in filename for c in problematic)

def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_image_files(data_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Scan the data directory and collect image files organized by class and subtype.
    
    Returns:
        Dict with structure:
        {
            'NE': {
                'Follicular': ['path1.jpg', 'path2.jpg', ...],
                'Luteal': [...],
                'Menstrual': [...]
            },
            'EH': {
                'Simple': [...],
                'Complex': [...]
            },
            'EP': {'': [...]},  # No subtypes
            'EA': {'': [...]}   # No subtypes
        }
    """
    image_files = {
        'NE': {
            'Follicular': [],
            'Luteal': [],
            'Menstrual': []
        },
        'EH': {
            'Simple': [],
            'Complex': []
        },
        'EP': {'': []},
        'EA': {'': []}
    }
    
    # Walk through the data directory
    for class_name in image_files.keys():
        class_path = Path(data_dir) / class_name
        if not class_path.exists():
            print(f"Warning: {class_path} not found")
            continue
            
        if class_name in ['NE', 'EH']:
            # Handle classes with subtypes
            for subtype in image_files[class_name].keys():
                subtype_path = class_path / subtype
                if not subtype_path.exists():
                    print(f"Warning: {subtype_path} not found")
                    continue
                    
                image_files[class_name][subtype].extend([
                    str(f) for f in subtype_path.glob("*.JPG")
                ])
        else:
            # Handle classes without subtypes
            image_files[class_name][''].extend([
                str(f) for f in class_path.glob("*.JPG")
            ])
    
    return image_files

def create_split_directories(output_dir: str) -> None:
    """Create train and test directories with class/subtype structure."""
    for split in ['train', 'test']:
        for class_name in ['NE', 'EH', 'EP', 'EA']:
            if class_name in ['NE', 'EH']:
                subtypes = ['Follicular', 'Luteal', 'Menstrual'] if class_name == 'NE' else ['Simple', 'Complex']
                for subtype in subtypes:
                    Path(output_dir, split, class_name, subtype).mkdir(parents=True, exist_ok=True)
            else:
                Path(output_dir, split, class_name).mkdir(parents=True, exist_ok=True)
                            image_files[class_name][subtype].extend([str(f) for f in subtype_path.glob("*.JPG") if clean_filename(str(f))])
def copy_files(files: List[str], src_root: str, dst_root: str) -> None:
    """Copy files from source to destination, preserving relative paths."""
                        image_files[class_name][''].extend([str(f) for f in class_path.glob("*.JPG") if clean_filename(str(f))])
        rel_path = os.path.relpath(file_path, src_root)
        dst_path = os.path.join(dst_root, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(file_path, dst_path)

def generate_manifest(split_data: Dict[str, Dict[str, Tuple[List[str], List[str]]]], output_dir: str) -> None:
    """Generate CSV manifest and JSON summary of the split."""
    # Generate CSV manifest
        with open(os.path.join(output_dir, 'split_manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'class', 'subtype', 'split'])
        
        for class_name, subtypes in split_data.items():
            for subtype, (train_files, test_files) in subtypes.items():
                for file_path in train_files:
                    writer.writerow([file_path, class_name, subtype, 'train'])
                for file_path in test_files:
                    writer.writerow([file_path, class_name, subtype, 'test'])
    
    # Generate JSON summary
    summary = {
        'train': {class_name: {
            subtype: len(files) for subtype, (files, _) in subtypes.items()
        } for class_name, subtypes in split_data.items()},
        'test': {class_name: {
            subtype: len(files) for subtype, (_, files) in subtypes.items()
        } for class_name, subtypes in split_data.items()}
    }
    
    with open(os.path.join(output_dir, 'split_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    # Load configuration
    config = load_config()
    data_dir = config['paths']['dataset_root']
    output_dir = config['paths']['split_output']
    test_size = config['split']['test_size']
    random_seed = config['split']['random_seed']
    
    # Set random seed
    random.seed(random_seed)
    
    # Collect image files
    image_files = get_image_files(data_dir)
    
    # Create output directories
    create_split_directories(output_dir)
    
    # Perform split
    split_data = {}
    for class_name, subtypes in image_files.items():
        split_data[class_name] = {}
        for subtype, files in subtypes.items():
            # Shuffle files
            random.shuffle(files)
            # Split
            split_idx = int(len(files) * (1 - test_size))
            train_files = files[:split_idx]
            test_files = files[split_idx:]
            split_data[class_name][subtype] = (train_files, test_files)
            
            # Copy files
            if subtype:  # For classes with subtypes
                train_dst = os.path.join(output_dir, 'train', class_name, subtype)
                test_dst = os.path.join(output_dir, 'test', class_name, subtype)
            else:  # For classes without subtypes
                train_dst = os.path.join(output_dir, 'train', class_name)
                test_dst = os.path.join(output_dir, 'test', class_name)
                
            for f in train_files:
                shutil.copy2(f, train_dst)
            for f in test_files:
                shutil.copy2(f, test_dst)
    
    # Generate manifest and summary
    generate_manifest(split_data, output_dir)
    
    print("Dataset split complete!")
    print("Check split_manifest.csv and split_summary.json for details.")

if __name__ == "__main__":
    main()