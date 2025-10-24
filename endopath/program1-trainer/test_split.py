"""
test_split.py - Minimal test script for dataset splitting
"""

import os
from pathlib import Path

def main():
    # Just list all files in the data directory
    data_dir = "data"
    for root, dirs, files in os.walk(data_dir):
        print(f"\nIn directory: {root}")
        print(f"Found {len(files)} files")
        if files:
            print("Sample files:")
            for f in files[:5]:
                print(f"  {f}")

if __name__ == "__main__":
    main()