# PathoPulse Trainer (Program 1)

Model training component for the PathoPulse project - an endometrial tissue slide classifier using CNN with hierarchical output.

## Quick Start

1. Ensure you have Python 3.11 installed (TensorFlow 2.15 supports up to Python 3.11 on Windows)
    - Windows (PowerShell):
       - Install via winget: `winget install -e --id Python.Python.3.11`
       - Verify: `py -3.11 --version`
2. Create and activate a virtual environment (recommended name: venv311):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate # Linux/MacOS
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify TensorFlow installation:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```
   Should print version 2.15.*

Note: If your workspace already has a different venv (e.g., Python 3.14), create a new one with Python 3.11 and use that interpreter for training and the backend server so TensorFlow can import correctly.

## Project Structure

```
program1-trainer/
├─ data/                           # Source data folders
│  ├─ NE/{Follicular,Luteal,Menstrual}/
│  ├─ EH/{Simple,Complex}/
│  ├─ EP/
│  └─ EA/
├─ dataset_split/                  # Train/test split output
├─ models/                         # Saved models and artifacts
├─ templates/                      # Report templates
├─ train_test_split.py            # Data splitting script
├─ train.py                       # Main training script
├─ eval.py                        # Evaluation script
├─ gradcam.py                     # Grad-CAM visualization
├─ config.yaml                    # Configuration file
└─ requirements.txt               # Python dependencies
```

## Usage

1. Place your dataset images in the appropriate folders under `data/`
2. Run the train/test split:
   ```bash
   python train_test_split.py
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python eval.py
   ```
5. Generate Grad-CAM visualization for a specific image:
   ```bash
   python gradcam.py --image path/to/img.jpg --out models/sample_gradcam.png
   ```

## Configuration

All configurable parameters are in `config.yaml`. Key settings include:
- Data split ratio and random seed
- Model architecture settings
- Training hyperparameters
- Data augmentation options

## Output

The training process generates:
- Trained model file (`models/model_v1.0.0.h5`)
- Model metadata (`models/metadata_v1.0.0.json`)
- Training and evaluation reports (PDFs)
- Performance plots and visualizations

## Troubleshooting

- TensorFlow fails to import: ensure you are using Python 3.11 and that `tensorflow==2.15.*` is installed in that environment.
- GPU vs CPU: This setup uses the CPU build (`tensorflow-intel`) by default on Windows. For GPU acceleration, install the correct CUDA/cuDNN stack compatible with TF 2.15 and the corresponding `tensorflow` package.