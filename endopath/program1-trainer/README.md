# PathoPulse Trainer (Program 1)

Model training component for the PathoPulse project - an endometrial tissue slide classifier using CNN with hierarchical output.

## Quick Start

1. Ensure you have Python 3.10+ installed
2. Create and activate a virtual environment:
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