# Training Model for Production

## Current Status

Training is running with production-ready settings:
- **Batch size**: 32
- **Initial epochs**: 10 (backbone frozen)
- **Fine-tune epochs**: 20 (top layers unfrozen)
- **Total training**: ~30 epochs with early stopping

## Monitor Training Progress

While training runs in the background, you can monitor progress by checking:

```powershell
cd endopath/program1-trainer

# Check if training is still running
Get-Process | Where-Object {$_.ProcessName -like "python*"}

# View training log (if saved)
Get-Content training_log.txt -Tail 20

# Check model metrics after completion
C:\dev\endoui\venv311\Scripts\python.exe check_model_metrics.py
```

## Expected Training Time

With ~2,349 training samples and ~590 validation samples:
- **Per epoch**: ~5-10 minutes (CPU)
- **Total time**: ~2-5 hours for full training

To speed up training, you can:
1. Use GPU acceleration (requires CUDA/cuDNN for TF 2.15)
2. Reduce epochs (but may impact accuracy)
3. Increase batch size (if memory allows)

## Evaluating Model Confidence

After training completes, run:

```powershell
C:\dev\endoui\venv311\Scripts\python.exe check_model_metrics.py
```

This will show:
- **Training accuracy**: How well the model learned from training data
- **Validation accuracy**: How well it generalizes to unseen data
- **Production readiness**: Assessment based on validation accuracy
  - ≥90%: Excellent, production-ready
  - 80-90%: Good, suitable with monitoring
  - 70-80%: Moderate, needs improvement
  - <70%: Low, requires significant work

## Using the Trained Model

Once training completes successfully:

1. **Model file location**: `endopath/program1-trainer/models/model_v1.0.0.h5`
2. **Metadata**: `endopath/program1-trainer/models/model_v1.0.0_metadata.json`

The backend (`endoserver`) is already configured to:
- Load the latest `.h5` from `endopath/program1-trainer/models`
- Run inference with the trained model
- Generate GradCAM visualizations

## Quick Test After Training

```powershell
# Start the backend server with TensorFlow environment
cd C:\dev\endoui
C:\dev\endoui\venv311\Scripts\python.exe -m uvicorn endopath.endoserver.app.main:app --reload

# Upload a test image via the UI at http://localhost:8000
# Check predictions and confidence scores in the results
```

## Next Steps

1. Wait for training to complete (~2-5 hours)
2. Check metrics with `check_model_metrics.py`
3. If validation accuracy ≥80%, proceed to backend testing
4. If accuracy <80%, consider:
   - Increasing epochs in `config.yaml`
   - Adding more training data
   - Reviewing data quality/class balance
