"""Inference worker for processing endometrial pathology images (strict models).

This module loads real models only (Program 3 or Program 1) and never uses a
mock fallback for predictions. The selected model can be overridden per-case.
"""
import logging
from datetime import datetime
from typing import Optional, List
import os
import glob
import numpy as np

from .db import get_conn
from .config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loader that supports multiple model keys and formats (.keras, .h5).

    Strict: if a model is not found or fails to load, an exception is raised.
    """

    def __init__(self):
        self._models = {}

    def _discover_model_path(self, key: str) -> Optional[str]:
        base_dir = settings.BASE_DIR  # endopath/endoserver
        candidates: List[str] = []

        try:
            if key == 'program3':
                p3_dir = os.path.normpath(os.path.join(base_dir, "..", "program3-trainer", "outputs"))
                if os.path.isdir(p3_dir):
                    candidates.extend(glob.glob(os.path.join(p3_dir, "*.keras")))
                    candidates.extend(glob.glob(os.path.join(p3_dir, "*.h5")))
            elif key == 'program1':
                p1_dir = os.path.normpath(os.path.join(base_dir, "..", "program1-trainer", "models"))
                if os.path.isdir(p1_dir):
                    candidates.extend(glob.glob(os.path.join(p1_dir, "*.h5")))

            # STRICT: no generic fallback when a specific key is requested

            if not candidates:
                return None
            candidates.sort(key=os.path.getmtime, reverse=True)
            return candidates[0]
        except Exception as e:
            logger.error(f"Model discovery failed for key={key}: {e}")
            return None

    def get(self, key: str):
        if key in self._models:
            return self._models[key]
        # For TF 2.15, use standalone keras package
        try:
            from keras.models import load_model as _k_load_model
            def _load(path):
                # Keras 3: disable compile/safe_mode to avoid deserialization issues
                return _k_load_model(path, compile=False, safe_mode=False)
        except ImportError:
            import tensorflow as tf
            def _load(path):
                return tf.keras.models.load_model(path, compile=False)
        model_path = self._discover_model_path(key)
        if not model_path:
            raise FileNotFoundError(f"Requested model '{key}' not found on server")
        logger.info(f"Loading model for key={key} from {model_path}")
        model = _load(model_path)
        self._models[key] = model
        return model


# Global model instance
_model_loader = ModelLoader()


def preprocess_image(image_data: bytes) -> Optional[np.ndarray]:
    """Preprocess JPEG bytes into a normalized model input (224x224, ImageNet)."""
    try:
        from PIL import Image
        import io

        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        return None


def run_inference(case_id: str, model_override: Optional[str] = None):
    """Run inference on a case and update DB with results (strict model usage)."""
    conn = get_conn()
    cur = conn.cursor()

    try:
        # Read case
        cur.execute('SELECT image_data, gradcam_requested, model FROM cases WHERE id = ?', (case_id,))
        row = cur.fetchone()
        if not row:
            logger.error(f"Case {case_id} not found")
            return

        image_data = row['image_data']
        gradcam_requested = row['gradcam_requested']
        # STRICT: must use explicit request or stored case model, no default fallback
        if model_override:
            model_key = model_override
        elif row['model']:
            model_key = row['model']
        else:
            raise ValueError("No model specified for case and no override provided")

        # Mark processing
        cur.execute('UPDATE cases SET status = ? WHERE id = ?', ('processing', case_id))
        conn.commit()

        # Preprocess
        processed = preprocess_image(image_data)
        if processed is None:
            raise RuntimeError("preprocess failed")

        # Predict
        model = _model_loader.get(model_key)
        predictions = model.predict(processed, verbose=0)

        if isinstance(predictions, list):
            main_pred = predictions[0][0]
            ne_pred = predictions[1][0] if len(predictions) > 1 else None
            eh_pred = predictions[2][0] if len(predictions) > 2 else None
        else:
            main_pred = predictions[0]
            ne_pred = None
            eh_pred = None

        class_labels = ['NE', 'EH', 'EP', 'EA']
        ne_subtypes_full = ['None', 'Follicular', 'Luteal', 'Menstrual']
        eh_subtypes_full = ['None', 'Simple', 'Complex']

        main_idx = int(np.argmax(main_pred))
        prediction = class_labels[main_idx]
        confidence = float(main_pred[main_idx])

        if prediction == 'NE' and ne_pred is not None:
            sub_idx = int(np.argmax(ne_pred))
            sub_label = ne_subtypes_full[sub_idx]
            if sub_label != 'None':
                prediction = f"{prediction} ({sub_label})"
        elif prediction == 'EH' and eh_pred is not None:
            sub_idx = int(np.argmax(eh_pred))
            sub_label = eh_subtypes_full[sub_idx]
            if sub_label != 'None':
                prediction = f"{prediction} ({sub_label})"

        # GradCAM (optional)
        gradcam_data = None
        if gradcam_requested in ['auto', 'on']:
            gradcam_data = generate_gradcam(model, processed, image_data)

        # Write results
        cur.execute('''
            UPDATE cases
            SET status = ?, prediction = ?, confidence = ?, gradcam_data = ?, processed_at = ?
            WHERE id = ?
        ''', (
            'completed', prediction, confidence, gradcam_data, datetime.utcnow().isoformat(), case_id
        ))
        conn.commit()

    except Exception as e:
        logger.error(f"Inference failed for case {case_id}: {e}")
        try:
            # Avoid storing megabyte-long error messages
            err = str(e)
            if len(err) > 500:
                err = err[:500] + '...'
            cur.execute('''
                UPDATE cases SET status = ?, prediction = ?, confidence = ?, processed_at = ? WHERE id = ?
            ''', ('failed', f'error: {err}', 0.0, datetime.utcnow().isoformat(), case_id))
            conn.commit()
        except Exception:
            conn.rollback()


def generate_gradcam(model, image: np.ndarray, original_image_data: bytes = None) -> Optional[bytes]:
    """Generate GradCAM visualization for the prediction (best effort)."""
    try:
        import tensorflow as tf
        from PIL import Image
        import io

        # Find last conv layer
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if not last_conv_layer_name:
            return None

        model_outputs = model.output
        if isinstance(model_outputs, list):
            main_output = model_outputs[0]
        else:
            main_output = model_outputs

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, main_output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if len(predictions.shape) > 1:
                class_idx = tf.argmax(predictions[0])
                class_channel = predictions[0, class_idx]
            else:
                class_idx = tf.argmax(predictions)
                class_channel = predictions[class_idx]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = np.squeeze(heatmap)
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap, mode='L').resize((224, 224), Image.BILINEAR)

        # Colorize
        from matplotlib import cm
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(np.array(heatmap_img) / 255.0)
        heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])
        overlay_img = Image.fromarray(heatmap_colored)

        # Compose on original
        if original_image_data:
            original_img = Image.open(io.BytesIO(original_image_data))
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            original_img = original_img.resize((224, 224))
        else:
            original_img = Image.fromarray(np.uint8(np.clip(image[0] * 255, 0, 255)))

        blended = Image.blend(original_img.convert('RGB'), overlay_img, alpha=0.4)
        buffer = io.BytesIO()
        blended.save(buffer, format='JPEG', quality=90)
        return buffer.getvalue()
    except Exception:
        return None


async def process_case_background(case_id: str):
    """Background task wrapper to process a case."""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_inference, case_id)
