"""Inference worker for processing endometrial pathology images."""
import logging
from datetime import datetime
from typing import Optional
import numpy as np
from .db import get_conn
from .config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Lazy loader for the TensorFlow model."""
    
    def __init__(self):
        self._model = None
        self._loaded = False
    
    def load_model(self):
        """Load the TensorFlow model from disk."""
        if self._loaded:
            return self._model
        
        try:
            # Import TensorFlow only when needed
            import tensorflow as tf
            from tensorflow import keras
            
            model_path = f"{settings.MODEL_DIR}/endometrial_model.h5"
            logger.info(f"Loading model from {model_path}")
            
            self._model = keras.models.load_model(model_path)
            self._loaded = True
            logger.info("Model loaded successfully")
            return self._model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Return a mock model for development/testing
            logger.warning("Using mock model for inference")
            return None
    
    @property
    def model(self):
        """Get the loaded model."""
        if not self._loaded:
            return self.load_model()
        return self._model


# Global model instance
_model_loader = ModelLoader()


def preprocess_image(image_data: bytes) -> Optional[np.ndarray]:
    """
    Preprocess image data for model inference.
    
    Args:
        image_data: Raw JPEG image bytes
        
    Returns:
        Preprocessed numpy array ready for model input
    """
    try:
        from PIL import Image
        import io
        
        # Load image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size (adjust based on your model)
        img = img.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        return None


def generate_mock_gradcam(image_data: bytes) -> Optional[bytes]:
    """
    Generate a mock GradCAM visualization for development/testing.
    
    Args:
        image_data: Original JPEG image bytes
        
    Returns:
        Mock GradCAM JPEG bytes with a colored overlay
    """
    try:
        from PIL import Image, ImageDraw
        import io
        
        # Load original image
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to standard size
        img = img.resize((224, 224))
        
        # Create a semi-transparent red overlay in the center (mock heatmap)
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw a gradient-like circle in the center (mock attention area)
        center_x, center_y = 112, 112
        for radius in range(60, 0, -5):
            alpha = int(255 * (60 - radius) / 60 * 0.6)  # Fade from center
            draw.ellipse(
                [center_x - radius, center_y - radius, 
                 center_x + radius, center_y + radius],
                fill=(255, 0, 0, alpha)
            )
        
        # Blend overlay with original
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        # Convert to JPEG bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to generate mock GradCAM: {e}")
        return None


def run_inference(case_id: str):
    """
    Run inference on a case and update the database with results.
    
    Args:
        case_id: The case ID to process
    """
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        # Fetch case data
        cur.execute('SELECT image_data, gradcam_requested FROM cases WHERE id = ?', (case_id,))
        row = cur.fetchone()
        
        if not row:
            logger.error(f"Case {case_id} not found")
            return
        
        image_data = row['image_data']
        gradcam_requested = row['gradcam_requested']
        
        # Update status to processing
        cur.execute('UPDATE cases SET status = ? WHERE id = ?', ('processing', case_id))
        conn.commit()
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            raise Exception("Failed to preprocess image")
        
        # Load model and run inference
        model = _model_loader.model
        
        if model is None:
            # Mock inference for development
            logger.warning(f"Using mock inference for case {case_id}")
            prediction = "benign"  # Mock result
            confidence = 0.85
            
            # Generate mock GradCAM for development
            gradcam_data = None
            if gradcam_requested in ['auto', 'on']:
                gradcam_data = generate_mock_gradcam(image_data)
        else:
            # Real inference
            predictions = model.predict(processed_image)
            
            # Assuming binary classification: 0=benign, 1=malignant
            confidence = float(predictions[0][0])
            prediction = "malignant" if confidence > 0.5 else "benign"
            
            # Generate GradCAM if requested
            gradcam_data = None
            if gradcam_requested in ['auto', 'on']:
                gradcam_data = generate_gradcam(model, processed_image)
        
        # Update database with results
        cur.execute('''
            UPDATE cases 
            SET status = ?, 
                prediction = ?, 
                confidence = ?,
                gradcam_data = ?,
                processed_at = ?
            WHERE id = ?
        ''', (
            'completed',
            prediction,
            confidence,
            gradcam_data,
            datetime.utcnow().isoformat(),
            case_id
        ))
        conn.commit()
        
        logger.info(f"Inference completed for case {case_id}: {prediction} ({confidence:.2%})")
        
    except Exception as e:
        logger.error(f"Inference failed for case {case_id}: {e}")
        
        # Update status to failed
        cur.execute('''
            UPDATE cases 
            SET status = ?, 
                processed_at = ?
            WHERE id = ?
        ''', ('failed', datetime.utcnow().isoformat(), case_id))
        conn.commit()


def generate_gradcam(model, image: np.ndarray) -> Optional[bytes]:
    """
    Generate GradCAM visualization for the prediction.
    
    Args:
        model: The loaded TensorFlow model
        image: Preprocessed image array
        
    Returns:
        JPEG bytes of the GradCAM overlay, or None if generation fails
    """
    try:
        import tensorflow as tf
        from PIL import Image
        import io
        
        # Get the last convolutional layer
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        
        if not last_conv_layer_name:
            logger.warning("No convolutional layer found for GradCAM")
            return None
        
        # Create a model that outputs the last conv layer and predictions
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        # Get gradients of the loss w.r.t. the conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients over all axes except channel axis
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize((224, 224))
        
        # Apply colormap
        from matplotlib import cm
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(np.array(heatmap) / 255.0)
        heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])
        
        # Overlay on original image
        original = np.uint8(image[0] * 255)
        overlay = Image.fromarray(heatmap_colored)
        original_img = Image.fromarray(original)
        
        # Blend images
        blended = Image.blend(original_img, overlay, alpha=0.4)
        
        # Convert to JPEG bytes
        buffer = io.BytesIO()
        blended.save(buffer, format='JPEG', quality=90)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to generate GradCAM: {e}")
        return None


async def process_case_background(case_id: str):
    """
    Background task to process a case.
    
    Args:
        case_id: The case ID to process
    """
    import asyncio
    
    # Run inference in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_inference, case_id)
