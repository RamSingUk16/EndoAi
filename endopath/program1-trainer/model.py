"""
model.py

Builds a multi-head ResNet50-based model for PathoPulse trainer.

Outputs:
- main: 4-class softmax (NE/EP/EH/EA)
- ne_sub: 3-class softmax (only meaningful for NE samples)
- eh_sub: 2-class softmax (only meaningful for EH samples)

Provides a helper to unfreeze the top half of the backbone for fine-tuning.
"""

from typing import Tuple
import tensorflow as tf
from constants import CLASS_ORDER, NE_SUBTYPES, EH_SUBTYPES


def build_model(config) -> tf.keras.Model:
    """Build the multi-head PathoPulse model.
    
    Args:
        config: Dictionary containing model configuration
            
    Returns:
        tf.keras.Model with three output heads
    """
    input_shape = config['model']['input_size']
    backbone_weights = config['model'].get('pretrained', 'imagenet')
    
    # Number of classes for each head
    num_main = len(CLASS_ORDER)  # NE, EP, EH, EA
    num_ne = len(NE_SUBTYPES)    # Follicular, Luteal, Menstrual
    num_eh = len(EH_SUBTYPES)    # Simple, Complex
    inputs = tf.keras.Input(shape=input_shape, name='input_image')

    # Backbone
    backbone = tf.keras.applications.ResNet50(include_top=False,
                                              weights=backbone_weights,
                                              input_tensor=inputs,
                                              pooling='avg')

    x = backbone.output  # global pooled features

    # Shared dense layer (small)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Main head
    main_logits = tf.keras.layers.Dense(256, activation='relu')(x)
    main_logits = tf.keras.layers.Dropout(0.3)(main_logits)
    main_out = tf.keras.layers.Dense(num_main, activation='softmax', name='main_output')(main_logits)

    # NE subtype head
    ne_logits = tf.keras.layers.Dense(128, activation='relu')(x)
    ne_logits = tf.keras.layers.Dropout(0.2)(ne_logits)
    ne_out = tf.keras.layers.Dense(num_ne, activation='softmax', name='ne_output')(ne_logits)

    # EH subtype head
    eh_logits = tf.keras.layers.Dense(128, activation='relu')(x)
    eh_logits = tf.keras.layers.Dropout(0.2)(eh_logits)
    eh_out = tf.keras.layers.Dense(num_eh, activation='softmax', name='eh_output')(eh_logits)

    model = tf.keras.Model(inputs=inputs, outputs=[main_out, ne_out, eh_out], name='pathopulse_resnet50')
    
    # Configure model with loss weights
    losses = {
        'main_output': 'sparse_categorical_crossentropy',
        'ne_output': 'sparse_categorical_crossentropy',
        'eh_output': 'sparse_categorical_crossentropy'
    }
    loss_weights = {
        'main_output': 1.0,
        'ne_output': 0.5,
        'eh_output': 0.5
    }
    metrics = {
        'main_output': ['accuracy'],
        'ne_output': ['accuracy'],
        'eh_output': ['accuracy']
    }
    model.compile(
        optimizer='adam',
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    return model


def unfreeze_top_half(model: tf.keras.Model) -> None:
    """Unfreeze the top half of the backbone layers (by number) for fine-tuning.

    This helper assumes the model was built with a backbone from a Keras application.
    """
    # Find backbone layers by name heuristic
    backbone_layers = [l for l in model.layers if l.__class__.__name__.startswith('Conv') or l.name.startswith('res')]
    # Fallback: use all layers and unfreeze top half
    all_layers = model.layers
    n = len(all_layers)
    half = n // 2
    for i, layer in enumerate(all_layers):
        layer.trainable = True if i >= half else False
