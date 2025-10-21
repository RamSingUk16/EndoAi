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


def build_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                num_main: int = 4,
                num_ne: int = 3,
                num_eh: int = 2,
                backbone_weights: str = 'imagenet') -> tf.keras.Model:
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
    main_out = tf.keras.layers.Dense(num_main, activation='softmax', name='main')(main_logits)

    # NE subtype head
    ne_logits = tf.keras.layers.Dense(128, activation='relu')(x)
    ne_logits = tf.keras.layers.Dropout(0.2)(ne_logits)
    ne_out = tf.keras.layers.Dense(num_ne, activation='softmax', name='ne_sub')(ne_logits)

    # EH subtype head
    eh_logits = tf.keras.layers.Dense(128, activation='relu')(x)
    eh_logits = tf.keras.layers.Dropout(0.2)(eh_logits)
    eh_out = tf.keras.layers.Dense(num_eh, activation='softmax', name='eh_sub')(eh_logits)

    model = tf.keras.Model(inputs=inputs, outputs=[main_out, ne_out, eh_out], name='pathopulse_resnet50')
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
