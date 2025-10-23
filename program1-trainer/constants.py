"""Common constants for the PathoPulse trainer."""

import numpy as np

VERSION = "v1.0.0"

# Model input normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class hierarchies
CLASS_ORDER = ['NE', 'EP', 'EH', 'EA']
NE_SUBTYPES = ['Follicular', 'Luteal', 'Menstrual']
EH_SUBTYPES = ['Simple', 'Complex']