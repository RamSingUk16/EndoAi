"""Generate dummy images for testing."""
import os
import numpy as np
from PIL import Image


def create_dummy_image(size=(224, 224)):
    """Create a random RGB image."""
    img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(img)


def save_dummy_images():
    """Save dummy images in the dataset hierarchy."""
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Create dummy images for each class/subtype
    for ne_subtype in ['Follicular', 'Luteal', 'Menstrual']:
        path = os.path.join(base_dir, 'NE', ne_subtype)
        for i in range(5):  # 5 images per subtype
            img_path = os.path.join(path, f'image_{i+1}.jpg')
            img = create_dummy_image()
            img.save(img_path, 'JPEG')

    for eh_subtype in ['Simple', 'Complex']:
        path = os.path.join(base_dir, 'EH', eh_subtype)
        for i in range(5):
            img_path = os.path.join(path, f'image_{i+1}.jpg')
            img = create_dummy_image()
            img.save(img_path, 'JPEG')

    for main_class in ['EP', 'EA']:
        path = os.path.join(base_dir, main_class)
        for i in range(5):
            img_path = os.path.join(path, f'image_{i+1}.jpg')
            img = create_dummy_image()
            img.save(img_path, 'JPEG')


if __name__ == '__main__':
    save_dummy_images()