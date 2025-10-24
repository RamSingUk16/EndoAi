"""
quick_save_model.py

Build and save an initial (untrained) PathoPulse model to models/model_<VERSION>.h5.
Useful for backend smoke tests when you need a valid .h5 quickly.
"""

import os
import yaml
from pathlib import Path
from constants import VERSION
from model import build_model


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    models_dir = config.get('paths', {}).get('models_output', 'models')
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(models_dir, f"model_{VERSION}.h5")

    print("Building model...")
    model = build_model(config)

    print(f"Saving initial model to {model_path} ...")
    model.save(model_path)
    print("Done.")


if __name__ == "__main__":
    main()
