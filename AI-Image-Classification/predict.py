"""Classify one or more images with the trained cat vs dog model.

Run:  python predict.py cat.jpg
      python predict.py photos/one.jpg photos/two.png
      python predict.py            (asks for a path, then keeps asking)

Train the model first with:  python train.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


def model_image_size(model):
    """Read the picture size the model was trained on.

    The size is taken from the model itself rather than written down here,
    so changing IMAGE_SIZE in train.py cannot leave this script out of step.
    """
    # input_shape looks like (None, height, width, 3).
    _, height, width, _ = model.input_shape
    return (height, width)


def load_trained_model():
    """Load the saved model and the class names that go with it."""
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No trained model at {MODEL_PATH}.\n"
            "Run this first:  python train.py"
        )

    model = keras.models.load_model(MODEL_PATH)

    if CLASS_NAMES_PATH.exists():
        class_names = json.loads(CLASS_NAMES_PATH.read_text())
    else:
        # Fall back to folder names if class_names.json is missing.
        train_dir = BASE_DIR / "dataset" / "train"
        class_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        print(f"Warning: {CLASS_NAMES_PATH} is missing, using folder names.")

    return model, class_names, model_image_size(model)


def predict_image(model, class_names, image_path, image_size):
    """Return the predicted label and confidence for one image."""
    image = keras.utils.load_img(image_path, target_size=image_size)
    array = keras.utils.img_to_array(image)

    # The model expects a batch, so turn one image into a batch of one.
    batch = np.expand_dims(array, axis=0)

    scores = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(scores))
    return class_names[best], float(scores[best]), scores


def show_prediction(model, class_names, raw_path, image_size):
    """Print the result for one path, or a clear message if it cannot be read."""
    path = Path(raw_path)
    if not path.exists():
        print(f"{raw_path}: file not found")
        return

    try:
        label, confidence, scores = predict_image(
            model, class_names, path, image_size
        )
    except Exception as error:
        print(f"{path.name}: could not read this image ({error})")
        return

    print(f"\n{path.name} -> {label}  ({confidence:.1%} confident)")
    for name, score in zip(class_names, scores):
        print(f"   {name:<10} {score:.1%}")


def main():
    model, class_names, image_size = load_trained_model()
    paths = sys.argv[1:]

    if paths:
        for raw_path in paths:
            show_prediction(model, class_names, raw_path, image_size)
        return

    # No paths given, so ask for them one at a time.
    print("Image Classifier")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Pictures are resized to {image_size[0]}x{image_size[1]}.")
    print("Type an image path, or 'quit' to exit.\n")

    while True:
        raw_path = input("Image: ").strip().strip('"')
        if raw_path.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not raw_path:
            print("Please type an image path.\n")
            continue

        show_prediction(model, class_names, raw_path, image_size)
        print()


if __name__ == "__main__":
    main()
