"""Train a cat vs dog image classifier.

Reads images from dataset/train (split into a training and validation set),
checks the finished model against dataset/test, and saves the result to
model.keras so predict.py can use it.

Run:  python train.py
"""

import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Folders, relative to this file, so the script works from any directory.
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

# Every image is resized to this before it reaches the network. Match it to
# the real size of your pictures: upscaling small images adds no detail, it
# just makes training slower and harder. CIFAR-10 pictures are 32x32, so this
# suits download_dataset.py. Raise it to (180, 180) for full-size photos.
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
# One epoch is one pass over every training picture. EarlyStopping below
# halts sooner if the model stops improving, so a high number is safe.
EPOCHS = 40
VALIDATION_SPLIT = 0.2
SEED = 123


def count_images(folder):
    """Count image files inside a folder and all of its subfolders."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def check_dataset():
    """Stop early with a clear message if the dataset folders are empty."""
    if not TRAIN_DIR.exists():
        raise SystemExit(f"Missing folder: {TRAIN_DIR}")

    train_count = count_images(TRAIN_DIR)
    if train_count == 0:
        raise SystemExit(
            f"No images found in {TRAIN_DIR}.\n"
            "Add pictures first, for example:\n"
            "  dataset/train/cat/cat001.jpg\n"
            "  dataset/train/dog/dog001.jpg\n"
            "Aim for at least a few hundred images per class."
        )

    print(f"Found {train_count} training images in {TRAIN_DIR}")
    if TEST_DIR.exists():
        print(f"Found {count_images(TEST_DIR)} test images in {TEST_DIR}")


def load_datasets():
    """Build the training and validation datasets from dataset/train."""
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # Keep the next batch ready while the current one trains.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes):
    """A small convolutional network, built from scratch."""
    model = keras.Sequential(
        [
            layers.Input(shape=IMAGE_SIZE + (3,)),
            # Augmentation makes the small dataset look bigger by flipping,
            # rotating and zooming images slightly. It only runs while training.
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            # Pixels arrive as 0-255; the network learns better from 0-1.
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            # Dropout randomly ignores some values, which reduces memorising.
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    check_dataset()
    train_ds, val_ds, class_names = load_datasets()

    model = build_model(len(class_names))
    model.summary()

    print("\nTraining...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            # Stop once validation accuracy stops improving, and keep the
            # best version of the model rather than the last one.
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
    )

    # Final score on images the model never saw during training.
    if TEST_DIR.exists() and count_images(TEST_DIR) > 0:
        test_ds = keras.utils.image_dataset_from_directory(
            TEST_DIR,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        loss, accuracy = model.evaluate(test_ds)
        print(f"\nTest accuracy: {accuracy:.2%}  (loss {loss:.4f})")
    else:
        print(f"\nSkipped the test step: no images in {TEST_DIR}")

    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved class names to {CLASS_NAMES_PATH}")
    print("Now try:  python predict.py path/to/your/image.jpg")


if __name__ == "__main__":
    main()
