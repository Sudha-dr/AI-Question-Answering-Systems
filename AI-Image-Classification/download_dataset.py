"""Fill dataset/ with cat and dog pictures taken from CIFAR-10.

CIFAR-10 is a free set of 60,000 small photos in 10 categories. Keras can
download it in one call, so this script grabs it, keeps only the cat and dog
pictures, and saves them as image files in the folders train.py expects.

One catch: CIFAR-10 pictures are only 32x32 pixels, which is tiny. They are
enough to prove the whole pipeline works, but do not expect great accuracy.
For good results, replace these with full-size photos later.

Run:  python download_dataset.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"

# CIFAR-10 numbers its categories 0-9. These are the two we want.
CIFAR_LABELS = {"cat": 3, "dog": 5}

# How many pictures to save per class. CIFAR-10 has 5000 train and 1000 test
# per category, so these numbers take all of them. Lower them to train faster.
MAX_TRAIN_PER_CLASS = 5000
MAX_TEST_PER_CLASS = 1000


def save_images(images, folder, prefix, limit):
    """Save an array of pictures into folder as prefix000.png, prefix001.png..."""
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, image in enumerate(images[:limit]):
        Image.fromarray(image).save(folder / f"{prefix}{index:04d}.png")
        count += 1
    return count


def extract_split(images, labels, split_name, limit):
    """Pull out the cat and dog pictures from one CIFAR-10 split."""
    # labels arrives shaped like [[3], [5], ...], so flatten it to [3, 5, ...].
    labels = labels.flatten()

    for class_name, cifar_label in CIFAR_LABELS.items():
        chosen = images[labels == cifar_label]
        folder = DATASET_DIR / split_name / class_name
        saved = save_images(chosen, folder, class_name, limit)
        print(f"  {split_name}/{class_name}: {saved} images")


def main():
    print("Downloading CIFAR-10 (about 163 MB the first time)...")
    print("It is cached afterwards, so later runs are instant.\n")

    (train_images, train_labels), (test_images, test_labels) = (
        keras.datasets.cifar10.load_data()
    )

    print("\nSaving cat and dog pictures:")
    extract_split(train_images, train_labels, "train", MAX_TRAIN_PER_CLASS)
    extract_split(test_images, test_labels, "test", MAX_TEST_PER_CLASS)

    print(f"\nDone. Images are in {DATASET_DIR}")
    print("Next:  python train.py")


if __name__ == "__main__":
    main()
