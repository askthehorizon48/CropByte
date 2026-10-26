import json
from pathlib import Path
from typing import Tuple

import tensorflow as tf


IMG_SIZE = 224
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 123
with (Path(__file__).resolve().parent / "cropbyte_labels.json").open(encoding="utf-8") as labels_file:
    EXPECTED_LABELS = json.load(labels_file)


def preprocess(image, label):
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def load_dataset(
    data_dir: Path,
    image_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"PlantVillage dataset folder not found: {data_dir}. "
            "Run download_dataset.py or provide a folder with the 15 supported classes."
        )

    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    dataset_options = {
        "image_size": (image_size, image_size),
        "batch_size": batch_size,
        "label_mode": "int",
        "class_names": EXPECTED_LABELS,
    }
    if train_dir.is_dir() and val_dir.is_dir():
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            seed=SEED,
            shuffle=True,
            **dataset_options,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            shuffle=False,
            **dataset_options,
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=SEED,
            **dataset_options,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=SEED,
            **dataset_options,
        )

    class_names = EXPECTED_LABELS

    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names
