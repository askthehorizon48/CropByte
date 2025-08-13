import tensorflow as tf
import json
import os

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "C:/Users/aadis/OneDrive/Documents/GitHub/CropByte/data/PlantVillage"


def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_dataset():
    # Load dataset using image_dataset_from_directory
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    # Save label names to labels.json
    label_names = train_ds.class_names
    label_map = {i: name for i, name in enumerate(label_names)}
    with open('labels.json', 'w') as f:
        json.dump(label_map, f)
    print("✅ Saved labels.json")

    # Apply preprocessing
    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, len(label_names)


if __name__ == "__main__":
    train_ds, val_ds, num_classes = load_dataset()
    print(f"✅ Loaded {num_classes} classes")
