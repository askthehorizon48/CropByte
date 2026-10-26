import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from preprocess import IMG_SIZE, load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "PlantVillage"
MODEL_PATH = ROOT / "cropbyte_model.tflite"
LABELS_PATH = ROOT / "cropbyte_labels.json"
METADATA_PATH = ROOT / "cropbyte_model.json"
PREPROCESSING = "mobilenet_v2_preprocess_input"


def train(data_dir: Path, epochs: int = 10) -> None:
    train_ds, val_ds, class_names = load_dataset(data_dir)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    features = GlobalAveragePooling2D()(base_model.output)
    features = Dense(128, activation="relu")(features)
    predictions = Dense(len(class_names), activation="softmax")(features)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    MODEL_PATH.write_bytes(tflite_model)
    LABELS_PATH.write_text(json.dumps(class_names, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "input_size": IMG_SIZE,
        "preprocessing": PREPROCESSING,
        "classes": class_names,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {MODEL_PATH.name}, {LABELS_PATH.name}, and {METADATA_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CropByte classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"PlantVillage folder (default: {DATA_DIR})",
    )
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.data_dir, args.epochs)
