import tensorflow_datasets as tfds
import tensorflow as tf
import json

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_dataset():
    # Load PlantVillage dataset
    dataset, info = tfds.load('plant_village', with_info=True, as_supervised=True)
    train_ds = dataset['train']

    # Save label names to labels.json
    label_names = info.features['label'].names
    label_map = {i: name for i, name in enumerate(label_names)}
    with open('labels.json', 'w') as f:
        json.dump(label_map, f)
    print("✅ Saved labels.json")

    # Shuffle and split
    train_size = int(0.8 * info.splits['train'].num_examples)
    val_size = info.splits['train'].num_examples - train_size

    train_ds = train_ds.map(preprocess).shuffle(1000).take(train_size).batch(BATCH_SIZE)
    val_ds = train_ds.map(preprocess).skip(train_size).take(val_size).batch(BATCH_SIZE)

    return train_ds, val_ds, len(label_names)

if __name__ == "__main__":
    train_ds, val_ds, num_classes = load_dataset()
    print(f"✅ Loaded {num_classes} classes")