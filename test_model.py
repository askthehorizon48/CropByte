import os
import cv2
import numpy as np
import tensorflow as tf

# 🔧 CONFIG
MODEL_PATH = "cropbyte_model.h5"  # Path to your original TensorFlow model
TEST_IMAGES_DIR = "test_images"  # Folder containing test images
IMG_SIZE = (224, 224)  # Match your model's input size
CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"
]
model = tf.keras.models.load_model(MODEL_PATH)


def load_images(folder):
    images = []
    filenames = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.png')):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0  # Normalize
            images.append(img)
            filenames.append(fname)
    return np.array(images), filenames


def test_model():
    images, filenames = load_images(TEST_IMAGES_DIR)
    for img, fname in zip(images, filenames):
        pred = model.predict(np.expand_dims(img, axis=0))[0]
        class_idx = np.argmax(pred)
        confidence = np.max(pred)
        print(f"{fname}: {CLASS_NAMES[class_idx]} ({confidence:.2f})")


print(model.output_shape)
if __name__ == "__main__":
    test_model()
