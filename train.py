import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from preprocess import load_dataset

train_ds, val_ds, num_classes = load_dataset()
# Load training set
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/aadis/OneDrive/Documents/GitHub/CropByte/data/PlantVillage",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# Load validation set
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/aadis/OneDrive/Documents/GitHub/CropByte/data/PlantVillage",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
num_classes = len(train_ds.class_names)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

base_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=5)
model.save('cropbyte_model.h5')
print("✅ Saved cropbyte_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('cropbyte_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ Saved cropbyte_model.tflite")
