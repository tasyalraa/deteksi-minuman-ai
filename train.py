import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# =========================
# DATASET
# =========================
DATASET_DIR = "dataset_clean"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# CEK DATASET
# =========================
if not os.path.exists(DATASET_DIR):
    raise ValueError("Dataset tidak ditemukan!")

# =========================
# DATA AUGMENTATION
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# =========================
# SAVE CLASS INDEX (PENTING)
# =========================
print("Class indices:", train_data.class_indices)

os.makedirs("models", exist_ok=True)

with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# =========================
# MODEL (TRANSFER LEARNING)
# =========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAINING
# =========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# =========================
# SAVE MODEL
# =========================
model.save("models/model_minuman.h5")

print("Training selesai")