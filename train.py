import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping

# =========================
# SETTINGS
# =========================

IMG_SIZE = 224

categories = ["authentic", "tampered"]

data = []
labels = []

# =========================
# LOAD DATASET
# =========================

for category in categories:

    path = os.path.join("dataset", category)

    label = categories.index(category)

    for img in os.listdir(path):

        try:
            img_path = os.path.join(path, img)

            image = cv2.imread(img_path)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            data.append(image)

            labels.append(label)

        except Exception as e:
            print(f"Error loading image {img}: {e}")

# =========================
# PREPARE DATA
# =========================

data = np.array(data, dtype="float32") / 255.0

labels = np.array(labels)

labels = to_categorical(labels, 2)

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

# =========================
# DATA AUGMENTATION
# =========================

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)

# =========================
# LOAD MOBILENETV2
# =========================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# BUILD MODEL
# =========================

model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# EARLY STOPPING
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=4),
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[early_stop]
)

# =========================
# EVALUATE MODEL
# =========================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================

os.makedirs("models", exist_ok=True)

model.save("models/ec8a_mobilenet_model.h5")

print("\nModel saved successfully!")