import os
import cv2
import numpy as np

IMG_SIZE = 128

data = []
labels = []

categories = ["authentic", "tampered"]

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
            print(f"Error loading {img}: {e}")

data = np.array(data) / 255.0
labels = np.array(labels)

print("Dataset loaded successfully!")
print(f"Images: {len(data)}")