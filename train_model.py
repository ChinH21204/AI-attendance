import cv2
import os
import numpy as np
from PIL import Image

dataset_path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    faceSamples = []
    names = []

    for imagePath in imagePaths:
        filename = os.path.split(imagePath)[-1]
        name = filename.split('_')[0]
        img = Image.open(imagePath).convert('L')
        img_numpy = np.array(img, 'uint8')

        faceSamples.append(img_numpy)
        names.append(name)

    return faceSamples, names

faces, names = getImagesAndLabels(dataset_path)

unique_names = list(set(names))
ids = [unique_names.index(n) for n in names]

recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')

with open('trainer/labels.txt', 'w') as f:
    for i, name in enumerate(unique_names):
        f.write(f"{i},{name}\n")

print(f"[INFO] Training completed. {len(unique_names)} people trained.")
