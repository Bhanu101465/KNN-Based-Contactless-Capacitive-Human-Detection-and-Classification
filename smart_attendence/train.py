import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []

names = {}
label_id = 0

for person_name in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_name)

    names[label_id] = person_name

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        faces.append(img)

        labels.append(label_id)

    label_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(labels))

recognizer.save("trainer.yml")

np.save("names.npy", names)

print("Training Complete")