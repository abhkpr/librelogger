import face_recognition
import os
import pickle

dataset_path = "dataset"
encodings = []
names = []

for person in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, person)
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        image = face_recognition.load_image_file(img_path)
        faces = face_recognition.face_encodings(image)

        if faces:
            encodings.append(faces[0])
            names.append(person)

data = {"encodings": encodings, "names": names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("Face encodings saved!")

