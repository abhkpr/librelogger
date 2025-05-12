import cv2
import face_recognition
import pickle
from datetime import datetime

# Load encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

seen_names = set()
video = cv2.VideoCapture(0)

print("Starting camera. Press 'q' to stop.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for encoding, face in zip(encodings, faces):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = data["names"][match_index]
            seen_names.add(name)

        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Save attendance
now = datetime.now()
filename = f"attendance_{now.strftime('%Y-%m-%d')}.csv"

with open(filename, "w") as f:
    f.write("Name,Date,Time\n")
    for name in seen_names:
        f.write(f"{name},{now.date()},{now.time().strftime('%H:%M:%S')}\n")

print(f"Attendance saved to {filename}")

