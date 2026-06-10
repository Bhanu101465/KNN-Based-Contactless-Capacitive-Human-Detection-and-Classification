import cv2
import numpy as np
import csv
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

names = np.load("names.npy", allow_pickle=True).item()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

marked_names = set()

def mark_attendance(name):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    already_marked = False

    with open("attendance.csv", "r") as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) > 0:

                if row[0] == name and row[1] == today:
                    already_marked = True
                    break

    if not already_marked:

        with open("attendance.csv", "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([name, today, current_time])

        print(f"{name} attendance marked")

    else:

        print(f"{name} already marked today")

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 70:

            name = names[label]

            color = (0,255,0)

            text = f"{name}"

            if name not in marked_names:

                mark_attendance(name)

                marked_names.add(name)

        else:

            text = "Unknown"

            color = (0,0,255)

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            color,
            2
        )

        cv2.putText(
            frame,
            text,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()