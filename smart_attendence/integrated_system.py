import cv2
import numpy as np
import csv
import serial
import time
from datetime import datetime

# ---------------- SERIAL CONNECTION ----------------

arduino = serial.Serial('COM3', 115200)

time.sleep(2)

# ---------------- FACE RECOGNITION SETUP ----------------

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

names = np.load("names.npy", allow_pickle=True).item()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- ATTENDANCE FUNCTION ----------------

def mark_attendance(name):

    today = datetime.now().strftime("%Y-%m-%d")

    filename = f"attendance_{today}.csv"

    current_time = datetime.now().strftime("%H:%M:%S")

    already_marked = False

    try:

        with open(filename, "r") as file:

            reader = csv.reader(file)

            for row in reader:

                if len(row) > 0:

                    if row[0] == name:
                        already_marked = True
                        break

    except FileNotFoundError:

        with open(filename, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(["Name", "Time"])

    if not already_marked:

        with open(filename, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([name, current_time])

        print(f"{name} attendance marked")

    else:

        print(f"{name} already marked today")

# ---------------- MAIN LOOP ----------------

while True:

    if arduino.in_waiting:

        data = arduino.readline().decode().strip()

        print("Arduino:", data)

        # -------- SENSOR DETECTED --------

        if data == "DETECTED":

            print("Starting Face Recognition...")

            cap = cv2.VideoCapture(0)

            recognized = False

            start_time = time.time()

            # Camera stays active for 10 seconds

            while time.time() - start_time < 10:

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

                    # -------- KNOWN PERSON --------

                    if confidence < 70:

                        name = names[label]

                        mark_attendance(name)

                        # Send success signal to Arduino
                        arduino.write(b"SUCCESS\n")

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x+w, y+h),
                            (0,255,0),
                            2
                        )

                        cv2.putText(
                            frame,
                            name,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2
                        )

                        recognized = True

                    # -------- UNKNOWN PERSON --------

                    else:

                        arduino.write(b"ALERT\n")

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x+w, y+h),
                            (0,0,255),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Unknown",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2
                        )

                cv2.imshow("Smart Attendance System", frame)

                if recognized:
                    break

                if cv2.waitKey(1) == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()