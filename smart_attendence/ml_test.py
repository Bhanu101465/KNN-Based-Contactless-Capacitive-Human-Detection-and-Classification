import serial
import numpy as np
import joblib

# ---------------- LOAD TRAINED MODEL ----------------

model = joblib.load("human_detector.pkl")

# ---------------- SERIAL CONNECTION ----------------

arduino = serial.Serial('COM3', 115200)

# ---------------- PARAMETERS ----------------

window_size = 150

samples = []

print("Live ML prediction started...")

while True:

    try:

        data = arduino.readline().decode().strip()

        value = int(data)

        samples.append(value)

        # When enough samples collected

        if len(samples) >= window_size:

            avg = np.mean(samples)

            maximum = np.max(samples)

            minimum = np.min(samples)

            signal_range = maximum - minimum

            variance = np.var(samples)

            features = [[
                avg,
                maximum,
                minimum,
                signal_range,
                variance
            ]]

            prediction = model.predict(features)

            print(
                "Prediction:",
                prediction[0],
                "| Avg:", round(avg,2),
                "| Range:", signal_range,
                "| Variance:", round(variance,2)
            )

            samples = []

    except:
        pass