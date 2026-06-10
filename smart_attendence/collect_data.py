import serial
import csv
import numpy as np

arduino = serial.Serial('COM3', 115200)

label = input("Enter label (Human/NonHuman/NoObject): ")

filename = "sensor_dataset.csv"

window_size = 150

samples = []

with open(filename, "a", newline="") as file:

    writer = csv.writer(file)

    print("Collecting data... CTRL+C to stop")

    try:

        while True:

            data = arduino.readline().decode().strip()

            try:

                value = int(data)

                samples.append(value)

                if len(samples) >= window_size:

                    avg = np.mean(samples)

                    maximum = np.max(samples)

                    minimum = np.min(samples)

                    signal_range = maximum - minimum

                    variance = np.var(samples)

                    writer.writerow([
                        avg,
                        maximum,
                        minimum,
                        signal_range,
                        variance,
                        label
                    ])

                    print(
                        avg,
                        maximum,
                        minimum,
                        signal_range,
                        variance,
                        label
                    )

                    samples = []

            except:
                pass

    except KeyboardInterrupt:

        print("Stopped.")