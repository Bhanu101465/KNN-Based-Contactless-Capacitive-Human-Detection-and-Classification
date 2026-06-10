# Smart Attendance System
### Capacitive Proximity-Based Attendance with ML Signal Classification & Face Recognition

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Arduino](https://img.shields.io/badge/Arduino-UNO-teal?logo=arduino)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

A contactless smart attendance system that combines **capacitive proximity sensing**, **machine learning signal classification**, and **LBPH face recognition** to automate attendance marking without physical interaction.

The system first uses an Arduino-based capacitive sensor to detect presence. An ML classifier then determines whether the detected object is a human or non-human. Only upon human confirmation does the camera activate for face recognition and attendance logging — eliminating false triggers and protecting privacy.

---

## Architecture

```
Arduino Sensor → Serial (115200 baud)
       ↓
Feature Engineering (Mean, Max, Min, Range, Variance)
       ↓
ML Classifier (Decision Tree) → Human / NonHuman / NoObject
       ↓ (Human only)
OpenCV Camera → Haar Cascade Face Detection
       ↓
LBPH Face Recognizer → Known / Unknown
       ↓
CSV Attendance Log (with duplicate prevention)
```

---

## Key Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| KNN (K=3) | 53.1% | Initial baseline |
| Decision Tree | **62.5%** | Selected model (+9.4% over KNN) |

- **Dataset:** 318 real sensor samples across 3 classes (Human, NonHuman, NoObject)
- **Features:** 5 engineered statistical features from raw ADC signals
- **Face recognition threshold:** Confidence < 70 (LBPH distance metric)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Hardware | Arduino UNO, LM358 Signal Conditioning Circuit |
| Sensing | Capacitive proximity sensing via analog ADC |
| ML | Scikit-learn (KNN, Decision Tree), NumPy, Pandas |
| Face Detection | OpenCV Haar Cascade |
| Face Recognition | LBPH Face Recognizer (OpenCV) |
| Communication | PySerial (115200 baud) |
| Data Logging | CSV with datetime-based duplicate prevention |

---

## Project Structure

```
smart-attendance-system/
│
├── collect_data.py        # Collects sensor data from Arduino → CSV
├── train_ml.py            # Trains KNN classifier on sensor_dataset.csv
├── train.py               # Trains LBPH face recognizer on dataset/
├── recognize.py           # Standalone face recognition + attendance
├── ml_test.py             # Live ML prediction from sensor stream
├── integrated_system.py   # Sensor + face recognition (threshold-based)
├── final_system_ml.py     # Full system: ML + face recognition pipeline
├── main.py                # Basic face detection test
│
├── sensor_dataset.csv     # Collected sensor training data (318 samples)
├── human_detector.pkl     # Saved KNN model (joblib)
├── trainer.yml            # Trained LBPH face recognizer weights
├── names.npy              # Label → name mapping
├── attendance.csv         # Master attendance log
└── dataset/               # Per-person face image folders for training
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Arduino UNO with capacitive sensing circuit (LM358)
- Webcam

### Install Dependencies

```bash
pip install opencv-contrib-python scikit-learn numpy pandas pyserial joblib
```

> **Note:** `opencv-contrib-python` is required (not `opencv-python`) for LBPH face recognizer support.

---

## Usage

### Step 1 — Collect Sensor Data
Connect Arduino to `COM3` (update port if needed), then run:
```bash
python collect_data.py
# Enter label: Human / NonHuman / NoObject
```
Data is appended to `sensor_dataset.csv` in windows of 150 samples.

### Step 2 — Train the ML Classifier
```bash
python train_ml.py
# Outputs: human_detector.pkl
# Prints: Accuracy on test split
```

### Step 3 — Collect Face Images
Create folders under `dataset/` named after each person:
```
dataset/
  Bhanu/
    1.jpg
    2.jpg
    ...
```

### Step 4 — Train Face Recognizer
```bash
python train.py
# Outputs: trainer.yml, names.npy
```

### Step 5 — Run Full System
```bash
python final_system_ml.py
```
The system will:
1. Read live sensor data from Arduino
2. Classify presence using the ML model
3. On human detection → open camera
4. Recognize face → mark attendance in `attendance_YYYY-MM-DD.csv`
5. Send `SUCCESS` or `ALERT` signal back to Arduino

---

## ML Details

### Feature Engineering
Raw ADC values are noisy single-point readings. Instead of using them directly, a sliding window of **150 samples** extracts:

| Feature | Description |
|---------|-------------|
| Average | Overall signal level |
| Maximum | Peak signal value |
| Minimum | Lowest signal point |
| Range | Max − Min (fluctuation) |
| Variance | Signal instability/spread |

### Model Selection
KNN (K=3) was chosen as the initial classifier for its simplicity and suitability for small datasets. After empirical benchmarking against Decision Tree on the collected data, Decision Tree was found to outperform KNN by **9.4%** in accuracy, particularly improving Human class precision (81% vs 55%).

---

## Future Improvements

- [ ] Replace Decision Tree with Random Forest / Ensemble voting for better noise robustness
- [ ] Replace LBPH with FaceNet or ArcFace for higher recognition accuracy
- [ ] Add a web dashboard for real-time attendance monitoring
- [ ] Extend to multi-sensor array for wider detection range
- [ ] Deploy on Raspberry Pi for standalone embedded operation

---

## Author

**Bhanu Prakash Akkireddi**
B.Tech — Electrical and Computer Engineering
Amrita Vishwa Vidyapeetham

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/bhanu-prakash-akkireddi-970730315/)
[![GitHub](https://img.shields.io/badge/GitHub-Bhanu101465-black?logo=github)](https://github.com/Bhanu101465)

---

## License

This project is licensed under the MIT License.
