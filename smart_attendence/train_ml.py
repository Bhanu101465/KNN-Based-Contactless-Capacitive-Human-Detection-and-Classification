import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

import joblib

# ---------------- LOAD DATASET ----------------

data = pd.read_csv(
    "sensor_dataset.csv",
    header=None,
    names=[
        "Average",
        "Max",
        "Min",
        "Range",
        "Variance",
        "Label"
    ]
)

# ---------------- FEATURES ----------------

X = data.iloc[:, 0:5]

# ---------------- LABELS ----------------

y = data["Label"]

# ---------------- SPLIT DATA ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- CREATE MODEL ----------------

model = KNeighborsClassifier(n_neighbors=3)

# ---------------- TRAIN MODEL ----------------

model.fit(X_train, y_train)

# ---------------- TEST MODEL ----------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# ---------------- SAVE MODEL ----------------

joblib.dump(model, "human_detector.pkl")

print("Model saved successfully.")