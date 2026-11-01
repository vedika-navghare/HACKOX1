import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_FILE = "roadsafe_model.joblib"

print("Loading RoadSafe AI model...")

saved = joblib.load(MODEL_FILE)
model = saved["model"]

# Load the same dataset used by the current training script.
df = pd.read_csv("ETP_4_New_Data_Accidents.csv")

df = df.dropna(subset=["Accident_Severity_C"]).copy()

df["Date"] = pd.to_datetime(
    df["Date"], dayfirst=True, errors="coerce"
)

df["Hour"] = pd.to_datetime(
    df["Time_of_Accident"],
    format="mixed",
    errors="coerce",
).dt.hour

df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

features = saved["features"]

X = df[features].copy()
y = df["Accident_Severity_C"].astype(str)

# IMPORTANT:
# This report evaluates the saved model on the complete dataset.
# It is NOT an independent test-set evaluation.
predictions = model.predict(X)

print("\n========== ROADSAFE AI MODEL REPORT ==========")
print("Records evaluated:", len(df))
print("Accuracy:", round(accuracy_score(y, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y, predictions), 4),
)

print("\nClassification report:")
print(classification_report(y, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y, predictions))

print("\nReport complete.")

