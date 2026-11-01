import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

# 1. Load real accident data
FILE = "ETP_4_New_Data_Accidents.csv"

df = pd.read_csv(FILE)
print("Dataset loaded:", df.shape)

# 2. Prepare the prediction target
TARGET = "Accident_Severity_C"

df = df.dropna(subset=[TARGET]).copy()

# Exclude identifiers and fields that would cause leakage.
# Date and time are transformed into useful features.
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Hour"] = pd.to_datetime(
    df["Time_of_Accident"], format="mixed", errors="coerce"
).dt.hour

df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Keep inputs that would be available when estimating severity.
FEATURES = [
    "Day_of_Week",
    "Hour",
    "Month",
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Causes_D",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H",
    "Vehicle_Type_Involved_J_V1",
    "Vehicle_Type_Involved_J_V2"
]

X = df[FEATURES].copy()
y = df[TARGET].astype(str)

# Treat coded categories as categories, not continuous measurements.
CATEGORICAL = [
    "Day_of_Week",
    "Month",
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Causes_D",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H",
    "Vehicle_Type_Involved_J_V1",
    "Vehicle_Type_Involved_J_V2"
]

NUMERIC = [
    "Hour",
    "Year",
    "Accident_Location_A_Chainage_km"
]

X["Year"] = df["Year"]
FEATURES.append("Year")

# 3. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Preprocess missing values and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            CATEGORICAL
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            NUMERIC
        )
    ]
)

# 5. Train the ML model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
    n_estimators=150,
    class_weight=None,
    random_state=42,
    n_jobs=-1
))
])

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# 6. Evaluate on held-out records
predictions = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========")
print("Training records:", len(X_train))
print("Test records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nTarget class distribution:")
print(y.value_counts().sort_index())

# 7. Save trained model and feature names
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "target": TARGET
    },
    "roadsafe_model.joblib"
)

print("\nSaved trained model to roadsafe_model.joblib")
print("ML TRAINING COMPLETE")