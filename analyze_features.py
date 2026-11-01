import pandas as pd

FILE = "Accdataset_validated_for_RoadSafeAI.csv"

df = pd.read_csv(FILE)

print("\n========== ROADSAFE AI: FEATURE ANALYSIS ==========")

# 1. Dataset overview
print("\nDataset shape:", df.shape)

# 2. Target distribution
target = "Accident_Severity_C"

print("\n--- Accident Severity Distribution ---")
print(df[target].value_counts(dropna=False).sort_index())

print("\nSeverity percentages:")
print(
    (df[target].value_counts(normalize=True, dropna=False) * 100)
    .sort_index()
    .round(2)
)

# 3. Missing values
print("\n--- Missing Values (%) ---")
missing = (df.isna().mean() * 100).sort_values(ascending=False)
print(missing[missing > 0].round(2).to_string())

# 4. Potentially useful features
features = [
    "Day_of_Week",
    "Time_of_Accident",
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H",
    "Vehicle_Type_Involved_J_V1",
    "Vehicle_Type_Involved_J_V2",
]

print("\n--- Available Candidate Features ---")
for col in features:
    if col in df.columns:
        print(f"{col}: {df[col].nunique(dropna=True)} unique values")

# 5. Check whether identifiers are unique
if "Accident_Index" in df.columns:
    print("\nUnique accident identifiers:", df["Accident_Index"].nunique())

# 6. Check whether potential post-accident information exists
print("\n--- Potentially Post-Accident Features ---")
for col in [
    "Number_of_Casualties",
    "Number_of_Casualties_GrievousInjury",
    "Number_of_Casualties_MinorInjury",
    "Number_of_Casualties_NotInjured",
]:
    if col in df.columns:
        print(f"{col}: present")

print("\nFeature analysis complete.")

