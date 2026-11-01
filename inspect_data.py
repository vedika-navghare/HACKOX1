import pandas as pd

file_path = "ETP_4_New_Data_Accidents.csv"

df = pd.read_csv(file_path)

print("\nDATASET SIZE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST FIVE ROWS:")
print(df.head().to_string())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATA TYPES:")
print(df.dtypes)

import pandas as pd

df = pd.read_csv("ETP_4_New_Data_Accidents.csv")

print("\nLocation codes:")
print(df["Accident_Location_A"].value_counts().sort_index())

print("\nChainage range:")
print(df["Accident_Location_A_Chainage_km"].describe())

print("\nRoadside codes:")
print(
    df["Accident_Location_A_Chainage_km_RoadSide"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nLocation and chainage examples:")
print(
    df[
        [
            "Accident_Location_A",
            "Accident_Location_A_Chainage_km",
            "Accident_Location_A_Chainage_km_RoadSide",
        ]
    ].head(20).to_string(index=False)
)

