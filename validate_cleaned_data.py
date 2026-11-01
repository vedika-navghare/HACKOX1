
import pandas as pd

FILE = "Accdataset_cleaned_for_RoadSafeAI.csv"

df = pd.read_csv(FILE)

col = "Accident_Location_A_Chainage_km"
df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n===== SUSPICIOUS CHAINAGE VALUES =====")

# Display the 20 largest values
print(
    df.nlargest(20, col)[
        [
            "Accident_Index",
            "Accident_Location_A",
            col,
            "Accident_Location_A_Chainage_km_RoadSide",
            "Accident_Severity_C",
        ]
    ].to_string(index=False)
)

print("\n===== VALUES ABOVE 1,000 =====")

suspicious = df[df[col] > 1000]

if suspicious.empty:
    print("No values above 1,000.")
else:
    print(
        suspicious[
            [
                "Accident_Index",
                "Accident_Location_A",
                col,
                "Accident_Severity_C",
            ]
        ].to_string(index=False)
    )

print("\n===== MISSING CHAINAGE RECORDS =====")

missing = df[df[col].isna()]

if missing.empty:
    print("No missing chainage values.")
else:
    print(
        missing[
            [
                "Accident_Index",
                "Accident_Location_A",
                "Date",
                "Accident_Severity_C",
            ]
        ].to_string(index=False)
    )

