import pandas as pd

INPUT_FILE = "Accdataset_cleaned_for_RoadSafeAI.csv"
OUTPUT_FILE = "Accdataset_validated_for_RoadSafeAI.csv"

df = pd.read_csv(INPUT_FILE)

chainage = "Accident_Location_A_Chainage_km"

df[chainage] = pd.to_numeric(df[chainage], errors="coerce")

# These two values are suspicious, but their correct values
# cannot be determined without checking the original source.
suspicious_values = [421160, 438500]

mask = df[chainage].isin(suspicious_values)

print("Suspicious records identified:")
print(
    df.loc[
        mask,
        ["Accident_Index", "Accident_Location_A", chainage]
    ].to_string(index=False)
)

# Preserve the records; mark only their unverified chainage as missing.
df.loc[mask, chainage] = pd.NA

df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning completed!")
print(f"Original rows: {len(pd.read_csv(INPUT_FILE))}")
print(f"Output rows:   {len(df)}")
print(f"Flagged chainage values: {mask.sum()}")
print(f"Remaining missing chainage values: {df[chainage].isna().sum()}")
print(f"Saved as: {OUTPUT_FILE}")

