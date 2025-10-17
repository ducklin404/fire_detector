import pandas as pd

# Read the original CSV
df = pd.read_csv("100_data.csv")

# Rename and reorder columns
df_new = pd.DataFrame({
    "temparature": df["Temperature"],  # keeping your misspelling consistent
    "humidity": df["humidity"],
    "gasAnalog": df["gasAnalog"],
    "flameDetected": (df["flameAnalog"] < 2000).astype(int),  # example logic: 1 if flameAnalog < 2000
    "fireStatus": df["label"].apply(lambda x: 1 if x == 1 else 0)
})

# Save the new CSV
df_new.to_csv("1000_data_cleaned.csv", index=False)

print("✅ Conversion complete. Saved as output.csv")
