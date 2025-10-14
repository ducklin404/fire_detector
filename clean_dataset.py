import pandas as pd
import re

INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "cleaned_dataset.csv"


df = pd.read_csv(INPUT_FILE)
df = df.dropna(axis=1, how="all")
df = df.dropna()

def clean_temperature(value):
    s = str(value)
    s = re.sub(r'[^0-9\.]', '', s)
    parts = s.split('.')
    if len(parts) > 2:
        s = parts[0] + '.' + ''.join(parts[1:])
    return s


df['temparature'] = df['temparature'].apply(clean_temperature).astype(float)
df['flameDetected'] = df['flameDetected'].astype(str).str.strip().str.upper().map({'TRUE': 1, 'FALSE': 0})
df['fireStatus'] = df['fireStatus'].astype(str).str.strip().str.upper().map({'TRUE': 1, 'FALSE': 0})



df.to_csv(OUTPUT_FILE, index=False)