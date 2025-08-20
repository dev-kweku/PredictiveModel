# check_raw_data.py
import pandas as pd

# Load raw data
df = pd.read_csv('data/raw/Nadmo_cleaned_refined.csv', header=None)

print("Raw data structure:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nFirst 10 rows:")
print(df.head(10))

print("\nColumn names:", list(df.columns))

# Check first row
print("\nFirst row values:")
print(df.iloc[0].values)

# Check if first row contains file info
if df.iloc[0, 0] == 'Nadmo_cleaned_refined.csv':
    print("\nFirst row contains file info, will be skipped")