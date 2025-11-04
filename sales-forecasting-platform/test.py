import pandas as pd

# Read the first few rows of the parquet file
df = pd.read_parquet("data/processed/train_features.parquet")

# Print column names and first few rows
print("\nColumn names:")
print("-" * 50)
print(df.columns.tolist())

print("\nFirst 5 rows:")
print("-" * 50)
print(df.head().to_string())

print("\nData types:")
print("-" * 50)
print(df.dtypes)