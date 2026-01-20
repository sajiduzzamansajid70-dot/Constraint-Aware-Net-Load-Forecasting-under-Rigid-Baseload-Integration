from src.data_loader import DataLoader
from pathlib import Path

loader = DataLoader(Path("data"))
df = loader.load_electricity_data()
print("Columns:", df.columns.tolist())
print("\nFirst row:")
print(df.iloc[0].to_dict())
print("\nData types:")
print(df.dtypes)
print("\nShape:", df.shape)
print("\nDemand stats:")
print(df['demand_mw'].describe())
print("\nGeneration stats:")
print(df['generation_mw'].describe())
