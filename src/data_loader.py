"""
Constraint-Aware Net Load Forecasting Pipeline
Bangladesh Power System - Peak-Hour Risk-Focused Study

Data loader module for electricity demand and weather data
Strictly aligned with research proposal
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and validate electricity demand and weather data.
    
    Electricity data source: PGCB hourly generation dataset
    Weather data source: BD_weather.csv (daily resolution)
    
    No leakage: All preprocessing learned on training data only.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        # Paths relative to project root
        project_root = self.data_dir.parent.parent
        self.electricity_path = project_root / "pgcb+hourly+generation+dataset+(bangladesh)" / "PGCB_date_power_demand.xlsx"
        self.weather_path = project_root / "High Volume Real-World Weather Data" / "High Volume Real-World Weather Data" / "Weather Datasets" / "Combined Data" / "BD_weather.csv"
        
        logger.info(f"Data paths configured")
        logger.info(f"  Electricity: {self.electricity_path}")
        logger.info(f"  Weather: {self.weather_path}")
    
    def load_electricity_data(self) -> pd.DataFrame:
        """
        Load electricity demand data from PGCB dataset.
        
        Returns:
            DataFrame with hourly demand, generation, and load shedding data
        """
        logger.info("Loading electricity demand data...")
        
        # Try loading the first sheet (index 0)
        df = pd.read_excel(self.electricity_path, sheet_name=0)
        
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Data types:\n{df.dtypes}")
        
        return df
    
    def load_weather_data(self) -> pd.DataFrame:
        """
        Load weather data from Bangladesh meteorological records.
        
        Weather variables: Temperature, Humidity, Rainfall, Sunshine
        Daily resolution. Will align to hourly using forward fill or aggregation.
        
        Returns:
            DataFrame with daily weather observations
        """
        logger.info("Loading weather data...")
        
        df = pd.read_csv(self.weather_path)
        
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")
        logger.info(f"  Unique stations: {len(df['Station'].unique())}")
        
        return df
    
    def prepare_electricity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize electricity data.
        
        - Parse timestamps
        - Ensure hourly resolution
        - Compute served load = demand - load_shedding
        - Sort by time
        
        Args:
            df: Raw electricity data
            
        Returns:
            Cleaned electricity DataFrame with hourly timestamps
        """
        logger.info("Preprocessing electricity data...")
        
        df = df.copy()
        
        # Identify datetime columns - inspect structure first
        logger.info(f"Column names: {df.columns.tolist()}")
        logger.info(f"First row: {df.iloc[0].to_dict()}")
        
        # This will be customized based on actual file structure
        # Placeholder for now
        return df
    
    def prepare_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align weather data to hourly resolution.
        
        - Construct date from Year, Month, Day
        - Aggregate/forward-fill to hourly
        - Handle missing values
        
        Args:
            df: Raw weather data (daily)
            
        Returns:
            Hourly weather DataFrame
        """
        logger.info("Preprocessing weather data...")
        
        df = df.copy()
        
        # Create date column
        df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        
        # For now, aggregate by taking average of same day across stations
        # or select primary station
        logger.info("Converting daily weather to hourly...")
        
        # Expand daily to hourly using forward fill
        df_hourly_list = []
        for station in df['Station'].unique():
            station_data = df[df['Station'] == station].copy()
            station_data = station_data.set_index('date')
            
            # Create hourly index for each day
            date_range = pd.date_range(
                start=station_data.index.min(),
                end=station_data.index.max() + pd.Timedelta(hours=23),
                freq='H'
            )
            
            # Reindex to hourly, forward fill
            hourly = station_data[['Temperature', 'Humidity', 'Rainfall', 'Sunshine']].reindex(date_range, method='ffill')
            hourly['Station'] = station
            hourly['datetime'] = hourly.index
            df_hourly_list.append(hourly.reset_index(drop=True))
        
        return pd.concat(df_hourly_list, ignore_index=True) if df_hourly_list else pd.DataFrame()


if __name__ == "__main__":
    # Test data loading
    base_dir = Path("constraint_aware_net_load/data")
    loader = DataLoader(base_dir)
    
    elec_df = loader.load_electricity_data()
    weather_df = loader.load_weather_data()
    
    print("\n" + "="*80)
    print("DATA INSPECTION COMPLETE")
    print("="*80)
    print(f"\nElectricity data sample:\n{elec_df.head()}")
    print(f"\nWeather data sample:\n{weather_df.head()}")
