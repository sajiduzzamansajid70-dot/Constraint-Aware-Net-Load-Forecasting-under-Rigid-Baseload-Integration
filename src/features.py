"""
Feature engineering module for constraint-aware net load forecasting.

Constructs target variable and feature sets following research proposal:
- Target: Net Load(t) = Served Load(t) - Rigid Baseload(t) - Renewable Output(t)
- Features: Lagged net load, calendar features, weather features
- NO LEAKAGE: All preprocessing fit on training data only

Strictly aligned with proposal section 7.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Construct constraint-aware net load target and feature sets.
    
    Target Definition (from proposal 6.3):
        Net Load(t) = Served Load(t) - Rigid Baseload(t) - Renewable Output(t)
        
        where:
        - Served Load = Demand - Load Shedding (actual delivered electricity)
        - Rigid Baseload = fixed scenario (default 2200 MW for nuclear planning)
        - Renewable Output = Solar + Wind (variable supply)
    
    Feature Set (from proposal 7.1):
        1. Lagged net load (1-48 hours, 7 days)
        2. Calendar features (hour, day of week, month, holiday if available)
        3. Weather features (temperature, humidity lagged)
        4. Interaction features (heat stress: temp × humidity if justified)
    """
    
    def __init__(self, rigid_baseload_mw: float = 2200.0):
        """
        Args:
            rigid_baseload_mw: Fixed must-run baseload in MW (default: 2200 for nuclear scenario)
        """
        self.rigid_baseload_mw = rigid_baseload_mw
        self.scalers = {}  # Store scalers fit on training data
        logger.info(f"Feature engineer initialized with rigid baseload = {rigid_baseload_mw} MW")
    
    def construct_net_load(self, 
                          electricity_df: pd.DataFrame,
                          solar_col: str = 'solar',
                          wind_col: str = 'wind') -> pd.DataFrame:
        """
        Construct constraint-aware net load target.
        
        Net Load(t) = Served Load(t) - Rigid Baseload(t) - Renewable Output(t)
        
        Args:
            electricity_df: DataFrame with demand, load_shedding, solar, wind
            solar_col: Column name for solar generation
            wind_col: Column name for wind generation
            
        Returns:
            DataFrame with net_load computed and sorted by time
        """
        df = electricity_df.copy()
        
        # Ensure data is sorted by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Compute served load
        df['served_load'] = df['demand_mw'] - df['load_shedding']
        
        # Compute renewable output
        df['renewable_output'] = df[solar_col].fillna(0) + df[wind_col].fillna(0)
        
        # Compute net load (residual that must be served by flexible generation)
        df['net_load'] = df['served_load'] - self.rigid_baseload_mw - df['renewable_output']
        
        logger.info(f"Net load constructed")
        logger.info(f"  Served load: mean={df['served_load'].mean():.1f} MW")
        logger.info(f"  Renewable output: mean={df['renewable_output'].mean():.1f} MW")
        logger.info(f"  Net load: mean={df['net_load'].mean():.1f} MW, std={df['net_load'].std():.1f} MW")
        logger.info(f"  Net load range: [{df['net_load'].min():.1f}, {df['net_load'].max():.1f}] MW")
        
        return df
    
    def align_weather_to_hourly(self,
                                electricity_df: pd.DataFrame,
                                weather_df: pd.DataFrame,
                                station: str = 'Dhaka') -> pd.DataFrame:
        """
        Align daily weather data to hourly electricity data.
        
        Uses forward fill to broadcast daily values to all hours of that day.
        
        Args:
            electricity_df: Hourly electricity data with datetime
            weather_df: Daily weather data
            station: Which weather station to use (default: 'Dhaka' if exists, else first)
            
        Returns:
            Combined DataFrame with hourly data and aligned weather
        """
        logger.info(f"Aligning weather data to hourly resolution...")
        
        df = electricity_df.copy()
        df['date'] = df['datetime'].dt.date
        
        # Extract date from weather
        weather = weather_df.copy()
        weather['date'] = pd.to_datetime(weather[['Year', 'Month', 'Day']]).dt.date
        
        # Select station
        if station in weather['Station'].unique():
            weather = weather[weather['Station'] == station]
        else:
            # Use first available station
            station = weather['Station'].iloc[0]
            weather = weather[weather['Station'] == station]
            logger.warning(f"Station '{station}' not found, using '{station}' instead")
        
        # Keep only weather features
        weather_features = weather[['date', 'Temperature', 'Humidity', 'Rainfall', 'Sunshine']].drop_duplicates('date')
        
        # Merge on date
        df = df.merge(weather_features, left_on='date', right_on='date', how='left')
        
        # Forward fill any missing values
        df['Temperature'] = df['Temperature'].ffill()
        df['Humidity'] = df['Humidity'].ffill()
        df['Rainfall'] = df['Rainfall'].fillna(0)  # Rainfall is sparse, fill with 0
        df['Sunshine'] = df['Sunshine'].ffill()
        
        logger.info(f"  Weather aligned from station: {station}")
        logger.info(f"  Temperature: {df['Temperature'].notna().sum()} / {len(df)} values")
        
        return df
    
    def create_lagged_features(self,
                               df: pd.DataFrame,
                               target_col: str = 'net_load',
                               short_lags: list = None,
                               medium_lags: list = None) -> pd.DataFrame:
        """
        Create lagged features for target variable.
        
        From proposal 7.1:
        - Short lags: 1-48 hours (captures recent trends)
        - Medium lags: 7 days = 168 hours (captures weekly patterns)
        
        Args:
            df: DataFrame with target column
            target_col: Name of column to lag
            short_lags: List of lag values for short horizon (default: [1,2,3,6,12,24,48])
            medium_lags: List of lag values for medium horizon (default: [168])
            
        Returns:
            DataFrame with lagged features added
        """
        if short_lags is None:
            short_lags = [1, 2, 3, 6, 12, 24, 48]
        if medium_lags is None:
            medium_lags = [168]  # 7 days
        
        df = df.copy()
        
        for lag in short_lags:
            df[f'{target_col}_lag{lag}h'] = df[target_col].shift(lag)
        
        for lag in medium_lags:
            df[f'{target_col}_lag{lag}h'] = df[target_col].shift(lag)
        
        logger.info(f"Created {len(short_lags) + len(medium_lags)} lagged features for {target_col}")
        
        return df
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar and time-of-use features.
        
        From proposal 7.1:
        - Hour of day (0-23, captures diurnal cycle)
        - Day of week (0-6, captures weekly patterns)
        - Month (0-11, captures seasonal patterns)
        - Holiday indicator (if available)
        
        Args:
            df: DataFrame with datetime column
            
        Returns:
            DataFrame with calendar features added
        """
        df = df.copy()
        
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_month'] = df['datetime'].dt.day
        
        # Create interaction: is_peak_hour (18:00 - 22:00, critical for this study)
        df['is_peak_hour'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
        
        logger.info("Created calendar features: hour, day_of_week, month, day_of_month, is_peak_hour")
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-derived features.
        
        From proposal 7.1:
        - Lagged weather (to capture delayed demand response)
        - Interaction features: heat stress (Temperature × Humidity)
        
        Args:
            df: DataFrame with weather columns
            
        Returns:
            DataFrame with weather features added
        """
        df = df.copy()
        
        # Lagged weather (12, 24 hours to capture delayed response)
        for lag in [12, 24]:
            df[f'temperature_lag{lag}h'] = df['Temperature'].shift(lag)
            df[f'humidity_lag{lag}h'] = df['Humidity'].shift(lag)
        
        # Heat stress interaction (if temperature and humidity both available)
        if 'Temperature' in df.columns and 'Humidity' in df.columns:
            df['heat_stress'] = (df['Temperature'] * df['Humidity']) / 100.0
            df['heat_stress_lag24h'] = df['heat_stress'].shift(24)
        
        logger.info("Created weather features: lagged temperature, lagged humidity, heat stress")
        
        return df
    
    def filter_physical_plausibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with physically impossible electricity values and data errors.
        
        Physical plausibility filtering (NOT statistical outlier removal):
        - Demand must be within Bangladesh system scale (500-15,000 MW)
        - Generation must be within system capacity (0-20,000 MW)
        - Load shedding must be non-negative and ≤ demand
        - Served load must be non-negative and ≤ demand
        - Net load must be within reasonable operational range (-1,000 to 12,000 MW)
        - Timestamps must be hourly resolution (no duplicates within same hour)
        
        These thresholds reflect the physical limits of Bangladesh's power system,
        not statistical measures. Rows outside these bounds indicate data errors or
        impossible states and are removed before model training.
        
        Args:
            df: DataFrame with electricity and timing columns
            
        Returns:
            Filtered DataFrame with invalid rows removed
        """
        df = df.copy()
        initial_count = len(df)
        
        # Define physical bounds for Bangladesh power system
        DEMAND_MIN_MW = 500           # Minimum realistic demand
        DEMAND_MAX_MW = 15000         # Maximum system capacity with safety margin
        GENERATION_MIN_MW = 0         # Generation cannot be negative
        GENERATION_MAX_MW = 20000     # Upper bound with margin
        LOAD_SHEDDING_MIN_MW = 0      # Load shedding cannot be negative
        LOAD_SHEDDING_MAX_MW = 10000  # Maximum realistic load shedding
        SERVED_LOAD_MIN_MW = 0        # Served load cannot be negative
        SERVED_LOAD_MAX_MW = 15000    # Upper bound same as demand
        NET_LOAD_MIN_MW = -1000       # Can be negative if demand < baseload + renewables
        NET_LOAD_MAX_MW = 12000       # Maximum flexible load needed
        
        # Track removal reasons
        removal_reasons = {
            'demand_out_of_bounds': 0,
            'generation_out_of_bounds': 0,
            'load_shedding_out_of_bounds': 0,
            'served_load_invalid': 0,
            'net_load_out_of_bounds': 0,
            'non_hourly_timestamp': 0,
            'other': 0
        }
        
        # Create detailed mask for each validation rule
        demand_valid = (df['demand_mw'] >= DEMAND_MIN_MW) & (df['demand_mw'] <= DEMAND_MAX_MW)
        
        # Generation check - only if column exists
        if 'generation_mw' in df.columns:
            generation_valid = (df['generation_mw'] >= GENERATION_MIN_MW) & (df['generation_mw'] <= GENERATION_MAX_MW)
        else:
            generation_valid = pd.Series([True] * len(df))
        
        # Load shedding check - only if column exists
        if 'load_shedding' in df.columns:
            load_shedding_valid = (df['load_shedding'] >= LOAD_SHEDDING_MIN_MW) & (df['load_shedding'] <= LOAD_SHEDDING_MAX_MW)
        else:
            load_shedding_valid = pd.Series([True] * len(df))
        
        # Served load checks
        served_load_valid = (
            (df['served_load'] >= SERVED_LOAD_MIN_MW) & 
            (df['served_load'] <= SERVED_LOAD_MAX_MW) &
            (df['served_load'] <= df['demand_mw'])  # Served load ≤ demand
        )
        
        # Net load checks
        net_load_valid = (df['net_load'] >= NET_LOAD_MIN_MW) & (df['net_load'] <= NET_LOAD_MAX_MW)
        
        # Check for hourly timestamp resolution (no duplicates in same hour)
        if 'datetime' in df.columns:
            df['date_hour'] = df['datetime'].dt.floor('H')
            hourly_valid = ~df.duplicated(subset=['date_hour'], keep=False)
        else:
            hourly_valid = pd.Series([True] * len(df))
        
        # Combine all masks
        plausible_mask = demand_valid & generation_valid & load_shedding_valid & served_load_valid & net_load_valid & hourly_valid
        
        # Count reasons for removal
        if not demand_valid.all():
            removal_reasons['demand_out_of_bounds'] = (~demand_valid).sum()
        if not generation_valid.all():
            removal_reasons['generation_out_of_bounds'] = (~generation_valid).sum()
        if not load_shedding_valid.all():
            removal_reasons['load_shedding_out_of_bounds'] = (~load_shedding_valid).sum()
        if not served_load_valid.all():
            removal_reasons['served_load_invalid'] = (~served_load_valid).sum()
        if not net_load_valid.all():
            removal_reasons['net_load_out_of_bounds'] = (~net_load_valid).sum()
        if not hourly_valid.all():
            removal_reasons['non_hourly_timestamp'] = (~hourly_valid).sum()
        
        # Filter to plausible rows
        df = df[plausible_mask].reset_index(drop=True)
        
        # Drop temporary columns
        if 'date_hour' in df.columns:
            df = df.drop(columns=['date_hour'])
        
        removed_count = initial_count - len(df)
        removed_pct = 100.0 * removed_count / initial_count if initial_count > 0 else 0
        
        logger.info(f"Physical plausibility filtering:")
        logger.info(f"  Demand bounds: [{DEMAND_MIN_MW}, {DEMAND_MAX_MW}] MW")
        logger.info(f"  Generation bounds: [{GENERATION_MIN_MW}, {GENERATION_MAX_MW}] MW")
        logger.info(f"  Load shedding bounds: [{LOAD_SHEDDING_MIN_MW}, {LOAD_SHEDDING_MAX_MW}] MW")
        logger.info(f"  Served load bounds: [{SERVED_LOAD_MIN_MW}, {SERVED_LOAD_MAX_MW}] MW")
        logger.info(f"  Net load bounds: [{NET_LOAD_MIN_MW}, {NET_LOAD_MAX_MW}] MW")
        logger.info(f"  Timestamp resolution: Hourly (no duplicates within same hour)")
        logger.info(f"  Rows removed by reason:")
        for reason, count in removal_reasons.items():
            if count > 0:
                logger.info(f"    - {reason}: {count} rows")
        logger.info(f"  Total rows removed: {removed_count} ({removed_pct:.2f}%)")
        logger.info(f"  Rows retained: {len(df)} ({100.0 - removed_pct:.2f}%)")
        
        return df
    
    def prepare_features(self,
                        electricity_df: pd.DataFrame,
                        weather_df: pd.DataFrame,
                        split_date: pd.Timestamp = None) -> tuple:
        """
        Full feature engineering pipeline.
        
        Process:
        1. Construct net load target
        2. Align weather to hourly
        3. Create lagged features
        4. Create calendar features
        5. Create weather features
        6. Handle NaN from lags
        7. Return train/test split
        
        Args:
            electricity_df: Raw electricity data
            weather_df: Raw weather data
            split_date: Date to split train/test (chronological). 
                       If None, uses 80/20 split.
        
        Returns:
            Tuple of (df_full, train_df, test_df, feature_cols, target_col)
        """
        # Construct net load
        df = self.construct_net_load(electricity_df)
        
        # Align weather
        df = self.align_weather_to_hourly(df, weather_df)
        
        # Create features
        df = self.create_lagged_features(df, target_col='net_load')
        df = self.create_calendar_features(df)
        df = self.create_weather_features(df)
        
        # Remove rows with NaN from lagging (max lag is 168 hours = 7 days)
        max_lag = 168
        df = df.iloc[max_lag:].reset_index(drop=True)
        
        logger.info(f"After feature engineering: {len(df)} rows, {df.shape[1]} columns")
        
        # Apply physical plausibility filtering (removes impossible values BEFORE training)
        logger.info("Applying physical plausibility checks...")
        df = self.filter_physical_plausibility(df)
        
        # Define feature columns (exclude target and meta columns)
        target_col = 'net_load'
        meta_cols = ['datetime', 'date', 'remarks', 'demand_mw', 'load_shedding', 
                     'served_load', 'renewable_output', 'generation_mw', 'gas', 'liquid_fuel',
                     'coal', 'hydro', 'solar', 'wind', 'india_bheramara_hvdc', 'india_tripura',
                     'india_adani', 'nepal', 'Station']
        
        feature_cols = [col for col in df.columns if col not in meta_cols and col != target_col]
        
        logger.info(f"Selected {len(feature_cols)} features for modeling")
        
        # Chronological train/test split
        if split_date is None:
            # 80/20 split
            split_idx = int(0.8 * len(df))
            split_date = df.iloc[split_idx]['datetime']
        
        train_mask = df['datetime'] < split_date
        test_mask = df['datetime'] >= split_date
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        logger.info(f"Train/test split (chronological):")
        logger.info(f"  Train: {len(df_train)} samples ({df_train['datetime'].min()} to {df_train['datetime'].max()})")
        logger.info(f"  Test: {len(df_test)} samples ({df_test['datetime'].min()} to {df_test['datetime'].max()})")
        
        return df, df_train, df_test, feature_cols, target_col


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader
    
    loader = DataLoader(Path("constraint_aware_net_load/data"))
    elec_df = loader.load_electricity_data()
    weather_df = loader.load_weather_data()
    
    engineer = FeatureEngineer(rigid_baseload_mw=2200.0)
    df, df_train, df_test, feature_cols, target_col = engineer.prepare_features(elec_df, weather_df)
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nFeatures created: {feature_cols}")
    print(f"\nTrain data sample:\n{df_train[['datetime', target_col] + feature_cols[:5]].head()}")
    print(f"\nTest data sample:\n{df_test[['datetime', target_col] + feature_cols[:5]].head()}")
