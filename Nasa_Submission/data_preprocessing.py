"""
Data preprocessing module for AQI forecasting.
Handles rolling averages, unit conversions, and data preparation for ML models.
"""

from typing import Dict, Optional

# Conditional imports with fallbacks
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas/numpy not available. Using simplified implementations.")
    PANDAS_AVAILABLE = False
    # Fallback classes
    class pd:
        class DataFrame:
            def __init__(self, data=None): 
                self.data = data or {}
            def copy(self): return self
            def set_index(self, col): return self
            def sort_index(self): return self
            @property
            def columns(self): return list(self.data.keys())
            def rolling(self, window, min_periods=None): return self
            def mean(self): return self
            def max(self): return self
            def fillna(self, method=None): return self
            def ffill(self): return self
            def bfill(self): return self
            def isnull(self): return self
            def sum(self): return 0
            def __len__(self): return len(list(self.data.values())[0]) if self.data else 0
            def __getitem__(self, key): return self.data.get(key, [])
        class DatetimeIndex: pass
        @staticmethod
        def to_datetime(x): return x
        @staticmethod
        def read_csv(path): return pd.DataFrame()
        @staticmethod
        def cut(x, bins, labels): return x
    class np:
        @staticmethod
        def array(x): return x
        nan = None
from typing import Dict, Optional


class AQIDataPreprocessor:
    """
    Preprocessor for air quality and weather data for AQI forecasting.
    
    Handles:
    - Rolling averages for different pollutants according to EPA standards
    - Unit conversions for AQI calculations
    - Data cleaning and feature engineering
    """
    
    def __init__(self):
        # EPA AQI calculation requires specific averaging periods
        self.averaging_periods = {
            'PM2.5': 24,  # 24-hour average
            'PM10': 24,   # 24-hour average
            'O3': 8,      # 8-hour maximum
            'CO': 8,      # 8-hour average
            'NO2': 1,     # 1-hour average
            'SO2': 1      # 1-hour average
        }
        
        # Unit conversion factors (if needed)
        # EPA AQI uses specific units for each pollutant
        self.unit_conversions = {
            'PM2.5': 1.0,  # μg/m³ (already correct)
            'PM10': 1.0,   # μg/m³ (already correct)
            'O3': 1.0,     # ppm (already correct)
            'CO': 1.0,     # ppm (already correct)
            'NO2': 1.0,    # ppm (already correct)
            'SO2': 1.0     # ppm (already correct)
        }
    
    def compute_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling averages for pollutants according to EPA standards.
        
        Args:
            df: DataFrame with hourly pollutant data
            
        Returns:
            DataFrame with additional rolling average columns
        """
        df = df.copy()
        
        # Ensure datetime index
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time').sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index or 'time' column")
        else:
            # Sort by time to ensure proper rolling calculations
            df = df.sort_index()
        
        # Store city column separately if it exists
        city_column = None
        if 'city' in df.columns:
            city_column = df['city'].copy()
            # Remove city from rolling calculations but keep it for later
            df_for_rolling = df.drop('city', axis=1)
        else:
            df_for_rolling = df
        
        # Compute rolling averages for each pollutant according to EPA guidelines
        for pollutant, window in self.averaging_periods.items():
            if pollutant in df_for_rolling.columns:
                if pollutant == 'O3':
                    # O3: 8-hour rolling maximum
                    df_for_rolling[f'{pollutant}_8hr_max'] = df_for_rolling[pollutant].rolling(
                        window=window, min_periods=6  # At least 75% of data
                    ).max()
                elif pollutant in ['NO2', 'SO2']:
                    # NO2 and SO2: Use hourly values directly (no averaging needed)
                    df_for_rolling[f'{pollutant}_1hr'] = df_for_rolling[pollutant]
                else:
                    # PM2.5, PM10, CO: Rolling average
                    avg_col = f'{pollutant}_{window}hr_avg'
                    df_for_rolling[avg_col] = df_for_rolling[pollutant].rolling(
                        window=window, min_periods=int(window * 0.75)  # At least 75% of data
                    ).mean()
        
        # Add city column back if it existed
        if city_column is not None:
            df_for_rolling['city'] = city_column
        
        return df_for_rolling
    
    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert pollutant concentrations to EPA AQI standard units if needed.
        
        Args:
            df: DataFrame with pollutant data
            
        Returns:
            DataFrame with converted units
        """
        df = df.copy()
        
        for pollutant, factor in self.unit_conversions.items():
            if pollutant in df.columns and factor != 1.0:
                df[pollutant] = df[pollutant] * factor
                
            # Also convert rolling averages
            for col in df.columns:
                if col.startswith(pollutant) and ('_avg' in col or '_max' in col):
                    if factor != 1.0:
                        df[col] = df[col] * factor
        
        return df
    
    def create_features(self, df: pd.DataFrame, lookback_hours: int = 24) -> pd.DataFrame:
        """
        Create additional features for ML models.
        
        Args:
            df: DataFrame with processed pollutant and weather data
            lookback_hours: Number of hours to look back for lag features
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['season'] = (df.index.month % 12 + 3) // 3  # 1=Spring, 2=Summer, 3=Fall, 4=Winter
        
        # City encoding for North American cities
        if 'city' in df.columns:
            # One-hot encode city names (for North American cities)
            city_dummies = pd.get_dummies(df['city'], prefix='city')
            df = pd.concat([df, city_dummies], axis=1)
            # Keep original city column for reference
            df['city_encoded'] = df['city'].astype('category').cat.codes
        
        # Lag features for pollutants (previous hours)
        pollutant_cols = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
        for pollutant in pollutant_cols:
            if pollutant in df.columns:
                for lag in [1, 3, 6, 12, 24]:
                    if lag <= lookback_hours:
                        df[f'{pollutant}_lag_{lag}h'] = df[pollutant].shift(lag)
        
        # Weather interaction features
        if all(col in df.columns for col in ['temperature', 'humidity', 'wind_speed']):
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['wind_temp_interaction'] = df['wind_speed'] * df['temperature']
        
        # Meteorological stability indicators
        if 'wind_speed' in df.columns:
            df['wind_stability'] = pd.cut(df['wind_speed'], 
                                        bins=[0, 2, 5, 10, float('inf')],
                                        labels=['calm', 'light', 'moderate', 'strong'])
            df['wind_stability'] = df['wind_stability'].cat.codes
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame, target_cols: list) -> tuple:
        """
        Prepare data for machine learning models.
        
        Args:
            df: DataFrame with features
            target_cols: List of target columns (pollutants to predict)
            
        Returns:
            tuple: (features_df, targets_df, feature_names)
        """
        df = df.copy()
        
        # Define feature columns (exclude targets, non-predictive, and categorical columns)
        exclude_cols = set(target_cols + ['time', 'city'])  # Exclude original city column
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any columns with too many missing values
        feature_cols = [col for col in feature_cols 
                       if df[col].isnull().sum() / len(df) < 0.5]
        
        # Get features and targets
        features = df[feature_cols].copy()
        targets = df[target_cols].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill') if hasattr(features, 'fillna') else features.ffill().bfill()
        targets = targets.fillna(method='ffill').fillna(method='bfill') if hasattr(targets, 'fillna') else targets.ffill().bfill()
        
        return features, targets, feature_cols
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_cols: Optional[list] = None) -> tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame with hourly data
            target_cols: Target columns for prediction (default: all pollutants)
            
        Returns:
            tuple: (processed_features, targets, feature_names) if target_cols provided,
                   else processed_df
        """
        # Store the original target_cols to check if modeling is requested
        modeling_requested = target_cols is not None
        
        if target_cols is None:
            target_cols = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
        
        # Step 1: Compute rolling averages
        df_processed = self.compute_rolling_averages(df)
        
        # Step 2: Convert units
        df_processed = self.convert_units(df_processed)
        
        # Step 3: Create additional features
        df_processed = self.create_features(df_processed)
        
        # Step 4: Prepare for modeling if targets specified
        if modeling_requested:
            # Only include target columns that exist in the data
            available_targets = [col for col in target_cols if col in df_processed.columns]
            if available_targets:
                features, targets, feature_names = self.prepare_for_modeling(
                    df_processed, available_targets
                )
                return features, targets, feature_names
        
        return df_processed


def load_and_preprocess_data(csv_path: str, target_cols: Optional[list] = None) -> tuple:
    """
    Convenience function to load and preprocess data from CSV.
    
    Args:
        csv_path: Path to CSV file with hourly data
        target_cols: Target columns for prediction
        
    Returns:
        tuple: (features, targets, feature_names) or processed_df
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Initialize preprocessor
    preprocessor = AQIDataPreprocessor()
    
    # Run preprocessing pipeline
    result = preprocessor.preprocess_pipeline(df, target_cols)
    
    return result


if __name__ == "__main__":
    # Example usage
    print("AQI Data Preprocessing Module")
    print("Use load_and_preprocess_data() to process your CSV file")
    
    # Example of expected CSV columns:
    expected_columns = [
        'time',  # datetime column
        'city',  # city/location name
        'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO',  # pollutants
        'temperature', 'humidity', 'wind_speed'  # weather features
    ]
    print(f"Expected CSV columns: {expected_columns}")