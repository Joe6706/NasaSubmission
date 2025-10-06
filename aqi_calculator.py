"""
AQI calculation module using official EPA formulas and breakpoints.
Computes Air Quality Index and identifies dominant pollutants.
"""

from typing import Dict, Tuple, Union

# Conditional imports with fallbacks
try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas/numpy not available. Using simplified implementations.")
    PANDAS_AVAILABLE = False
    # Fallback classes
    class np:
        @staticmethod
        def nan(): return None
        nan = None
    class pd:
        @staticmethod
        def isna(x): return x is None
        class DataFrame:
            def __init__(self, data=None): pass
            def copy(self): return self
            def iterrows(self): return []
            @property
            def columns(self): return []
            def loc(self, *args): pass
from typing import Dict, Tuple, Union


class EPAAQICalculator:
    """
    EPA Air Quality Index calculator with official breakpoints and formulas.
    
    Uses the official EPA AQI calculation method with standard breakpoints
    for each pollutant category.
    """
    
    def __init__(self):
        # EPA AQI Breakpoints (as of 2023)
        # Format: pollutant -> [(C_low, C_high, I_low, I_high), ...]
        # Where C = concentration, I = AQI value
        
        self.breakpoints = {
            'PM2.5': [  # μg/m³, 24-hour average
                (0.0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500)
            ],
            'PM10': [  # μg/m³, 24-hour average
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 504, 301, 400),
                (505, 604, 401, 500)
            ],
            'O3': [  # ppm, 8-hour maximum
                (0.000, 0.054, 0, 50),
                (0.055, 0.070, 51, 100),
                (0.071, 0.085, 101, 150),
                (0.086, 0.105, 151, 200),
                (0.106, 0.200, 201, 300),
                (0.201, 0.300, 301, 400),
                (0.301, 0.500, 401, 500)
            ],
            'CO': [  # ppm, 8-hour average
                (0.0, 4.4, 0, 50),
                (4.5, 9.4, 51, 100),
                (9.5, 12.4, 101, 150),
                (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300),
                (30.5, 40.4, 301, 400),
                (40.5, 50.4, 401, 500)
            ],
            'NO2': [  # ppm, 1-hour average
                (0.000, 0.053, 0, 50),
                (0.054, 0.100, 51, 100),
                (0.101, 0.360, 101, 150),
                (0.361, 0.649, 151, 200),
                (0.650, 1.249, 201, 300),
                (1.250, 1.649, 301, 400),
                (1.650, 2.049, 401, 500)
            ],
            'SO2': [  # ppm, 1-hour average
                (0.000, 0.035, 0, 50),
                (0.036, 0.075, 51, 100),
                (0.076, 0.185, 101, 150),
                (0.186, 0.304, 151, 200),
                (0.305, 0.604, 201, 300),
                (0.605, 0.804, 301, 400),
                (0.805, 1.004, 401, 500)
            ]
        }
        
        # AQI category labels
        self.aqi_categories = {
            (0, 50): "Good",
            (51, 100): "Moderate", 
            (101, 150): "Unhealthy for Sensitive Groups",
            (151, 200): "Unhealthy",
            (201, 300): "Very Unhealthy",
            (301, 500): "Hazardous"
        }
    
    def calculate_pollutant_aqi(self, concentration: float, pollutant: str) -> int:
        """
        Calculate AQI for a single pollutant using EPA formula.
        
        Formula: I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
        
        Args:
            concentration: Pollutant concentration in EPA standard units
            pollutant: Pollutant name (PM2.5, PM10, O3, CO, NO2, SO2)
            
        Returns:
            AQI value for the pollutant
        """
        if pollutant not in self.breakpoints:
            raise ValueError(f"Unknown pollutant: {pollutant}")
        
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        
        breakpoints = self.breakpoints[pollutant]
        
        # Find the appropriate breakpoint
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= concentration <= c_high:
                # EPA AQI formula
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                return int(round(aqi))
        
        # If concentration exceeds highest breakpoint, return maximum AQI
        return 500
    
    def calculate_aqi_from_concentrations(self, concentrations: Dict[str, float]) -> Tuple[int, str]:
        """
        Calculate overall AQI and dominant pollutant from multiple concentrations.
        
        Args:
            concentrations: Dictionary of pollutant concentrations
                          Keys should match AQI standard averaging periods:
                          - PM2.5_24hr_avg, PM10_24hr_avg (24-hour averages)
                          - O3_8hr_max (8-hour maximum)
                          - CO_8hr_avg (8-hour average)
                          - NO2_1hr_avg, SO2_1hr_avg (1-hour averages)
        
        Returns:
            tuple: (overall_AQI, dominant_pollutant)
        """
        aqi_values = {}
        
        # Map concentration keys to pollutant names
        concentration_mapping = {
            'PM2.5_24hr_avg': 'PM2.5',
            'PM10_24hr_avg': 'PM10', 
            'O3_8hr_max': 'O3',
            'CO_8hr_avg': 'CO',
            'NO2_1hr_avg': 'NO2',
            'SO2_1hr_avg': 'SO2',
            # Also accept simple pollutant names
            'PM2.5': 'PM2.5',
            'PM10': 'PM10',
            'O3': 'O3',
            'CO': 'CO',
            'NO2': 'NO2',
            'SO2': 'SO2'
        }
        
        # Calculate AQI for each available pollutant
        for conc_key, concentration in concentrations.items():
            if conc_key in concentration_mapping and not pd.isna(concentration):
                pollutant = concentration_mapping[conc_key]
                aqi = self.calculate_pollutant_aqi(concentration, pollutant)
                if not pd.isna(aqi):
                    aqi_values[pollutant] = aqi
        
        if not aqi_values:
            return np.nan, "No valid data"
        
        # Overall AQI is the maximum of all pollutant AQIs
        overall_aqi = max(aqi_values.values())
        dominant_pollutant = max(aqi_values, key=aqi_values.get)
        
        return overall_aqi, dominant_pollutant
    
    def get_aqi_category(self, aqi: int) -> str:
        """
        Get AQI category label for a given AQI value.
        
        Args:
            aqi: AQI value
            
        Returns:
            AQI category string
        """
        if pd.isna(aqi):
            return "No Data"
        
        for (low, high), category in self.aqi_categories.items():
            if low <= aqi <= high:
                return category
        
        return "Hazardous"  # For AQI > 500
    
    def calculate_dataframe_aqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate AQI for a DataFrame with multiple time periods.
        
        Args:
            df: DataFrame with pollutant concentration columns
            
        Returns:
            DataFrame with added AQI columns
        """
        df = df.copy()
        
        # Initialize result columns
        df['AQI'] = np.nan
        df['dominant_pollutant'] = "No Data"
        df['AQI_category'] = "No Data"
        
        # Calculate AQI for each row
        for idx, row in df.iterrows():
            # Prepare concentrations dictionary
            concentrations = {}
            
            # Try to find appropriate averaged concentrations
            for col in df.columns:
                if any(avg_col in col for avg_col in ['_24hr_avg', '_8hr_max', '_8hr_avg', '_1hr_avg']):
                    concentrations[col] = row[col]
                elif col in ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']:
                    concentrations[col] = row[col]
            
            # Calculate AQI and dominant pollutant
            aqi, dominant = self.calculate_aqi_from_concentrations(concentrations)
            
            df.loc[idx, 'AQI'] = aqi
            df.loc[idx, 'dominant_pollutant'] = dominant
            df.loc[idx, 'AQI_category'] = self.get_aqi_category(aqi)
        
        return df
    
    def calculate_aqi_from_predictions(self, predictions: Dict[str, float]) -> Dict:
        """
        Calculate AQI from predicted pollutant concentrations.
        
        Args:
            predictions: Dictionary with predicted concentrations
            
        Returns:
            Dictionary with AQI results
        """
        aqi, dominant = self.calculate_aqi_from_concentrations(predictions)
        category = self.get_aqi_category(aqi)
        
        return {
            'AQI': int(aqi) if not pd.isna(aqi) else None,
            'dominant_pollutant': dominant,
            'AQI_category': category,
            'pollutant_AQIs': {
                pollutant: self.calculate_pollutant_aqi(
                    predictions.get(pollutant, np.nan), pollutant
                )
                for pollutant in self.breakpoints.keys()
                if pollutant in predictions and not pd.isna(predictions.get(pollutant))
            }
        }


def calculate_aqi_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to calculate AQI for a DataFrame.
    
    Args:
        df: DataFrame with pollutant concentrations
        
    Returns:
        DataFrame with AQI calculations
    """
    calculator = EPAAQICalculator()
    return calculator.calculate_dataframe_aqi(df)


if __name__ == "__main__":
    # Example usage
    calculator = EPAAQICalculator()
    
    # Example concentrations
    example_concentrations = {
        'PM2.5': 35.0,  # μg/m³
        'O3': 0.070,    # ppm
        'NO2': 0.050    # ppm
    }
    
    aqi, dominant = calculator.calculate_aqi_from_concentrations(example_concentrations)
    print(f"Example AQI: {aqi}, Dominant pollutant: {dominant}")
    print(f"AQI Category: {calculator.get_aqi_category(aqi)}")