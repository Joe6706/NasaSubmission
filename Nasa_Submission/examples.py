"""
Example usage and testing script for the AQI Forecasting System.
Demonstrates how to use each component of the system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import our modules
try:
    from data_preprocessing import AQIDataPreprocessor, load_and_preprocess_data
    from aqi_calculator import EPAAQICalculator
    from ml_models import PollutantPredictor, train_multiple_models
    from forecasting import AQIForecaster, create_forecast_from_csv
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    exit(1)


def create_sample_data(hours=168, save_file='sample_data.csv'):
    """
    Create sample TEMPO satellite and weather data for testing.
    
    Args:
        hours: Number of hours of data to generate
        save_file: Path to save the sample CSV
    """
    print(f"Creating {hours} hours of sample data...")
    
    # Generate time series
    start_date = datetime(2025, 10, 1)
    dates = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Set random seed for reproducible data
    np.random.seed(42)
    
    # Generate realistic pollutant concentrations
    # PM2.5: typically 10-50 μg/m³ in urban areas
    pm25_base = 25
    pm25_trend = np.sin(np.arange(hours) * 2 * np.pi / 24) * 10  # Daily cycle
    pm25_noise = np.random.normal(0, 5, hours)
    pm25 = np.maximum(0, pm25_base + pm25_trend + pm25_noise)
    
    # PM10: typically 1.2-2x PM2.5
    pm10 = pm25 * np.random.uniform(1.2, 2.0, hours)
    
    # O3: typical range 0.02-0.12 ppm with strong daily cycle  
    o3_base = 0.05
    o3_trend = np.sin((np.arange(hours) - 6) * 2 * np.pi / 24) * 0.03  # Peak in afternoon
    o3_trend = np.maximum(o3_trend, -0.04)  # No negative values
    o3_noise = np.random.normal(0, 0.01, hours)
    o3 = np.maximum(0, o3_base + o3_trend + o3_noise)
    
    # NO2: typically 0.01-0.05 ppm, higher during rush hours
    no2_base = 0.025
    no2_rush = np.where((np.arange(hours) % 24 >= 7) & (np.arange(hours) % 24 <= 9) |
                       (np.arange(hours) % 24 >= 17) & (np.arange(hours) % 24 <= 19), 0.01, 0)
    no2_noise = np.random.normal(0, 0.005, hours)
    no2 = np.maximum(0, no2_base + no2_rush + no2_noise)
    
    # SO2: typically low in urban areas, 0.001-0.02 ppm
    so2_base = 0.005
    so2_noise = np.random.normal(0, 0.002, hours)
    so2 = np.maximum(0, so2_base + so2_noise)
    
    # CO: typically 0.5-3 ppm in urban areas
    co_base = 1.0
    co_traffic = np.where((np.arange(hours) % 24 >= 6) & (np.arange(hours) % 24 <= 22), 0.5, 0)
    co_noise = np.random.normal(0, 0.3, hours)
    co = np.maximum(0, co_base + co_traffic + co_noise)
    
    # Weather data
    # Temperature: seasonal and daily variation
    temp_seasonal = 20 + 10 * np.sin(np.arange(hours) * 2 * np.pi / (24 * 365))
    temp_daily = 5 * np.sin((np.arange(hours) - 6) * 2 * np.pi / 24)
    temp_noise = np.random.normal(0, 2, hours)
    temperature = temp_seasonal + temp_daily + temp_noise
    
    # Humidity: anti-correlated with temperature
    humidity_base = 60
    humidity_temp_effect = -0.5 * (temperature - 20)  # Higher temp = lower humidity
    humidity_noise = np.random.normal(0, 10, hours)
    humidity = np.clip(humidity_base + humidity_temp_effect + humidity_noise, 0, 100)
    
    # Wind speed: random but realistic
    wind_speed = np.random.gamma(2, 2, hours)  # Gamma distribution for wind
    wind_speed = np.clip(wind_speed, 0, 15)  # Cap at 15 m/s
    
    # Create DataFrame
    data = {
        'time': [d.strftime('%Y-%m-%dT%H:%M:%SZ') for d in dates],
        'PM2.5': pm25.round(1),
        'PM10': pm10.round(1), 
        'O3': o3.round(4),
        'NO2': no2.round(4),
        'SO2': so2.round(4),
        'CO': co.round(2),
        'temperature': temperature.round(1),
        'humidity': humidity.round(1),
        'wind_speed': wind_speed.round(1)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(save_file, index=False)
    print(f"Sample data saved to {save_file}")
    
    # Print summary
    print(f"\nData Summary:")
    print(f"Time range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print(f"PM2.5 range: {df['PM2.5'].min():.1f} - {df['PM2.5'].max():.1f} μg/m³")
    print(f"O3 range: {df['O3'].min():.3f} - {df['O3'].max():.3f} ppm")
    print(f"Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")
    
    return df


def example_preprocessing():
    """Demonstrate data preprocessing functionality."""
    print("\n" + "="*50)
    print("EXAMPLE 1: Data Preprocessing")
    print("="*50)
    
    # Create sample data
    df = create_sample_data(72, 'example_data.csv')  # 3 days
    
    # Initialize preprocessor
    preprocessor = AQIDataPreprocessor()
    
    # Compute rolling averages
    print("\nComputing rolling averages...")
    df_processed = preprocessor.compute_rolling_averages(df)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"After processing: {list(df_processed.columns)}")
    
    # Show some rolling averages
    if 'PM2.5_24hr_avg' in df_processed.columns:
        print(f"\nPM2.5 24-hour averages (last 5 values):")
        print(df_processed[['time', 'PM2.5', 'PM2.5_24hr_avg']].tail())
    
    return df_processed


def example_aqi_calculation():
    """Demonstrate AQI calculation functionality."""
    print("\n" + "="*50)
    print("EXAMPLE 2: AQI Calculation")
    print("="*50)
    
    # Initialize AQI calculator
    calculator = EPAAQICalculator()
    
    # Example 1: Single pollutant AQI
    print("Single pollutant AQI calculations:")
    examples = [
        ('PM2.5', 35.0, 'μg/m³'),
        ('O3', 0.070, 'ppm'),
        ('NO2', 0.053, 'ppm'),
        ('CO', 4.4, 'ppm')
    ]
    
    for pollutant, concentration, unit in examples:
        aqi = calculator.calculate_pollutant_aqi(concentration, pollutant)
        category = calculator.get_aqi_category(aqi)
        print(f"{pollutant}: {concentration} {unit} → AQI {aqi} ({category})")
    
    # Example 2: Combined AQI
    print("\nCombined AQI calculation:")
    concentrations = {
        'PM2.5': 35.0,
        'O3': 0.070,
        'NO2': 0.025,
        'CO': 1.5
    }
    
    overall_aqi, dominant = calculator.calculate_aqi_from_concentrations(concentrations)
    category = calculator.get_aqi_category(overall_aqi)
    
    print(f"Concentrations: {concentrations}")
    print(f"Overall AQI: {overall_aqi} ({category})")
    print(f"Dominant pollutant: {dominant}")


def example_model_training():
    """Demonstrate model training functionality."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Model Training")
    print("="*50)
    
    # Create training data
    df = create_sample_data(336, 'training_data.csv')  # 2 weeks
    
    # Preprocess data
    features, targets, feature_names = load_and_preprocess_data(
        'training_data.csv',
        target_cols=['PM2.5', 'PM10', 'O3', 'NO2']
    )
    
    print(f"Training data shape: Features {features.shape}, Targets {targets.shape}")
    print(f"Feature columns: {len(feature_names)}")
    print(f"Target pollutants: {list(targets.columns)}")
    
    # Train Random Forest model (faster for demo)
    print("\nTraining Random Forest model...")
    model = PollutantPredictor('random_forest', n_estimators=50)
    model.fit(features, targets)
    
    print("Training completed!")
    print(f"Validation metrics:")
    for pollutant, metrics in model.validation_metrics.items():
        print(f"  {pollutant}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
    
    # Save model
    model.save_model('example_model')
    print("Model saved as 'example_model'")
    
    return model


def example_forecasting():
    """Demonstrate forecasting functionality."""
    print("\n" + "="*50)
    print("EXAMPLE 4: AQI Forecasting")
    print("="*50)
    
    # Use the create_forecast_from_csv convenience function
    print("Generating forecast from CSV data...")
    
    # This will use the training_data.csv from previous example
    forecasts = create_forecast_from_csv(
        'training_data.csv',
        model_path='example_model',
        forecast_hours=12,
        output_path='example_forecast.json'
    )
    
    print(f"Generated {len(forecasts)} hourly forecasts")
    
    # Display first few forecasts
    print("\nFirst 3 forecast hours:")
    for i, forecast in enumerate(forecasts[:3]):
        print(f"Hour {i+1}:")
        print(f"  Time: {forecast['time']}")
        print(f"  AQI: {forecast['predicted_AQI']} ({forecast['AQI_category']})")
        print(f"  Dominant: {forecast['dominant_pollutant']}")
        print(f"  PM2.5: {forecast['predicted_pollutants']['PM2.5']:.1f} μg/m³")
    
    # Load and display summary
    with open('example_forecast.json', 'r') as f:
        full_output = json.load(f)
    
    summary = full_output['summary']
    print(f"\nForecast Summary:")
    print(f"  Max AQI: {summary['aqi_summary']['max_aqi']}")
    print(f"  Average AQI: {summary['aqi_summary']['avg_aqi']}")
    print(f"  Worst hour: {summary['worst_hour']['time']}")
    print(f"  Most frequent pollutant: {summary['most_frequent_pollutant']}")


def example_realtime_aqi():
    """Demonstrate real-time AQI calculation for current conditions."""
    print("\n" + "="*50)
    print("EXAMPLE 5: Real-time AQI")
    print("="*50)
    
    calculator = EPAAQICalculator()
    
    # Simulate real-time measurements
    current_measurements = [
        {"time": "2025-10-05T08:00:00Z", "PM2.5": 45, "O3": 0.065, "NO2": 0.035},
        {"time": "2025-10-05T12:00:00Z", "PM2.5": 38, "O3": 0.085, "NO2": 0.028},
        {"time": "2025-10-05T16:00:00Z", "PM2.5": 42, "O3": 0.095, "NO2": 0.032},
        {"time": "2025-10-05T20:00:00Z", "PM2.5": 35, "O3": 0.055, "NO2": 0.025},
    ]
    
    print("Real-time AQI calculations:")
    for measurement in current_measurements:
        time_str = measurement['time']
        concentrations = {k: v for k, v in measurement.items() if k != 'time'}
        
        aqi, dominant = calculator.calculate_aqi_from_concentrations(concentrations)
        category = calculator.get_aqi_category(aqi)
        
        print(f"\n{time_str}")
        print(f"  Measurements: {concentrations}")
        print(f"  AQI: {aqi} ({category})")
        print(f"  Dominant: {dominant}")


def run_all_examples():
    """Run all examples in sequence."""
    print("AQI FORECASTING SYSTEM - EXAMPLES")
    print("="*60)
    
    try:
        # Run examples
        example_preprocessing()
        example_aqi_calculation()
        model = example_model_training()
        example_forecasting()
        example_realtime_aqi()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nFiles created:")
        print("- example_data.csv (sample preprocessing data)")
        print("- training_data.csv (model training data)")
        print("- example_model.pkl (trained model)")
        print("- example_forecast.json (forecast output)")
        print("\nYou can now use these files to explore the system further!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()