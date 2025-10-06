"""
Main AQI Forecasting Script
End-to-end pipeline for TEMPO satellite data AQI forecasting.

This script provides a complete workflow from raw CSV data to AQI predictions
and JSON output for frontend consumption.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    from data_preprocessing import load_and_preprocess_data, AQIDataPreprocessor
    from aqi_calculator import EPAAQICalculator, calculate_aqi_for_dataframe
    from ml_models import PollutantPredictor, train_multiple_models
    from forecasting import AQIForecaster, create_forecast_from_csv
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='AQI Forecasting System for TEMPO Satellite Data')
    
    # Input/Output arguments
    parser.add_argument('--data', '-d', required=True, 
                       help='Path to CSV file with hourly pollutant and weather data')
    parser.add_argument('--output', '-o', default='aqi_forecast.json',
                       help='Output path for forecast JSON (default: aqi_forecast.json)')
    parser.add_argument('--model-path', '-m', default=None,
                       help='Path to saved model (if not provided, trains new model)')
    
    # Model configuration
    parser.add_argument('--model-type', choices=['random_forest', 'gradient_boosting', 'lstm'],
                       default='random_forest', help='Type of ML model to use')
    parser.add_argument('--forecast-hours', type=int, default=24,
                       help='Number of hours to forecast (default: 24)')
    
    # Data configuration  
    parser.add_argument('--target-pollutants', nargs='+', 
                       default=['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO'],
                       help='Pollutants to predict')
    
    # Options
    parser.add_argument('--train-only', action='store_true',
                       help='Only train model, do not generate forecast')
    parser.add_argument('--compare-models', action='store_true',
                       help='Train and compare multiple model types')
    parser.add_argument('--save-model', default=None,
                       help='Path to save trained model')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found")
        return 1
    
    try:
        # Load and validate data
        print(f"Loading data from {args.data}...")
        df = pd.read_csv(args.data)
        
        if args.verbose:
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        
        # Validate required columns
        required_cols = ['time'] + args.target_pollutants
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return 1
        
        # Compare models if requested
        if args.compare_models:
            print("Comparing multiple model types...")
            
            # Preprocess data for model comparison
            preprocessor = AQIDataPreprocessor()
            features, targets, feature_names = preprocessor.preprocess_pipeline(
                df, args.target_pollutants
            )
            
            # Train multiple models
            models = train_multiple_models(features, targets)
            
            # Display comparison
            print("\nModel Comparison Results:")
            print("-" * 50)
            for model_name, model in models.items():
                if hasattr(model, 'test_metrics'):
                    print(f"\n{model_name.upper()}:")
                    for pollutant, metrics in model.test_metrics.items():
                        print(f"  {pollutant}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")
            
            # Ask user to select best model
            if not args.train_only:
                print("\nSelect model for forecasting:")
                for i, name in enumerate(models.keys(), 1):
                    print(f"{i}. {name}")
                
                while True:
                    try:
                        choice = int(input("Enter model number: ")) - 1
                        selected_model = list(models.keys())[choice]
                        break
                    except (ValueError, IndexError):
                        print("Invalid choice. Please try again.")
                
                # Use selected model for forecasting
                forecaster = AQIForecaster()
                forecaster.predictor = models[selected_model]
                forecaster.is_trained = True
        
        # Single model workflow
        else:
            if args.model_path and os.path.exists(args.model_path + '.pkl'):
                print(f"Loading model from {args.model_path}...")
                forecaster = AQIForecaster(args.model_path)
            else:
                print(f"Training new {args.model_type} model...")
                forecaster = AQIForecaster(model_type=args.model_type)
                
                # Split data for training (use 80% for training)
                train_size = int(len(df) * 0.8)
                train_df = df.iloc[:train_size]
                
                forecaster.train_model(train_df, args.target_pollutants)
                
                # Save model if requested
                if args.save_model:
                    forecaster.save_model(args.save_model)
                    print(f"Model saved to {args.save_model}")
        
        # Generate forecast if not train-only
        if not args.train_only:
            print(f"Generating {args.forecast_hours}-hour forecast...")
            
            # Use recent data for forecasting
            recent_df = df.tail(100)  # Last 100 hours as context
            
            forecasts = forecaster.generate_forecast(recent_df, args.forecast_hours)
            
            # Save forecast to JSON
            forecaster.save_forecast_json(forecasts, args.output, include_summary=True)
            
            # Display summary
            summary = forecaster.generate_summary_forecast(forecasts)
            print("\nForecast Summary:")
            print("-" * 40)
            print(f"Period: {summary['forecast_period']['start_time']} to {summary['forecast_period']['end_time']}")
            print(f"Max AQI: {summary['aqi_summary']['max_aqi']} ({summary['aqi_summary']['max_category']})")
            print(f"Average AQI: {summary['aqi_summary']['avg_aqi']}")
            print(f"Most frequent pollutant: {summary['most_frequent_pollutant']}")
            
            print(f"\nForecast saved to {args.output}")
            
            if args.verbose:
                print("\nFirst few forecast entries:")
                for i, forecast in enumerate(forecasts[:3]):
                    print(f"Hour {i+1}: AQI={forecast['predicted_AQI']}, "
                          f"Dominant={forecast['dominant_pollutant']}")
        
        print("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def quick_demo():
    """Quick demonstration with synthetic data."""
    print("AQI Forecasting System Demo")
    print("Generating synthetic data for demonstration...")
    
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=168, freq='H')  # 1 week
    np.random.seed(42)
    
    # Synthetic pollutant data with realistic ranges
    data = {
        'time': dates,
        'PM2.5': np.random.normal(25, 10, 168).clip(0, 100),
        'PM10': np.random.normal(40, 15, 168).clip(0, 150),
        'O3': np.random.normal(0.05, 0.02, 168).clip(0, 0.15),
        'NO2': np.random.normal(0.03, 0.01, 168).clip(0, 0.1),
        'SO2': np.random.normal(0.01, 0.005, 168).clip(0, 0.05),
        'CO': np.random.normal(1.0, 0.5, 168).clip(0, 5),
        'temperature': np.random.normal(20, 5, 168),
        'humidity': np.random.normal(60, 20, 168).clip(0, 100),
        'wind_speed': np.random.normal(3, 2, 168).clip(0, 20)
    }
    
    df = pd.DataFrame(data)
    
    # Save demo data
    demo_file = 'demo_data.csv'
    df.to_csv(demo_file, index=False)
    print(f"Demo data saved to {demo_file}")
    
    # Run forecasting
    try:
        forecasts = create_forecast_from_csv(
            demo_file, 
            forecast_hours=12,
            output_path='demo_forecast.json'
        )
        
        print(f"Generated {len(forecasts)} hourly forecasts")
        print("Demo completed! Check demo_forecast.json for results")
        
        # Clean up
        os.remove(demo_file)
        
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    import numpy as np
    
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        print("No arguments provided. Running demo...")
        quick_demo()
    else:
        # Run main pipeline
        sys.exit(main())