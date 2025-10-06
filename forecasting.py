"""
Prediction and output module for AQI forecasting system.
Generates forecasts and produces JSON output for frontend consumption.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
try:
    from data_preprocessing import AQIDataPreprocessor
    from aqi_calculator import EPAAQICalculator
    from ml_models import PollutantPredictor
except ImportError as e:
    print(f"Import warning: {e}")
    # Define placeholder classes if imports fail
    class AQIDataPreprocessor:
        def __init__(self): pass
    class EPAAQICalculator:
        def __init__(self): pass
    class PollutantPredictor:
        def __init__(self, *args, **kwargs): pass


class AQIForecaster:
    """
    Main forecasting class that combines preprocessing, ML prediction, and AQI calculation.
    Generates structured output for frontend consumption.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = 'random_forest'):
        """
        Initialize forecaster with trained model.
        
        Args:
            model_path: Path to saved model (if None, creates new model)
            model_type: Type of ML model to use if creating new
        """
        self.preprocessor = AQIDataPreprocessor()
        self.aqi_calculator = EPAAQICalculator()
        
        if model_path:
            self.predictor = PollutantPredictor.load_model(model_path)
        else:
            self.predictor = PollutantPredictor(model_type)
        
        self.is_trained = hasattr(self.predictor, 'is_fitted') and self.predictor.is_fitted
    
    def train_model(self, df: pd.DataFrame, target_pollutants: Optional[List[str]] = None):
        """
        Train the forecasting model on historical data.
        
        Args:
            df: Historical data DataFrame
            target_pollutants: List of pollutants to predict
        """
        if target_pollutants is None:
            target_pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
        
        # Preprocess data
        features, targets, feature_names = self.preprocessor.preprocess_pipeline(
            df, target_pollutants
        )
        
        # Train model
        self.predictor.fit(features, targets)
        self.is_trained = True
        
        print(f"Model trained on {len(features)} samples")
        print(f"Features: {len(feature_names)} columns")
        print(f"Targets: {list(targets.columns)}")
    
    def generate_forecast(self, df: pd.DataFrame, forecast_hours: int = 24) -> List[Dict]:
        """
        Generate complete AQI forecast with predictions and JSON output.
        
        Args:
            df: Recent historical data for making predictions
            forecast_hours: Number of hours to forecast
            
        Returns:
            List of forecast dictionaries, one per hour
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating forecasts")
        
        # Preprocess recent data (no target_cols to get DataFrame, not tuple)
        df_processed = self.preprocessor.preprocess_pipeline(df, target_cols=None)
        
        # Ensure we have enough data for prediction
        min_required = getattr(self.predictor.params, 'sequence_length', 24)
        if len(df_processed) < min_required:
            raise ValueError(f"Need at least {min_required} hours of recent data")
        
        # Generate predictions for next hours
        predictions_df = self.predictor.predict_next_hours(
            df_processed[self.predictor.feature_names].tail(100),  # Use last 100 hours as context
            forecast_hours
        )
        
        # Generate forecast JSON for each hour
        forecasts = []
        
        for i, (timestamp, row) in enumerate(predictions_df.iterrows()):
            # Prepare pollutant concentrations for AQI calculation
            concentrations = {}
            predicted_pollutants = {}
            
            for pollutant in self.predictor.target_names:
                concentration = row[pollutant]
                
                # Ensure non-negative values
                concentration = max(0, concentration)
                
                # Store for JSON output
                predicted_pollutants[pollutant] = round(concentration, 3)
                
                # Map to AQI calculation format (use appropriate averaging)
                if pollutant in ['PM2.5', 'PM10']:
                    concentrations[f'{pollutant}_24hr_avg'] = concentration
                elif pollutant == 'O3':
                    concentrations['O3_8hr_max'] = concentration
                elif pollutant == 'CO':
                    concentrations['CO_8hr_avg'] = concentration
                elif pollutant in ['NO2', 'SO2']:
                    concentrations[f'{pollutant}_1hr_avg'] = concentration
            
            # Calculate AQI
            aqi_result = self.aqi_calculator.calculate_aqi_from_predictions(concentrations)
            
            # Create forecast entry
            forecast_entry = {
                "time": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "forecast_hour": i + 1,
                "predicted_AQI": aqi_result['AQI'],
                "dominant_pollutant": aqi_result['dominant_pollutant'],
                "AQI_category": aqi_result['AQI_category'],
                "predicted_pollutants": predicted_pollutants,
                "individual_AQIs": aqi_result.get('pollutant_AQIs', {})
            }
            
            forecasts.append(forecast_entry)
        
        return forecasts
    
    def generate_summary_forecast(self, forecasts: List[Dict]) -> Dict:
        """
        Generate summary statistics from hourly forecasts.
        
        Args:
            forecasts: List of hourly forecast dictionaries
            
        Returns:
            Summary forecast dictionary
        """
        if not forecasts:
            return {}
        
        # Extract AQI values
        aqi_values = [f['predicted_AQI'] for f in forecasts if f['predicted_AQI'] is not None]
        
        if not aqi_values:
            return {"error": "No valid AQI predictions"}
        
        # Calculate summary statistics
        max_aqi = max(aqi_values)
        min_aqi = min(aqi_values)
        avg_aqi = sum(aqi_values) / len(aqi_values)
        
        # Find worst hour
        worst_hour = max(forecasts, key=lambda x: x['predicted_AQI'] or 0)
        
        # Count hours by category
        category_counts = {}
        for forecast in forecasts:
            category = forecast.get('AQI_category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Dominant pollutant frequency
        pollutant_counts = {}
        for forecast in forecasts:
            pollutant = forecast.get('dominant_pollutant', 'Unknown')
            pollutant_counts[pollutant] = pollutant_counts.get(pollutant, 0) + 1
        
        most_dominant = max(pollutant_counts, key=pollutant_counts.get)
        
        summary = {
            "forecast_period": {
                "start_time": forecasts[0]['time'],
                "end_time": forecasts[-1]['time'],
                "total_hours": len(forecasts)
            },
            "aqi_summary": {
                "max_aqi": int(max_aqi),
                "min_aqi": int(min_aqi), 
                "avg_aqi": round(avg_aqi, 1),
                "max_category": self.aqi_calculator.get_aqi_category(max_aqi)
            },
            "worst_hour": {
                "time": worst_hour['time'],
                "aqi": worst_hour['predicted_AQI'],
                "category": worst_hour['AQI_category'],
                "dominant_pollutant": worst_hour['dominant_pollutant']
            },
            "category_distribution": category_counts,
            "dominant_pollutants": pollutant_counts,
            "most_frequent_pollutant": most_dominant,
            "health_recommendations": self._get_health_recommendations(max_aqi)
        }
        
        return summary
    
    def _get_health_recommendations(self, max_aqi: int) -> List[str]:
        """Generate health recommendations based on maximum AQI."""
        recommendations = []
        
        if max_aqi <= 50:
            recommendations = [
                "Air quality is good. Normal outdoor activities are safe.",
                "Great day for outdoor exercise and activities."
            ]
        elif max_aqi <= 100:
            recommendations = [
                "Air quality is moderate. Sensitive individuals should consider limiting prolonged outdoor exertion.",
                "Generally safe for most outdoor activities."
            ]
        elif max_aqi <= 150:
            recommendations = [
                "Unhealthy for sensitive groups. Children, elderly, and people with respiratory/heart conditions should limit outdoor activities.",
                "Consider moving vigorous outdoor activities indoors."
            ]
        elif max_aqi <= 200:
            recommendations = [
                "Unhealthy air quality. Everyone should limit prolonged outdoor exertion.",
                "Sensitive individuals should avoid outdoor activities.",
                "Consider wearing masks when outdoors."
            ]
        elif max_aqi <= 300:
            recommendations = [
                "Very unhealthy air quality. All outdoor activities should be avoided.",
                "Stay indoors with windows and doors closed.",
                "Use air purifiers if available."
            ]
        else:
            recommendations = [
                "Hazardous air quality. Emergency conditions affecting entire population.",
                "Avoid all outdoor activities.",
                "Stay indoors with air filtration systems if possible."
            ]
        
        return recommendations
    
    def save_forecast_json(self, forecasts: List[Dict], filepath: str, include_summary: bool = True):
        """
        Save forecasts to JSON file.
        
        Args:
            forecasts: List of forecast dictionaries
            filepath: Output file path
            include_summary: Whether to include summary statistics
        """
        output = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model_type": self.predictor.model_type,
                "forecast_hours": len(forecasts)
            },
            "hourly_forecasts": forecasts
        }
        
        if include_summary:
            output["summary"] = self.generate_summary_forecast(forecasts)
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Forecast saved to {filepath}")
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.predictor.save_model(filepath)
        print(f"Model saved to {filepath}")


def create_forecast_from_csv(csv_path: str, 
                           model_path: Optional[str] = None,
                           forecast_hours: int = 24,
                           output_path: Optional[str] = None) -> List[Dict]:
    """
    Convenience function to generate forecast from CSV data.
    
    Args:
        csv_path: Path to historical data CSV
        model_path: Path to trained model (if None, trains new model)
        forecast_hours: Number of hours to forecast
        output_path: Path to save JSON output (optional)
        
    Returns:
        List of forecast dictionaries
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Initialize forecaster
    forecaster = AQIForecaster(model_path)
    
    # Train model if needed
    if not forecaster.is_trained:
        print("Training new model...")
        # Use 80% of data for training, keep 20% for forecasting context
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        forecaster.train_model(train_df)
    
    # Generate forecast using recent data
    recent_df = df.tail(100)  # Use last 100 hours as context
    forecasts = forecaster.generate_forecast(recent_df, forecast_hours)
    
    # Save to file if requested
    if output_path:
        forecaster.save_forecast_json(forecasts, output_path)
    
    return forecasts


if __name__ == "__main__":
    print("AQI Forecasting and Output Module")
    print("Use AQIForecaster class or create_forecast_from_csv() function")
    print("\nExample usage:")
    print("forecasts = create_forecast_from_csv('data.csv', output_path='forecast.json')")