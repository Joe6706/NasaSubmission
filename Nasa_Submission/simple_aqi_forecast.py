"""
Simplified AQI Forecasting System
Works without external dependencies for quick testing and demonstration.
"""

import json
import csv
import random
from datetime import datetime, timedelta
from math import sin, pi


class SimpleAQICalculator:
    """Simple EPA AQI calculator without external dependencies."""
    
    def __init__(self):
        # EPA AQI Breakpoints
        self.breakpoints = {
            'PM2.5': [
                (0.0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500)
            ],
            'PM10': [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 504, 301, 400),
                (505, 604, 401, 500)
            ],
            'O3': [
                (0.000, 0.054, 0, 50),
                (0.055, 0.070, 51, 100),
                (0.071, 0.085, 101, 150),
                (0.086, 0.105, 151, 200),
                (0.106, 0.200, 201, 300),
                (0.201, 0.300, 301, 400),
                (0.301, 0.500, 401, 500)
            ],
            'NO2': [
                (0.000, 0.053, 0, 50),
                (0.054, 0.100, 51, 100),
                (0.101, 0.360, 101, 150),
                (0.361, 0.649, 151, 200),
                (0.650, 1.249, 201, 300),
                (1.250, 1.649, 301, 400),
                (1.650, 2.049, 401, 500)
            ],
            'SO2': [
                (0.000, 0.035, 0, 50),
                (0.036, 0.075, 51, 100),
                (0.076, 0.185, 101, 150),
                (0.186, 0.304, 151, 200),
                (0.305, 0.604, 201, 300),
                (0.605, 0.804, 301, 400),
                (0.805, 1.004, 401, 500)
            ],
            'CO': [
                (0.0, 4.4, 0, 50),
                (4.5, 9.4, 51, 100),
                (9.5, 12.4, 101, 150),
                (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300),
                (30.5, 40.4, 301, 400),
                (40.5, 50.4, 401, 500)
            ]
        }
    
    def calculate_pollutant_aqi(self, concentration, pollutant):
        """Calculate AQI for a single pollutant."""
        if pollutant not in self.breakpoints:
            return None
        
        breakpoints = self.breakpoints[pollutant]
        
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= concentration <= c_high:
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                return int(round(aqi))
        
        return 500  # Maximum AQI
    
    def calculate_overall_aqi(self, concentrations):
        """Calculate overall AQI and dominant pollutant."""
        aqi_values = {}
        
        for pollutant, concentration in concentrations.items():
            if concentration is not None:
                aqi = self.calculate_pollutant_aqi(concentration, pollutant)
                if aqi is not None:
                    aqi_values[pollutant] = aqi
        
        if not aqi_values:
            return None, "No Data"
        
        overall_aqi = max(aqi_values.values())
        dominant_pollutant = max(aqi_values, key=aqi_values.get)
        
        return overall_aqi, dominant_pollutant
    
    def get_aqi_category(self, aqi):
        """Get AQI health category."""
        if aqi is None:
            return "No Data"
        elif aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"


class SimpleForecastModel:
    """Simple forecasting model using basic trends."""
    
    def __init__(self):
        self.historical_data = []
        random.seed(42)  # For reproducible results
    
    def train(self, data):
        """Train on historical data."""
        self.historical_data = data
        print(f"Model trained on {len(data)} data points")
    
    def predict_next_hours(self, n_hours=24):
        """Predict next n hours using simple trend extrapolation."""
        if not self.historical_data:
            return []
        
        predictions = []
        last_data = self.historical_data[-1]
        base_time = datetime.fromisoformat(last_data['time'].replace('Z', '+00:00'))
        
        for hour in range(1, n_hours + 1):
            pred_time = base_time + timedelta(hours=hour)
            
            # Simple prediction using last values with some variation
            predicted = {
                'time': pred_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'PM2.5': max(0, last_data['PM2.5'] + random.uniform(-5, 5)),
                'PM10': max(0, last_data['PM10'] + random.uniform(-8, 8)),
                'O3': max(0, last_data['O3'] + random.uniform(-0.01, 0.01)),
                'NO2': max(0, last_data['NO2'] + random.uniform(-0.005, 0.005)),
                'SO2': max(0, last_data['SO2'] + random.uniform(-0.002, 0.002)),
                'CO': max(0, last_data['CO'] + random.uniform(-0.2, 0.2))
            }
            
            predictions.append(predicted)
        
        return predictions


def load_csv_data(filename):
    """Load data from CSV file."""
    data = []
    
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert numeric columns
                numeric_cols = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 
                               'temperature', 'humidity', 'wind_speed']
                
                for col in numeric_cols:
                    if col in row:
                        try:
                            row[col] = float(row[col])
                        except (ValueError, TypeError):
                            row[col] = None
                
                data.append(row)
        
        print(f"Loaded {len(data)} records from {filename}")
        return data
        
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []


def create_sample_data(filename='sample_tempo_data.csv', hours=72):
    """Create sample TEMPO satellite data."""
    print(f"Creating {hours} hours of sample TEMPO data...")
    
    start_time = datetime(2025, 10, 1)
    random.seed(42)
    
    with open(filename, 'w', newline='') as file:
        fieldnames = ['time', 'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 
                     'temperature', 'humidity', 'wind_speed']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for hour in range(hours):
            timestamp = start_time + timedelta(hours=hour)
            
            # Generate realistic pollutant values with daily patterns
            hour_of_day = hour % 24
            daily_cycle = sin(2 * pi * hour_of_day / 24)
            
            # PM2.5: higher during rush hours and calm conditions
            pm25_base = 25 + 10 * daily_cycle
            pm25 = max(0, pm25_base + random.uniform(-8, 8))
            
            # PM10: typically 1.3-1.8x PM2.5
            pm10 = pm25 * random.uniform(1.3, 1.8)
            
            # O3: peaks in afternoon due to photochemical processes
            o3_base = 0.05 + 0.02 * sin(2 * pi * (hour_of_day - 6) / 24)
            o3 = max(0, o3_base + random.uniform(-0.01, 0.01))
            
            # NO2: higher during rush hours
            no2_base = 0.025
            if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
                no2_base += 0.01
            no2 = max(0, no2_base + random.uniform(-0.005, 0.005))
            
            # SO2: relatively stable, low levels
            so2 = max(0, 0.008 + random.uniform(-0.003, 0.003))
            
            # CO: higher during rush hours and winter
            co_base = 1.0
            if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
                co_base += 0.3
            co = max(0, co_base + random.uniform(-0.2, 0.2))
            
            # Weather data
            temp = 20 + 5 * sin(2 * pi * hour_of_day / 24) + random.uniform(-3, 3)
            humidity = 60 + 20 * sin(2 * pi * hour_of_day / 24 + pi) + random.uniform(-10, 10)
            humidity = max(0, min(100, humidity))
            wind_speed = max(0, 3 + random.uniform(-2, 4))
            
            row = {
                'time': timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'PM2.5': round(pm25, 1),
                'PM10': round(pm10, 1),
                'O3': round(o3, 4),
                'NO2': round(no2, 4),
                'SO2': round(so2, 4),
                'CO': round(co, 2),
                'temperature': round(temp, 1),
                'humidity': round(humidity, 1),
                'wind_speed': round(wind_speed, 1)
            }
            
            writer.writerow(row)
    
    print(f"Sample data saved to {filename}")
    return filename


def run_aqi_forecast(csv_file, output_file='aqi_forecast.json', forecast_hours=24):
    """Run complete AQI forecast pipeline."""
    print(f"\\nRunning AQI Forecast Pipeline")
    print("=" * 50)
    
    # Load data
    data = load_csv_data(csv_file)
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Date range: {data[0]['time']} to {data[-1]['time']}")
    
    # Initialize components
    aqi_calc = SimpleAQICalculator()
    model = SimpleForecastModel()
    
    # Train model (use 80% of data for training)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    model.train(train_data)
    
    # Generate predictions
    print(f"Generating {forecast_hours}-hour forecast...")
    predictions = model.predict_next_hours(forecast_hours)
    
    # Calculate AQI for each prediction
    forecasts = []
    aqi_values = []
    
    for i, pred in enumerate(predictions):
        # Extract pollutant concentrations
        concentrations = {
            'PM2.5': pred['PM2.5'],
            'PM10': pred['PM10'],
            'O3': pred['O3'],
            'NO2': pred['NO2'],
            'SO2': pred['SO2'],
            'CO': pred['CO']
        }
        
        # Calculate AQI
        overall_aqi, dominant = aqi_calc.calculate_overall_aqi(concentrations)
        category = aqi_calc.get_aqi_category(overall_aqi)
        
        # Create forecast entry
        forecast_entry = {
            "time": pred['time'],
            "forecast_hour": i + 1,
            "predicted_AQI": overall_aqi,
            "dominant_pollutant": dominant,
            "AQI_category": category,
            "predicted_pollutants": {
                "PM2.5": round(pred['PM2.5'], 1),
                "PM10": round(pred['PM10'], 1),
                "O3": round(pred['O3'], 4),
                "NO2": round(pred['NO2'], 4),
                "SO2": round(pred['SO2'], 4),
                "CO": round(pred['CO'], 2)
            }
        }
        
        forecasts.append(forecast_entry)
        if overall_aqi is not None:
            aqi_values.append(overall_aqi)
    
    # Generate summary
    if aqi_values:
        max_aqi = max(aqi_values)
        min_aqi = min(aqi_values)
        avg_aqi = sum(aqi_values) / len(aqi_values)
        worst_hour = max(forecasts, key=lambda x: x['predicted_AQI'] or 0)
        
        summary = {
            "forecast_period": {
                "start_time": forecasts[0]['time'],
                "end_time": forecasts[-1]['time'],
                "total_hours": len(forecasts)
            },
            "aqi_summary": {
                "max_aqi": max_aqi,
                "min_aqi": min_aqi,
                "avg_aqi": round(avg_aqi, 1),
                "max_category": aqi_calc.get_aqi_category(max_aqi)
            },
            "worst_hour": {
                "time": worst_hour['time'],
                "aqi": worst_hour['predicted_AQI'],
                "category": worst_hour['AQI_category'],
                "dominant_pollutant": worst_hour['dominant_pollutant']
            }
        }
    else:
        summary = {"error": "No valid AQI predictions generated"}
    
    # Create output
    output = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_type": "simple_trend",
            "forecast_hours": forecast_hours,
            "source_data": csv_file
        },
        "summary": summary,
        "hourly_forecasts": forecasts
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Forecast saved to {output_file}")
    
    # Display summary
    if aqi_values:
        print(f"\\nForecast Summary:")
        print(f"Max AQI: {max_aqi} ({aqi_calc.get_aqi_category(max_aqi)})")
        print(f"Average AQI: {avg_aqi:.1f}")
        print(f"Worst hour: {worst_hour['time']}")
        
        print(f"\\nFirst 3 forecasts:")
        for forecast in forecasts[:3]:
            print(f"  {forecast['time']}: AQI {forecast['predicted_AQI']} ({forecast['AQI_category']})")
    
    return forecasts


def main():
    """Main function."""
    print("Simplified AQI Forecasting System")
    print("=" * 60)
    print("Works without external dependencies!")
    
    # Create sample data
    sample_file = create_sample_data('tempo_sample.csv', 72)
    
    # Run forecast
    forecasts = run_aqi_forecast(sample_file, 'simple_forecast.json', 24)
    
    print(f"\\n🎉 Success! Generated {len(forecasts)} hourly forecasts")
    print("\\nFiles created:")
    print("- tempo_sample.csv (sample TEMPO data)")
    print("- simple_forecast.json (AQI forecast)")
    print("\\nTo use with your own data:")
    print("1. Replace 'tempo_sample.csv' with your TEMPO data file")
    print("2. Run: python simple_aqi_forecast.py")


if __name__ == "__main__":
    main()