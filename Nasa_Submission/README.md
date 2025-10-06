# AQI Forecasting System for TEMPO Satellite Data

A comprehensive Python system for forecasting Air Quality Index (AQI) using TEMPO satellite pollutant data, ground station measurements, and weather data.

## Features

- **Data Preprocessing**: Automated rolling averages for EPA AQI calculations
- **Multiple ML Models**: Random Forest, Gradient Boosting, and LSTM options
- **EPA AQI Calculation**: Official EPA formula with all pollutant breakpoints
- **JSON Output**: Frontend-ready forecasts with hourly predictions
- **Modular Design**: Easy to integrate individual components

## Quick Start

### 1. Basic Usage

```python
from forecasting import create_forecast_from_csv

# Generate 24-hour forecast from your CSV data
forecasts = create_forecast_from_csv(
    csv_path='your_data.csv',
    forecast_hours=24,
    output_path='aqi_forecast.json'
)
```

### 2. Command Line Usage

```bash
# Basic forecast
python main.py --data your_data.csv --output forecast.json

# Train new model and forecast
python main.py --data your_data.csv --model-type lstm --forecast-hours 48

# Compare multiple models
python main.py --data your_data.csv --compare-models

# Train only (no forecast)
python main.py --data your_data.csv --train-only --save-model my_model
```

### 3. Run Demo

```bash
# Run with no arguments to see a demonstration
python main.py
```

## CSV Data Format

Your CSV file should contain the following columns:

### Required Columns

| Column | Description | Units | Example |
|--------|-------------|-------|---------|
| time | Timestamp | ISO format | 2025-10-05T14:00:00Z |
| PM2.5 | Fine particulate matter | μg/m³ | 35.2 |
| PM10 | Coarse particulate matter | μg/m³ | 45.8 |
| O3 | Ozone | ppm | 0.072 |
| NO2 | Nitrogen dioxide | ppm | 0.021 |
| SO2 | Sulfur dioxide | ppm | 0.008 |
| CO | Carbon monoxide | ppm | 1.2 |

### Optional Weather Columns

| Column | Description | Units | Example |
|--------|-------------|-------|---------|
| temperature | Air temperature | °C | 22.5 |
| humidity | Relative humidity | % | 65.2 |
| wind_speed | Wind speed | m/s | 3.8 |

### Example CSV Structure

```csv
time,PM2.5,PM10,O3,NO2,SO2,CO,temperature,humidity,wind_speed
2025-10-01T00:00:00Z,25.3,42.1,0.045,0.018,0.005,0.8,18.2,72.5,2.1
2025-10-01T01:00:00Z,23.8,39.7,0.042,0.016,0.004,0.7,17.9,74.1,1.9
2025-10-01T02:00:00Z,22.1,37.2,0.038,0.014,0.003,0.6,17.5,75.8,1.7
...
```

## Output Format

The system generates JSON output with the following structure:

```json
{
  "metadata": {
    "generated_at": "2025-10-05T14:30:00Z",
    "model_type": "random_forest",
    "forecast_hours": 24
  },
  "summary": {
    "forecast_period": {
      "start_time": "2025-10-05T15:00:00Z",
      "end_time": "2025-10-06T14:00:00Z",
      "total_hours": 24
    },
    "aqi_summary": {
      "max_aqi": 112,
      "min_aqi": 45,
      "avg_aqi": 78.5,
      "max_category": "Unhealthy for Sensitive Groups"
    },
    "worst_hour": {
      "time": "2025-10-05T18:00:00Z",
      "aqi": 112,
      "category": "Unhealthy for Sensitive Groups",
      "dominant_pollutant": "PM2.5"
    },
    "health_recommendations": [
      "Unhealthy for sensitive groups...",
      "Consider moving vigorous outdoor activities indoors."
    ]
  },
  "hourly_forecasts": [
    {
      "time": "2025-10-05T15:00:00Z",
      "forecast_hour": 1,
      "predicted_AQI": 98,
      "dominant_pollutant": "PM2.5",
      "AQI_category": "Moderate",
      "predicted_pollutants": {
        "PM2.5": 38.2,
        "PM10": 52.1,
        "O3": 0.072,
        "NO2": 0.021,
        "SO2": 0.008,
        "CO": 1.1
      },
      "individual_AQIs": {
        "PM2.5": 98,
        "PM10": 75,
        "O3": 89,
        "NO2": 45,
        "SO2": 32,
        "CO": 28
      }
    }
    // ... more hourly forecasts
  ]
}
```

## Advanced Usage

### 1. Custom Model Training

```python
from ml_models import PollutantPredictor
from data_preprocessing import load_and_preprocess_data

# Load and preprocess data
features, targets, feature_names = load_and_preprocess_data(
    'your_data.csv', 
    target_cols=['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
)

# Train custom LSTM model
model = PollutantPredictor('lstm', units=100, epochs=100)
model.fit(features, targets)

# Save model
model.save_model('my_lstm_model')
```

### 2. Real-time AQI Calculation

```python
from aqi_calculator import EPAAQICalculator

calculator = EPAAQICalculator()

# Calculate AQI from current measurements
concentrations = {
    'PM2.5': 35.0,  # μg/m³
    'O3': 0.070,    # ppm  
    'NO2': 0.025    # ppm
}

aqi, dominant = calculator.calculate_aqi_from_concentrations(concentrations)
print(f"Current AQI: {aqi}, Dominant pollutant: {dominant}")
```

### 3. Batch Processing

```python
from forecasting import AQIForecaster

# Initialize with trained model
forecaster = AQIForecaster('path/to/saved/model')

# Process multiple datasets
for csv_file in ['site1.csv', 'site2.csv', 'site3.csv']:
    df = pd.read_csv(csv_file)
    forecasts = forecaster.generate_forecast(df, forecast_hours=24)
    
    output_file = csv_file.replace('.csv', '_forecast.json')
    forecaster.save_forecast_json(forecasts, output_file)
```

## Model Performance

The system includes three ML model types optimized for different scenarios:

### Random Forest
- **Best for**: Robust predictions with minimal tuning
- **Pros**: Fast training, handles missing data well
- **Cons**: Limited ability to capture complex temporal patterns

### Gradient Boosting  
- **Best for**: High accuracy on structured data
- **Pros**: Excellent predictive performance
- **Cons**: Longer training time, requires more tuning

### LSTM
- **Best for**: Complex temporal dependencies
- **Pros**: Captures long-term patterns in air quality data
- **Cons**: Requires more data, longer training time

## EPA AQI Calculations

The system implements official EPA AQI calculations with proper averaging periods:

- **PM2.5, PM10**: 24-hour average
- **O3**: 8-hour maximum  
- **CO**: 8-hour average
- **NO2, SO2**: 1-hour average

AQI categories:
- 0-50: Good (Green)
- 51-100: Moderate (Yellow)
- 101-150: Unhealthy for Sensitive Groups (Orange)
- 151-200: Unhealthy (Red)
- 201-300: Very Unhealthy (Purple)
- 301-500: Hazardous (Maroon)

## Integration Tips

### For Web Applications
1. Use the JSON output directly in your frontend
2. Set up automated runs with cron jobs
3. Store forecasts in a database for historical analysis

### For Research
1. Use the modular components for custom analysis
2. Compare different model types with `--compare-models`
3. Export model metrics for publication

### For Operations
1. Set up alerts for high AQI predictions
2. Integrate with air quality monitoring networks
3. Use health recommendations for public advisories

## Troubleshooting

### Common Issues

1. **Missing columns error**: Ensure your CSV has all required pollutant columns
2. **Insufficient data**: Need at least 24-48 hours of data for training
3. **Memory issues with LSTM**: Reduce sequence length or batch size
4. **Poor predictions**: Check data quality and consider more training data

### Performance Optimization

1. **For large datasets**: Use Random Forest for faster training
2. **For better accuracy**: Use Gradient Boosting with cross-validation
3. **For temporal patterns**: Use LSTM with longer sequences

## License

MIT License - feel free to use for your hackathon and beyond!

## Support

For questions about the code or EPA AQI calculations, check the inline documentation in each module.