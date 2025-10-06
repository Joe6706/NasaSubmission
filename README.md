# AQI Forecasting System for TEMPO Satellite Data

A comprehensive Python system for forecasting Air Quality Index (AQI) using TEMPO satellite pollutant data, ground station measurements, and weather data.

## Features

- **Data Preprocessing**: Automated rolling averages for EPA AQI calculations
- **Multiple ML Models**: Random Forest, Gradient Boosting, and LSTM options
- **EPA AQI Calculation**: Official EPA formula with all pollutant breakpoints
- **JSON Output**: Frontend-ready forecasts with hourly predictions
- **Modular Design**: Easy to integrate individual components
- **Web Interface**: Complete web application for visualization and interaction

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AQI-Forecasting-System.git
cd AQI-Forecasting-System

# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 2. Configuration

Copy `.env.example` to `.env` and configure the following:

- **NASA Earthdata Login**: Get credentials from [NASA Earthdata](https://urs.earthdata.nasa.gov/users/new)
- **LAADS DAAC Token**: Get from [LAADS Web](https://ladsweb.modaps.eosdis.nasa.gov/profile/app-keys)
- **OpenAQ API Key**: Get from [OpenAQ Documentation](https://docs.openaq.org)

### 3. Basic Usage

```python
from forecasting import create_forecast_from_csv

# Generate 24-hour forecast from your CSV data
forecasts = create_forecast_from_csv(
    csv_path='your_data.csv',
    forecast_hours=24,
    output_path='aqi_forecast.json'
)
```

### 4. Command Line Usage

```bash
# Basic forecast
python main.py --data your_data.csv --output forecast.json

# Train new model and forecast
python main.py --data your_data.csv --model-type lstm --forecast-hours 48

# Compare multiple models
python main.py --data your_data.csv --compare-models
```

### 5. Web Application

```bash
# Start the web interface
cd web_app
python server.py

# Open browser to http://localhost:5000
```

## Project Structure

```
AQI-Forecasting-System/
├── main.py                 # Main CLI interface
├── data_preprocessing.py   # Data cleaning and preprocessing
├── aqi_calculator.py       # EPA AQI calculation functions
├── ml_models.py           # Machine learning models
├── forecasting.py         # Forecasting pipeline
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
├── web_app/              # Web application
│   ├── server.py         # Flask web server
│   ├── index.html        # Web interface
│   └── ...              # Deployment files
└── docs/                 # Documentation
    ├── API_SPECIFICATION.md
    ├── INTEGRATION_READY.md
    └── READY_TO_USE.md
```

## Data Requirements

The system expects CSV data with the following columns:
- `time`: Timestamp (ISO format)
- `PM2.5`, `PM10`: Particulate matter concentrations (µg/m³)
- `O3`, `NO2`, `SO2`, `CO`: Gas concentrations (various units)
- Weather data (optional): temperature, humidity, wind speed

## Model Types

- **Random Forest**: Fast, reliable baseline model
- **Gradient Boosting**: Better accuracy for complex patterns
- **LSTM**: Neural network for time series patterns

## Output Format

The system generates JSON forecasts compatible with web frontends:

```json
{
  "forecast_metadata": {
    "model_type": "random_forest",
    "forecast_hours": 24,
    "generated_at": "2025-10-06T12:00:00Z"
  },
  "hourly_forecasts": [
    {
      "time": "2025-10-06T13:00:00Z",
      "aqi": 85,
      "category": "Moderate",
      "pollutants": {
        "PM2.5": 25.5,
        "O3": 0.065
      }
    }
  ]
}
```

## API Integration

See [API_SPECIFICATION.md](API_SPECIFICATION.md) for detailed API integration guidelines.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

For questions and support, please open an issue on GitHub.