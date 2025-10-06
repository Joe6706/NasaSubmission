# 🌍 NASA TEMPO AQI Forecasting System - API Specification

## 📊 **Complete JSON Output Format**

Your web app will receive **exactly** this JSON structure from our Python forecasting system:

### **Root JSON Structure:**
```json
{
  "metadata": {
    "generated_at": "2025-10-04T19:03:51Z",
    "model_type": "random_forest", 
    "forecast_hours": 24
  },
  "hourly_forecasts": [
    // Array of 24 hourly forecast objects (see below)
  ]
}
```

### **Metadata Object Properties:**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `generated_at` | string (ISO 8601) | UTC timestamp when forecast was generated | `"2025-10-04T19:03:51Z"` |
| `model_type` | string | ML model used for prediction | `"random_forest"` |
| `forecast_hours` | integer | Number of forecast hours (always 24) | `24` |

### **Hourly Forecast Object (Array Element):**
```json
{
  "time": "2025-10-04T00:00:00Z",
  "forecast_hour": 1,
  "predicted_AQI": 173,
  "dominant_pollutant": "O3",
  "AQI_category": "Unhealthy",
  "predicted_pollutants": {
    "PM2.5": 62.164,
    "PM10": 83.889,
    "O3": 0.095,
    "NO2": 0.041,
    "SO2": 0.004,
    "CO": 1.151
  },
  "individual_AQIs": {}
}
```

### **Hourly Forecast Properties:**
| Field | Type | Range/Values | Description |
|-------|------|--------------|-------------|
| `time` | string (ISO 8601) | UTC timestamps | Hour for this forecast |
| `forecast_hour` | integer | 1-24 | Sequential hour number |
| `predicted_AQI` | integer | 0-500+ | Overall EPA AQI value |
| `dominant_pollutant` | string | `"PM2.5"`, `"PM10"`, `"O3"`, `"NO2"`, `"SO2"`, `"CO"` | Pollutant causing highest AQI |
| `AQI_category` | string | See EPA categories below | Human-readable AQI level |
| `predicted_pollutants` | object | See pollutant units below | Concentration predictions |
| `individual_AQIs` | object | `{}` (empty) | Reserved for future use |

## 🎨 **EPA AQI Categories (Exact Values):**

Your web app must use these **exact** category names and color codes:

| AQI Range | Category | Color Code | RGB | 
|-----------|----------|------------|-----|
| 0-50 | `"Good"` | `#00E400` | `(0, 228, 0)` |
| 51-100 | `"Moderate"` | `#FFFF00` | `(255, 255, 0)` |
| 101-150 | `"Unhealthy for Sensitive Groups"` | `#FF7E00` | `(255, 126, 0)` |
| 151-200 | `"Unhealthy"` | `#FF0000` | `(255, 0, 0)` |
| 201-300 | `"Very Unhealthy"` | `#8F3F97` | `(143, 63, 151)` |
| 301+ | `"Hazardous"` | `#7E0023` | `(126, 0, 35)` |

## 🧪 **Pollutant Concentrations & Units:**

| Pollutant | Key | Units | Typical Range | Description |
|-----------|-----|-------|---------------|-------------|
| Fine Particulates | `PM2.5` | µg/m³ | 0-200 | Particles ≤ 2.5 micrometers |
| Coarse Particulates | `PM10` | µg/m³ | 0-300 | Particles ≤ 10 micrometers |
| Ozone | `O3` | ppm | 0.000-0.200 | Ground-level ozone |
| Nitrogen Dioxide | `NO2` | ppm | 0.000-0.100 | Traffic/industrial emissions |
| Sulfur Dioxide | `SO2` | ppm | 0.000-0.050 | Industrial emissions |
| Carbon Monoxide | `CO` | ppm | 0.0-20.0 | Vehicle emissions |

## 🔧 **TypeScript Interface Definition:**

```typescript
interface ForecastMetadata {
  generated_at: string; // ISO 8601 UTC timestamp
  model_type: "random_forest";
  forecast_hours: 24;
}

interface PredictedPollutants {
  "PM2.5": number;  // µg/m³
  "PM10": number;   // µg/m³
  "O3": number;     // ppm
  "NO2": number;    // ppm
  "SO2": number;    // ppm
  "CO": number;     // ppm
}

type DominantPollutant = "PM2.5" | "PM10" | "O3" | "NO2" | "SO2" | "CO";

type AQICategory = 
  | "Good" 
  | "Moderate" 
  | "Unhealthy for Sensitive Groups"
  | "Unhealthy" 
  | "Very Unhealthy" 
  | "Hazardous";

interface HourlyForecast {
  time: string;                    // ISO 8601 UTC timestamp
  forecast_hour: number;           // 1-24
  predicted_AQI: number;           // 0-500+
  dominant_pollutant: DominantPollutant;
  AQI_category: AQICategory;
  predicted_pollutants: PredictedPollutants;
  individual_AQIs: {};             // Always empty object
}

interface ForecastResponse {
  metadata: ForecastMetadata;
  hourly_forecasts: HourlyForecast[]; // Always 24 elements
}
```

## 📝 **JavaScript Example Usage:**

```javascript
// Fetch forecast from your Python system
const response = await fetch('/api/forecast');
const forecast = await response.json();

// Access metadata
console.log(forecast.metadata.generated_at);     // "2025-10-04T19:03:51Z"
console.log(forecast.metadata.model_type);       // "random_forest"
console.log(forecast.metadata.forecast_hours);   // 24

// Process hourly forecasts (always 24 hours)
forecast.hourly_forecasts.forEach((hour, index) => {
  console.log(`Hour ${hour.forecast_hour}: AQI ${hour.predicted_AQI}`);
  console.log(`Category: ${hour.AQI_category}`);
  console.log(`Dominant: ${hour.dominant_pollutant}`);
  
  // Access specific pollutant concentrations
  const pm25 = hour.predicted_pollutants["PM2.5"];  // µg/m³
  const ozone = hour.predicted_pollutants.O3;       // ppm
});

// Get color for AQI visualization
function getAQIColor(aqi) {
  if (aqi <= 50) return "#00E400";        // Good
  if (aqi <= 100) return "#FFFF00";       // Moderate  
  if (aqi <= 150) return "#FF7E00";       // Unhealthy for Sensitive
  if (aqi <= 200) return "#FF0000";       // Unhealthy
  if (aqi <= 300) return "#8F3F97";       // Very Unhealthy
  return "#7E0023";                       // Hazardous
}
```

## ⚠️ **Important Implementation Notes:**

### **Data Consistency:**
- **Always 24 forecast hours** in the array
- **UTC timestamps** in ISO 8601 format
- **Pollutant concentrations** rounded to 3 decimal places
- **AQI values** are integers (no decimals)
- **individual_AQIs** is always an empty object `{}`

### **Error Handling:**
```javascript
// Handle potential null/undefined values
const safeAQI = hour.predicted_AQI || 0;
const safeCategory = hour.AQI_category || "Unknown";
```

### **Time Zone Conversion:**
```javascript
// Convert UTC to local time for display
const utcTime = new Date(hour.time);
const localTime = utcTime.toLocaleString();
```

## 🚀 **Production Integration:**

Your web app should expect this **exact format** when calling:
```bash
python main.py --data tempo_data.csv --output forecast.json
```

The output file will contain exactly the JSON structure defined above, ready for immediate consumption by your frontend application.

## ✅ **Validation Checklist:**

- [ ] Parse `metadata` object correctly
- [ ] Handle exactly 24 `hourly_forecasts` elements  
- [ ] Use EPA-compliant AQI color coding
- [ ] Display pollutant concentrations with correct units
- [ ] Handle UTC timestamp conversion
- [ ] Implement proper error handling for missing data
- [ ] Validate AQI categories match EPA standards exactly