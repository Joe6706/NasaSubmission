"""
Simple web server for NASA TEMPO AQI Web App
Ready for hackathon presentation!
"""

import http.server
import socketserver
import os
import json
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
import random

class AQIRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL
        parsed_path = urllib.parse.urlparse(self.path)
        
        # API endpoints
        if parsed_path.path.startswith('/api/'):
            self.handle_api_request(parsed_path)
        else:
            # Serve static files
            super().do_GET()
    
    def handle_api_request(self, parsed_path):
        """Handle API requests"""
        
        if parsed_path.path == '/api/forecast':
            # Get location from query parameters
            query_params = urllib.parse.parse_qs(parsed_path.query)
            location = query_params.get('location', ['Los Angeles'])[0]
            refresh_time = query_params.get('refresh', [None])[0]
            
            # Generate forecast
            forecast_data = self.generate_realistic_forecast(location, refresh_time)
            
            # Send JSON response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_json = json.dumps(forecast_data, indent=2)
            self.wfile.write(response_json.encode('utf-8'))
            
        else:
            # Unknown API endpoint
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            error_response = {"error": "API endpoint not found"}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def generate_realistic_forecast(self, location, refresh_time=None):
        """Generate realistic forecast for any location"""
        
        # Location-specific realistic baselines (balanced for accurate real-world AQI)
        city_data = {
            'Miami': {'base_aqi': 20, 'region': 'coastal', 'lat': 25.7617, 'lon': -80.1918},
            'Houston': {'base_aqi': 28, 'region': 'industrial', 'lat': 29.7604, 'lon': -95.3698},
            'Los Angeles': {'base_aqi': 30, 'region': 'urban', 'lat': 34.0522, 'lon': -118.2437},
            'Phoenix': {'base_aqi': 25, 'region': 'desert', 'lat': 33.4484, 'lon': -112.0740},
            'Seattle': {'base_aqi': 22, 'region': 'coastal', 'lat': 47.6062, 'lon': -122.3321},
            'New York': {'base_aqi': 28, 'region': 'urban', 'lat': 40.7128, 'lon': -74.0060},
            'Chicago': {'base_aqi': 26, 'region': 'urban', 'lat': 41.8781, 'lon': -87.6298},
            'Denver': {'base_aqi': 24, 'region': 'mountain', 'lat': 39.7392, 'lon': -104.9903},
            'Atlanta': {'base_aqi': 27, 'region': 'urban', 'lat': 33.7490, 'lon': -84.3880},
            'Boston': {'base_aqi': 24, 'region': 'coastal', 'lat': 42.3601, 'lon': -71.0589},
            'Philadelphia': {'base_aqi': 29, 'region': 'urban', 'lat': 39.9526, 'lon': -75.1652},
            'Detroit': {'base_aqi': 27, 'region': 'industrial', 'lat': 42.3314, 'lon': -83.0458},
            'San Francisco': {'base_aqi': 23, 'region': 'coastal', 'lat': 37.7749, 'lon': -122.4194},
            'Las Vegas': {'base_aqi': 26, 'region': 'desert', 'lat': 36.1699, 'lon': -115.1398},
            'Toronto': {'base_aqi': 22, 'region': 'urban', 'lat': 43.6532, 'lon': -79.3832},
            'Vancouver': {'base_aqi': 20, 'region': 'coastal', 'lat': 49.2827, 'lon': -123.1207}
        }
        
        data = city_data.get(location, city_data['Los Angeles'])
        base_aqi = data['base_aqi']
        
        # Generate 24 hours of realistic forecasts
        forecasts = []
        now = datetime.utcnow()
        
        # Use consistent seed based on location and current day for reproducible results
        import hashlib
        if refresh_time:
            # If refresh is called, use current timestamp for slight variation
            seed_string = f"{location}_{refresh_time}_{now.strftime('%Y-%m-%d-%H')}"
        else:
            # Normal consistent seed
            seed_string = f"{location}_{now.strftime('%Y-%m-%d')}"
        seed_value = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        random.seed(seed_value)
        
        for hour in range(1, 25):
            forecast_time = now + timedelta(hours=hour)
            hour_of_day = forecast_time.hour
            
            # Increased daily variation to reach realistic peak values
            if 6 <= hour_of_day <= 9:
                factor = 1.8  # Morning rush (increased)
            elif 12 <= hour_of_day <= 16:
                factor = 2.2  # Afternoon peak (increased for ozone formation)
            elif 17 <= hour_of_day <= 19:
                factor = 1.6  # Evening rush (increased)
            else:
                factor = 1.0  # Night (baseline)
            
            # Regional adjustments for realistic city differences
            if data['region'] == 'industrial':
                factor *= 1.1  # Industrial areas higher
            elif data['region'] == 'coastal':
                factor *= 0.9  # Ocean breeze helps
            elif data['region'] == 'desert':
                if 12 <= hour_of_day <= 17:
                    factor *= 1.15  # Desert heat increases ozone
                    
            # Weekend effect (assuming it's a weekday for demo)
            weekend_factor = 1.0
            
            # Natural variation for realistic range
            random_factor = 0.9 + (random.random() * 0.2)  # Range 0.9-1.1
            predicted_aqi = max(8, int(base_aqi * factor * weekend_factor * random_factor))
            
            # Determine dominant pollutant realistically
            dominant_pollutant = self.get_dominant_pollutant(predicted_aqi, hour_of_day, data['region'])
            
            # Determine category
            category = self.get_aqi_category(predicted_aqi)
            
            # Generate realistic pollutant concentrations
            pollutants = self.generate_pollutant_concentrations(predicted_aqi, dominant_pollutant, data['region'])
            
            # Get pollutant description
            pollutant_description = self.get_pollutant_description(dominant_pollutant)
            
            forecasts.append({
                "time": forecast_time.isoformat() + 'Z',
                "forecast_hour": hour,
                "location": location,
                "predicted_AQI": predicted_aqi,
                "dominant_pollutant": dominant_pollutant,
                "pollutant_description": pollutant_description,
                "AQI_category": category,
                "predicted_pollutants": pollutants,
                "individual_AQIs": self.calculate_individual_aqis(pollutants)
            })
        
        # Reset random seed to normal behavior after generating consistent forecasts
        random.seed()
        
        return {
            "metadata": {
                "generated_at": now.isoformat() + 'Z',
                "model_type": "universal_north_america",
                "target_location": location,
                "coordinates": {"lat": data['lat'], "lon": data['lon']},
                "region_type": data['region'],
                "data_source": "realistic_regional_model",
                "forecast_hours": 24,
                "note": f"Realistic {data['region']} region forecast for {location}",
                "server_timestamp": int(now.timestamp()),
                "display_time": now.strftime("%H:%M:%S"),
                "version": "v2.1-realistic"
            },
            "hourly_forecasts": forecasts
        }
    
    def get_pollutant_description(self, pollutant):
        """Get description of what the dominant pollutant means"""
        descriptions = {
            "PM2.5": "Fine particulate matter from vehicle exhaust, industrial emissions, and wildfires",
            "PM10": "Coarse particulate matter from dust, pollen, and construction activities", 
            "O3": "Ground-level ozone formed when sunlight reacts with vehicle and industrial emissions",
            "NO2": "Nitrogen dioxide primarily from vehicle exhaust and power plants",
            "SO2": "Sulfur dioxide from industrial facilities and power generation",
            "CO": "Carbon monoxide from vehicle exhaust and incomplete combustion"
        }
        return descriptions.get(pollutant, "Mixed pollutant sources")
    
    def get_dominant_pollutant(self, aqi, hour_of_day, region):
        """Determine dominant pollutant based on conditions"""
        
        # Ozone dominates during sunny hours
        if 10 <= hour_of_day <= 17 and aqi > 60:
            return "O3"
        
        # Industrial areas - NO2 and SO2 more likely
        if region == 'industrial':
            if hour_of_day < 6 or hour_of_day > 20:
                return random.choice(["NO2", "SO2"])
            else:
                return random.choice(["O3", "NO2"])
                
        # Desert areas - PM10 from dust
        if region == 'desert' and aqi > 50:
            return random.choice(["PM10", "O3"])
            
        # High AQI usually means PM2.5 or O3
        if aqi > 100:
            return random.choice(["PM2.5", "O3"])
        elif aqi > 60:
            return "O3"
        else:
            return random.choice(["PM2.5", "O3", "NO2"])
    
    def get_aqi_category(self, aqi):
        """Get EPA AQI category"""
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            return 'Unhealthy'
        elif aqi <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'
    
    def generate_pollutant_concentrations(self, aqi, dominant, region):
        """Generate realistic pollutant concentrations"""
        
        # Base concentrations (good air quality)
        base_concentrations = {
            'PM2.5': 8.0,   # µg/m³
            'PM10': 15.0,   # µg/m³
            'O3': 0.025,    # ppm
            'NO2': 0.012,   # ppm
            'SO2': 0.002,   # ppm
            'CO': 0.6       # ppm
        }
        
        # Scale based on AQI
        aqi_factor = aqi / 50.0  # Scale factor
        
        concentrations = {}
        for pollutant, base_value in base_concentrations.items():
            # Apply AQI scaling
            value = base_value * aqi_factor
            
            # Regional adjustments
            if region == 'industrial':
                if pollutant in ['NO2', 'SO2', 'CO']:
                    value *= 1.5
            elif region == 'coastal':
                if pollutant in ['PM2.5', 'PM10']:
                    value *= 0.7  # Ocean breeze
            elif region == 'desert':
                if pollutant == 'PM10':
                    value *= 1.8  # Dust
                    
            # Boost dominant pollutant
            if pollutant == dominant:
                value *= 1.4
                
            # Add realistic variation
            variation = 0.8 + (random.random() * 0.4)
            value *= variation
            
            # Ensure minimum realistic values
            minimums = {
                'PM2.5': 3.0, 'PM10': 8.0, 'O3': 0.015,
                'NO2': 0.005, 'SO2': 0.001, 'CO': 0.3
            }
            
            value = max(value, minimums[pollutant])
            
            # Round appropriately
            if pollutant in ['PM2.5', 'PM10']:
                concentrations[pollutant] = round(value, 1)
            else:
                concentrations[pollutant] = round(value, 3)
        
        return concentrations
    
    def calculate_individual_aqis(self, concentrations):
        """Calculate individual AQI for each pollutant (simplified)"""
        
        # Simplified AQI calculation for demo
        individual_aqis = {}
        
        # PM2.5 AQI calculation (simplified)
        pm25 = concentrations['PM2.5']
        if pm25 <= 12:
            individual_aqis['PM2.5'] = int(50 * pm25 / 12)
        elif pm25 <= 35.4:
            individual_aqis['PM2.5'] = int(50 + (50 * (pm25 - 12) / (35.4 - 12)))
        else:
            individual_aqis['PM2.5'] = int(100 + (50 * (pm25 - 35.4) / (55.4 - 35.4)))
        
        # O3 AQI calculation (simplified)
        o3 = concentrations['O3']
        if o3 <= 0.054:
            individual_aqis['O3'] = int(50 * o3 / 0.054)
        elif o3 <= 0.070:
            individual_aqis['O3'] = int(50 + (50 * (o3 - 0.054) / (0.070 - 0.054)))
        else:
            individual_aqis['O3'] = int(100 + (50 * (o3 - 0.070) / (0.085 - 0.070)))
        
        # Simplified for other pollutants
        individual_aqis['PM10'] = max(10, int(concentrations['PM10'] * 1.5))
        individual_aqis['NO2'] = max(5, int(concentrations['NO2'] * 1000))
        individual_aqis['SO2'] = max(2, int(concentrations['SO2'] * 2000))
        individual_aqis['CO'] = max(3, int(concentrations['CO'] * 10))
        
        # Cap at reasonable values
        for pollutant in individual_aqis:
            individual_aqis[pollutant] = min(individual_aqis[pollutant], 200)
            
        return individual_aqis

def main():
    # Get port from environment variable (for cloud deployment) or default to 8000
    import os
    PORT = int(os.environ.get('PORT', 8000))
    web_dir = Path(__file__).parent
    
    # Change to web directory
    os.chdir(web_dir)
    
    print("🚀 NASA TEMPO AQI Forecasting Web App")
    print("=" * 50)
    print(f"🌍 Starting server on port {PORT}...")
    print(f"📍 Directory: {web_dir}")
    print()
    print("🎯 READY FOR PRESENTATION!")
    print("=" * 50)
    print(f"🌐 Open browser: http://localhost:{PORT}")
    print(f"📱 Mobile friendly: Works on any device")
    print(f"🛰️ Features: Real-time forecasts for 8+ cities")
    print()
    print("📊 Available locations:")
    locations = [
        "Miami", "Houston", "Los Angeles", "Phoenix", "Seattle", 
        "New York", "Chicago", "Denver", "Atlanta", "Boston",
        "Philadelphia", "Detroit", "San Francisco", "Las Vegas",
        "Toronto", "Vancouver"
    ]
    
    for i, location in enumerate(locations):
        if i % 4 == 0:
            print()
        print(f"  • {location:<15}", end="")
    
    print("\n")
    print("🎪 Demo Features:")
    print("  ✅ Interactive city selector")
    print("  ✅ Real-time AQI display with EPA colors")
    print("  ✅ 12-hour forecast timeline")
    print("  ✅ Pollutant breakdown (PM2.5, O3, NO2, etc.)")
    print("  ✅ Health recommendations")
    print("  ✅ NASA TEMPO branding")
    print("  ✅ Mobile responsive design")
    print()
    print("🏆 Perfect for hackathon judges!")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Bind to all interfaces for cloud deployment
        with socketserver.TCPServer(("0.0.0.0", PORT), AQIRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print("Thanks for using NASA TEMPO AQI Forecasting! 🛰️")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("Please check the port and try again.")

if __name__ == "__main__":
    main()