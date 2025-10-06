import csv
import random
from datetime import datetime, timedelta

# North American cities for the sample data
north_american_cities = [
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
    'Toronto', 'Montreal', 'Vancouver', 'Calgary', 'Ottawa',
    'Mexico City', 'Guadalajara', 'Monterrey', 'Puebla', 'Tijuana'
]

def create_city_sample_data():
    """Create sample data with city information for North American cities."""
    
    # Create 10 days of data for multiple cities
    start_time = datetime.fromisoformat('2025-09-24T00:00:00Z'.replace('Z', '+00:00'))
    hours = 240  # 10 days
    
    all_data = []
    
    # Select a few cities for the sample
    sample_cities = ['Toronto', 'New York', 'Los Angeles', 'Chicago', 'Mexico City']
    
    for city in sample_cities:
        print(f"Generating data for {city}...")
        
        # City-specific pollution patterns
        city_factors = {
            'Toronto': {'pm_factor': 1.0, 'ozone_factor': 0.9, 'no2_factor': 1.1},
            'New York': {'pm_factor': 1.3, 'ozone_factor': 1.2, 'no2_factor': 1.4},
            'Los Angeles': {'pm_factor': 1.5, 'ozone_factor': 1.8, 'no2_factor': 1.6},
            'Chicago': {'pm_factor': 1.1, 'ozone_factor': 1.0, 'no2_factor': 1.2},
            'Mexico City': {'pm_factor': 2.0, 'ozone_factor': 2.2, 'no2_factor': 1.8}
        }
        
        factors = city_factors.get(city, {'pm_factor': 1.0, 'ozone_factor': 1.0, 'no2_factor': 1.0})
        
        for i in range(hours):
            timestamp = start_time + timedelta(hours=i)
            hour = timestamp.hour
            
            # Daily patterns with city-specific factors
            pm25_base = 25 * factors['pm_factor'] + 15 * (1 if 6 <= hour <= 22 else 0.5)
            pm10_base = 35 * factors['pm_factor'] + 20 * (1 if 6 <= hour <= 22 else 0.5)
            o3_base = 0.04 * factors['ozone_factor'] + 0.03 * (1 if 10 <= hour <= 16 else 0.2)
            no2_base = 0.025 * factors['no2_factor'] + 0.015 * (1 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0.3)
            so2_base = 0.008
            co_base = 1.0 + 0.5 * (1 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0.3)
            
            # Weather patterns (city-specific)
            temp_base = 20 + 8 * (1 if 10 <= hour <= 16 else 0.3)
            humidity_base = 60 - 20 * (1 if 10 <= hour <= 16 else 0)
            wind_base = 3 + 2 * random.random()
            
            row = {
                'time': timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'city': city,
                'PM2.5': round(pm25_base + random.uniform(-8, 8), 1),
                'PM10': round(pm10_base + random.uniform(-12, 12), 1),
                'O3': round(o3_base + random.uniform(-0.02, 0.02), 4),
                'NO2': round(no2_base + random.uniform(-0.01, 0.01), 4),
                'SO2': round(so2_base + random.uniform(-0.004, 0.004), 4),
                'CO': round(co_base + random.uniform(-0.3, 0.3), 2),
                'temperature': round(temp_base + random.uniform(-5, 5), 1),
                'humidity': round(max(20, min(90, humidity_base + random.uniform(-15, 15))), 1),
                'wind_speed': round(max(0, wind_base + random.uniform(-2, 2)), 1)
            }
            all_data.append(row)
    
    # Sort by time
    all_data.sort(key=lambda x: x['time'])
    
    # Write to file
    with open('tempo_sample_with_cities.csv', 'w', newline='') as f:
        fieldnames = ['time','city','PM2.5','PM10','O3','NO2','SO2','CO','temperature','humidity','wind_speed']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f'Created tempo_sample_with_cities.csv with {len(all_data)} rows')
    print(f'Cities included: {sample_cities}')
    print(f'Time range: {all_data[0]["time"]} to {all_data[-1]["time"]}')
    
    return 'tempo_sample_with_cities.csv'

if __name__ == "__main__":
    create_city_sample_data()