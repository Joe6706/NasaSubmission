import csv
import random
from datetime import datetime, timedelta

# Read existing data
with open('tempo_sample.csv', 'r') as f:
    reader = csv.DictReader(f)
    existing_data = list(reader)

print(f'Current data: {len(existing_data)} hours')

# Generate additional historical data (going backwards)
start_time = datetime.fromisoformat('2025-10-01T00:00:00Z'.replace('Z', '+00:00'))
extended_start = start_time - timedelta(days=7)

# Generate 7 days (168 hours) of historical data
new_data = []
for i in range(168):
    timestamp = extended_start + timedelta(hours=i)
    hour = timestamp.hour
    
    # Daily patterns for pollutants
    pm25_base = 25 + 15 * (1 if 6 <= hour <= 22 else 0.5)
    pm10_base = 35 + 20 * (1 if 6 <= hour <= 22 else 0.5)
    o3_base = 0.04 + 0.03 * (1 if 10 <= hour <= 16 else 0.2)
    no2_base = 0.025 + 0.015 * (1 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0.3)
    so2_base = 0.008
    co_base = 1.0 + 0.5 * (1 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0.3)
    
    temp_base = 20 + 8 * (1 if 10 <= hour <= 16 else 0.3)
    humidity_base = 60 - 20 * (1 if 10 <= hour <= 16 else 0)
    wind_base = 3 + 2 * random.random()
    
    row = {
        'time': timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
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
    new_data.append(row)

# Combine data
all_data = new_data + existing_data

# Write extended dataset
with open('tempo_sample.csv', 'w', newline='') as f:
    fieldnames = ['time','PM2.5','PM10','O3','NO2','SO2','CO','temperature','humidity','wind_speed']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_data)

print(f'Extended data: {len(all_data)} hours ({len(all_data)/24:.1f} days)')
print(f'Date range: {all_data[0]["time"]} to {all_data[-1]["time"]}')