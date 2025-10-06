# Calculate minimum data requirements for EPA AQI
requirements = {
    'PM2.5': {'window': 24, 'min_percent': 75, 'description': '24-hour average'},
    'PM10': {'window': 24, 'min_percent': 75, 'description': '24-hour average'},
    'O3': {'window': 8, 'min_percent': 75, 'description': '8-hour maximum'},
    'CO': {'window': 8, 'min_percent': 75, 'description': '8-hour average'},
    'NO2': {'window': 1, 'min_percent': 100, 'description': '1-hour value'},
    'SO2': {'window': 1, 'min_percent': 100, 'description': '1-hour value'}
}

print('EPA AQI Minimum Data Requirements:')
print('=' * 50)

max_lookback = 0
for pollutant, req in requirements.items():
    min_hours = int(req['window'] * req['min_percent'] / 100)
    max_lookback = max(max_lookback, req['window'])
    desc = req['description']
    print(f'{pollutant:>6}: {desc:18} | Min hours needed: {min_hours:2d}/{req["window"]:2d}')

print(f'\nMaximum lookback window: {max_lookback} hours')
print(f'Minimum total dataset for ONE valid AQI calculation: {max_lookback} hours')

# For forecasting, we need enough VALID AQI calculations to train
print(f'\nFor ML training:')
print(f'- Need at least 24-48 valid AQI calculations for basic training')
print(f'- Total minimum dataset: {max_lookback + 48} hours ({(max_lookback + 48)/24:.1f} days)')
print(f'- Recommended dataset: {max_lookback + 168} hours ({(max_lookback + 168)/24:.1f} days)')

# Check our current dataset
import pandas as pd
df = pd.read_csv('tempo_sample.csv')
current_hours = len(df)
print(f'\nCurrent dataset: {current_hours} hours ({current_hours/24:.1f} days)')
if current_hours >= max_lookback + 48:
    print('✅ Sufficient data for ML training!')
else:
    needed = max_lookback + 48 - current_hours
    print(f'❌ Need {needed} more hours of data')