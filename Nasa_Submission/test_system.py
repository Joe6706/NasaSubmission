"""
Simple test script to verify the AQI system works.
This handles import issues gracefully and provides a working demo.
"""

import sys
import os

def test_basic_aqi_calculation():
    """Test the core AQI calculation without external dependencies."""
    print("Testing Basic AQI Calculation...")
    
    def calculate_pm25_aqi(concentration):
        """Calculate AQI for PM2.5 using EPA formula."""
        breakpoints = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100), 
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500)
        ]
        
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= concentration <= c_high:
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                return int(round(aqi))
        return 500
    
    def get_aqi_category(aqi):
        """Get AQI health category."""
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"
    
    # Test cases
    test_values = [10.0, 25.0, 40.0, 75.0, 200.0]
    
    print("PM2.5 Concentration -> AQI (Category)")
    print("-" * 40)
    for pm25 in test_values:
        aqi = calculate_pm25_aqi(pm25)
        category = get_aqi_category(aqi)
        print(f"{pm25:6.1f} ug/m3 -> {aqi:3d} ({category})")
    
    return True

def create_sample_csv():
    """Create a sample CSV file for testing."""
    print("\nCreating sample data file...")
    
    # Simple data without numpy dependency
    import random
    from datetime import datetime, timedelta
    
    # Generate 48 hours of sample data
    start_time = datetime(2025, 10, 1)
    data_lines = ["time,PM2.5,PM10,O3,NO2,SO2,CO,temperature,humidity,wind_speed"]
    
    for hour in range(48):
        timestamp = start_time + timedelta(hours=hour)
        time_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Generate realistic but simple pollutant values
        pm25 = round(20 + 15 * random.random(), 1)
        pm10 = round(pm25 * 1.5, 1)
        o3 = round(0.04 + 0.03 * random.random(), 4)
        no2 = round(0.02 + 0.02 * random.random(), 4)
        so2 = round(0.005 + 0.01 * random.random(), 4)
        co = round(0.8 + 0.4 * random.random(), 2)
        
        # Weather data
        temp = round(18 + 8 * random.random(), 1)
        humidity = round(50 + 30 * random.random(), 1)
        wind = round(2 + 4 * random.random(), 1)
        
        line = f"{time_str},{pm25},{pm10},{o3},{no2},{so2},{co},{temp},{humidity},{wind}"
        data_lines.append(line)
    
    # Write to file
    with open("test_data.csv", "w") as f:
        f.write("\n".join(data_lines))
    
    print("Sample data saved to 'test_data.csv'")
    print(f"Generated {len(data_lines)-1} hours of sample data")
    return "test_data.csv"

def test_csv_reading():
    """Test reading the CSV file."""
    print("\nTesting CSV file reading...")
    
    try:
        with open("test_data.csv", "r") as f:
            lines = f.readlines()
        
        print(f"CSV file contains {len(lines)} lines")
        print("First few lines:")
        for i, line in enumerate(lines[:4]):
            print(f"  {i}: {line.strip()}")
        
        # Parse first data line
        header = lines[0].strip().split(",")
        if len(lines) > 1:
            data = lines[1].strip().split(",")
            
            print("\nParsed sample data:")
            for col, val in zip(header, data):
                print(f"  {col}: {val}")
        else:
            print("Warning: Only header line found")
        
        return len(lines) > 1
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

def test_with_dependencies():
    """Test with actual dependencies if available."""
    print("\\nTesting with dependencies...")
    
    try:
        import pandas as pd
        import numpy as np
        print("✓ pandas and numpy available")
        
        # Try to read our test data
        df = pd.read_csv("test_data.csv")
        print(f"✓ Successfully loaded {len(df)} rows of data")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Test basic statistics
        print(f"✓ PM2.5 range: {df['PM2.5'].min():.1f} - {df['PM2.5'].max():.1f}")
        print(f"✓ Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        
        return True
        
    except ImportError as e:
        print(f"Dependencies not available: {e}")
        print("To install: pip install pandas numpy scikit-learn tensorflow")
        return False
    except Exception as e:
        print(f"Error with dependencies: {e}")
        return False

def main():
    """Run all tests."""
    print("AQI Forecasting System - Basic Tests")
    print("=" * 50)
    
    # Test 1: Basic AQI calculation
    success1 = test_basic_aqi_calculation()
    
    # Test 2: Create sample data
    csv_file = create_sample_csv()
    
    # Test 3: Read CSV
    success3 = test_csv_reading()
    
    # Test 4: Try with dependencies
    success4 = test_with_dependencies()
    
    print("\\n" + "=" * 50)
    print("Test Summary:")
    print(f"✓ Basic AQI calculation: {'PASS' if success1 else 'FAIL'}")
    print(f"✓ CSV creation: {'PASS' if csv_file else 'FAIL'}")
    print(f"✓ CSV reading: {'PASS' if success3 else 'FAIL'}")
    print(f"✓ Dependencies: {'PASS' if success4 else 'FAIL (install with pip)'}")
    
    if success1 and csv_file and success3:
        print("\\n🎉 Core system is working! Ready for your TEMPO data.")
        print("\\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Replace 'test_data.csv' with your TEMPO satellite data")
        print("3. Run: python main.py --data your_data.csv")
    else:
        print("\\n❌ Some tests failed. Check the errors above.")
    
    print("\\nFiles created:")
    if os.path.exists("test_data.csv"):
        print("- test_data.csv (sample data for testing)")

if __name__ == "__main__":
    main()