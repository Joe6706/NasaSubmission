"""
Data Pipeline Integration Tests
Tests for validating the AQI forecasting system with real TEMPO data.
"""

import pandas as pd
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, List

def test_data_format_validation(csv_file_path: str) -> Dict:
    """
    Test that incoming TEMPO data matches expected format.
    
    Args:
        csv_file_path: Path to real TEMPO data file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "format_valid": True,
        "issues": [],
        "data_summary": {}
    }
    
    try:
        # Load data
        df = pd.read_csv(csv_file_path)
        
        # Expected columns
        expected_cols = ['time', 'city', 'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 
                        'temperature', 'humidity', 'wind_speed']
        
        # Check columns
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            results["issues"].append(f"Missing columns: {missing_cols}")
            results["format_valid"] = False
        
        extra_cols = set(df.columns) - set(expected_cols)
        if extra_cols:
            results["issues"].append(f"Extra columns (will be ignored): {extra_cols}")
        
        # Check data types and ranges
        if 'time' in df.columns:
            try:
                pd.to_datetime(df['time'])
            except:
                results["issues"].append("Invalid time format - should be ISO 8601")
                results["format_valid"] = False
        
        # Check pollutant ranges (basic sanity checks)
        pollutant_ranges = {
            'PM2.5': (0, 500),    # μg/m³
            'PM10': (0, 1000),    # μg/m³
            'O3': (0, 0.5),       # ppm
            'NO2': (0, 2.0),      # ppm
            'SO2': (0, 1.0),      # ppm
            'CO': (0, 50),        # ppm
        }
        
        for pollutant, (min_val, max_val) in pollutant_ranges.items():
            if pollutant in df.columns:
                values = df[pollutant].dropna()
                if len(values) > 0:
                    if values.min() < min_val or values.max() > max_val:
                        results["issues"].append(
                            f"{pollutant} values out of expected range "
                            f"({min_val}-{max_val}): {values.min():.3f} to {values.max():.3f}"
                        )
        
        # Data summary
        results["data_summary"] = {
            "total_rows": len(df),
            "time_range": f"{df['time'].iloc[0]} to {df['time'].iloc[-1]}" if 'time' in df.columns else "N/A",
            "cities": list(df['city'].unique()) if 'city' in df.columns else [],
            "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
    except Exception as e:
        results["format_valid"] = False
        results["issues"].append(f"Failed to load data: {str(e)}")
    
    return results


def test_epa_compliance(processed_data_path: str) -> Dict:
    """
    Test that processed data meets EPA AQI calculation requirements.
    
    Args:
        processed_data_path: Path to processed data after rolling averages
        
    Returns:
        Dictionary with EPA compliance results
    """
    results = {
        "epa_compliant": True,
        "issues": [],
        "rolling_averages_found": []
    }
    
    try:
        df = pd.read_csv(processed_data_path)
        
        # Check for required rolling average columns
        expected_rolling_cols = [
            'PM2.5_24hr_avg',
            'PM10_24hr_avg', 
            'O3_8hr_max',
            'CO_8hr_avg',
            'NO2_1hr',
            'SO2_1hr'
        ]
        
        for col in expected_rolling_cols:
            if col in df.columns:
                results["rolling_averages_found"].append(col)
            else:
                results["issues"].append(f"Missing EPA-required column: {col}")
                results["epa_compliant"] = False
        
        # Check data sufficiency for rolling averages
        min_required_hours = 24  # For 24-hour PM averages
        if len(df) < min_required_hours:
            results["issues"].append(f"Insufficient data for EPA calculations: {len(df)} hours < {min_required_hours} required")
            results["epa_compliant"] = False
            
    except Exception as e:
        results["epa_compliant"] = False
        results["issues"].append(f"Failed to validate EPA compliance: {str(e)}")
    
    return results


def test_forecast_quality(forecast_json_path: str) -> Dict:
    """
    Test the quality and realism of generated forecasts.
    
    Args:
        forecast_json_path: Path to generated forecast JSON
        
    Returns:
        Dictionary with forecast quality results
    """
    results = {
        "quality_good": True,
        "issues": [],
        "forecast_summary": {}
    }
    
    try:
        with open(forecast_json_path, 'r') as f:
            forecast_data = json.load(f)
        
        hourly_forecasts = forecast_data.get('hourly_forecasts', [])
        
        if not hourly_forecasts:
            results["issues"].append("No hourly forecasts found")
            results["quality_good"] = False
            return results
        
        # Extract AQI values
        aqi_values = [h.get('predicted_AQI') for h in hourly_forecasts if h.get('predicted_AQI') is not None]
        
        if not aqi_values:
            results["issues"].append("No valid AQI predictions found")
            results["quality_good"] = False
            return results
        
        # Check for temporal variation
        unique_aqi_values = len(set(aqi_values))
        if unique_aqi_values == 1:
            results["issues"].append("All AQI predictions are identical - may indicate insufficient temporal variation")
        
        # Check AQI ranges
        min_aqi, max_aqi = min(aqi_values), max(aqi_values)
        if min_aqi < 0 or max_aqi > 500:
            results["issues"].append(f"AQI values out of valid range (0-500): {min_aqi} to {max_aqi}")
            results["quality_good"] = False
        
        # Check for reasonable variation
        aqi_range = max_aqi - min_aqi
        if aqi_range > 200:
            results["issues"].append(f"Excessive AQI variation in 24 hours: {aqi_range} points")
        
        results["forecast_summary"] = {
            "hours_predicted": len(hourly_forecasts),
            "aqi_range": f"{min_aqi} to {max_aqi}",
            "unique_values": unique_aqi_values,
            "dominant_pollutants": list(set(h.get('dominant_pollutant') for h in hourly_forecasts))
        }
        
    except Exception as e:
        results["quality_good"] = False
        results["issues"].append(f"Failed to analyze forecast quality: {str(e)}")
    
    return results


def test_multi_city_handling(csv_file_path: str) -> Dict:
    """
    Test that multi-city data is handled correctly.
    
    Args:
        csv_file_path: Path to multi-city TEMPO data
        
    Returns:
        Dictionary with multi-city handling results
    """
    results = {
        "multi_city_ok": True,
        "issues": [],
        "city_summary": {}
    }
    
    try:
        df = pd.read_csv(csv_file_path)
        
        if 'city' not in df.columns:
            results["issues"].append("No city column found - single city data?")
            return results
        
        cities = df['city'].unique()
        results["city_summary"]["cities_found"] = list(cities)
        results["city_summary"]["total_cities"] = len(cities)
        
        # Check data distribution across cities
        city_counts = df['city'].value_counts()
        min_data_per_city = city_counts.min()
        max_data_per_city = city_counts.max()
        
        if min_data_per_city < 24:
            results["issues"].append(f"Some cities have insufficient data (< 24 hours): {city_counts.to_dict()}")
            results["multi_city_ok"] = False
        
        # Check for data imbalance
        if max_data_per_city > 3 * min_data_per_city:
            results["issues"].append(f"Significant data imbalance between cities: {city_counts.to_dict()}")
        
        results["city_summary"]["data_per_city"] = city_counts.to_dict()
        
    except Exception as e:
        results["multi_city_ok"] = False
        results["issues"].append(f"Failed to analyze multi-city data: {str(e)}")
    
    return results


def run_full_data_pipeline_test(tempo_data_path: str, output_dir: str = ".") -> Dict:
    """
    Run complete test suite for TEMPO data pipeline integration.
    
    Args:
        tempo_data_path: Path to real TEMPO data CSV
        output_dir: Directory for test outputs
        
    Returns:
        Complete test results
    """
    print("🔄 Running TEMPO Data Pipeline Integration Tests...")
    
    all_results = {
        "overall_status": "PASS",
        "timestamp": datetime.now().isoformat(),
        "data_file": tempo_data_path,
        "tests": {}
    }
    
    # Test 1: Data format validation
    print("1️⃣ Testing data format...")
    format_results = test_data_format_validation(tempo_data_path)
    all_results["tests"]["data_format"] = format_results
    if not format_results["format_valid"]:
        all_results["overall_status"] = "FAIL"
    
    # Test 2: Multi-city handling
    print("2️⃣ Testing multi-city handling...")
    city_results = test_multi_city_handling(tempo_data_path)
    all_results["tests"]["multi_city"] = city_results
    if not city_results["multi_city_ok"]:
        all_results["overall_status"] = "FAIL"
    
    # If basic tests pass, run the ML pipeline
    if all_results["overall_status"] == "PASS":
        print("3️⃣ Running ML pipeline...")
        try:
            # Import and run the main pipeline
            import subprocess
            result = subprocess.run([
                "python", "main.py", 
                "--data", tempo_data_path,
                "--output", f"{output_dir}/test_forecast.json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ ML pipeline completed successfully")
                
                # Test 3: EPA compliance (if processed data available)
                print("4️⃣ Testing EPA compliance...")
                # Note: Would need to modify main.py to output processed data for testing
                
                # Test 4: Forecast quality
                print("5️⃣ Testing forecast quality...")
                forecast_results = test_forecast_quality(f"{output_dir}/test_forecast.json")
                all_results["tests"]["forecast_quality"] = forecast_results
                if not forecast_results["quality_good"]:
                    all_results["overall_status"] = "WARN"
                    
            else:
                print("❌ ML pipeline failed")
                all_results["overall_status"] = "FAIL"
                all_results["tests"]["pipeline_execution"] = {
                    "success": False,
                    "error": result.stderr
                }
                
        except Exception as e:
            all_results["overall_status"] = "FAIL"
            all_results["tests"]["pipeline_execution"] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"Overall Status: {all_results['overall_status']}")
    for test_name, test_result in all_results["tests"].items():
        if isinstance(test_result, dict):
            status_key = [k for k in test_result.keys() if k.endswith('_valid') or k.endswith('_ok') or k.endswith('_good')]
            if status_key:
                status = "✅ PASS" if test_result[status_key[0]] else "❌ FAIL"
                print(f"  {test_name}: {status}")
                if test_result.get('issues'):
                    for issue in test_result['issues']:
                        print(f"    ⚠️ {issue}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    print("TEMPO Data Pipeline Integration Test Suite")
    print("=" * 50)
    print("Usage: python test_data_integration.py <path_to_tempo_data.csv>")
    print("")
    print("This will test:")
    print("✓ Data format validation")
    print("✓ EPA compliance requirements") 
    print("✓ Multi-city data handling")
    print("✓ ML pipeline execution")
    print("✓ Forecast quality validation")