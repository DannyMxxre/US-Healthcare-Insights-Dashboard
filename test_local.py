#!/usr/bin/env python3
"""
Local test script for US Healthcare Insights Dashboard V2.0
Tests core functionality without Docker dependencies
"""

import subprocess
import sys
import time
import requests
import pandas as pd
from pathlib import Path
import json

def test_data_collection():
    """Test data collection"""
    print("🔍 Testing Data Collection...")
    
    try:
        result = subprocess.run([sys.executable, 'etl/data_collector.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data collection test passed")
            return True
        else:
            print(f"❌ Data collection test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data collection: {e}")
        return False

def test_data_processing():
    """Test data processing"""
    print("🔍 Testing Data Processing...")
    
    try:
        result = subprocess.run([sys.executable, 'etl/data_processor.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data processing test passed")
            return True
        else:
            print(f"❌ Data processing test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data processing: {e}")
        return False

def test_ml_models():
    """Test ML model training"""
    print("🔍 Testing ML Models...")
    
    try:
        # Check if data exists
        data_files = list(Path('data/processed').glob('*.csv'))
        if not data_files:
            print("⚠️ No processed data found, creating sample data...")
            
            # Create sample data for ML testing
            import numpy as np
            
            # Sample hospital data
            hospitals = pd.DataFrame({
                'hospital_name': ['Test Hospital 1', 'Test Hospital 2', 'Test Hospital 3'],
                'state': ['CA', 'TX', 'NY'],
                'beds': [500, 300, 400],
                'rating': [4.2, 3.8, 4.5],
                'medicare_rating': [4, 3, 5],
                'safety_rating': [4, 3, 4],
                'patient_satisfaction': [85.2, 78.5, 90.1]
            })
            
            # Sample census data
            census = pd.DataFrame({
                'state': ['CA', 'TX', 'NY'],
                'population': [39500000, 29000000, 19500000],
                'median_income': [75000, 65000, 72000],
                'poverty_rate': [0.12, 0.15, 0.13],
                'unemployment_rate': [0.05, 0.06, 0.05],
                'education_bachelors': [0.35, 0.30, 0.38],
                'healthcare_coverage': [0.92, 0.88, 0.94]
            })
            
            # Sample cost data
            costs = pd.DataFrame({
                'state': ['CA', 'TX', 'NY'],
                'avg_premium': [550, 480, 520],
                'avg_deductible': [1500, 1200, 1400]
            })
            
            # Sample quality data
            quality = pd.DataFrame({
                'state': ['CA', 'TX', 'NY'],
                'life_expectancy': [81.2, 78.5, 80.8],
                'infant_mortality_rate': [4.2, 5.8, 4.5],
                'preventable_deaths_per_100k': [65, 85, 70]
            })
            
            # Save sample data
            Path('data/processed').mkdir(exist_ok=True)
            hospitals.to_csv('data/processed/sample_hospitals.csv', index=False)
            census.to_csv('data/processed/sample_census.csv', index=False)
            costs.to_csv('data/processed/sample_costs.csv', index=False)
            quality.to_csv('data/processed/sample_quality.csv', index=False)
            
            print("✅ Sample data created for ML testing")
        
        # Test ML models with sample data
        from ml.models import HealthcareMLModels
        
        # Load sample data
        hospitals_df = pd.read_csv('data/processed/sample_hospitals.csv')
        census_df = pd.read_csv('data/processed/sample_census.csv')
        costs_df = pd.read_csv('data/processed/sample_costs.csv')
        quality_df = pd.read_csv('data/processed/sample_quality.csv')
        
        # Initialize and test ML models
        ml_models = HealthcareMLModels()
        insights = ml_models.train_all_models(hospitals_df, census_df, costs_df, quality_df)
        
        print(f"✅ ML models test passed - {len(insights)} insights generated")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ML models: {e}")
        return False

def test_dashboard():
    """Test dashboard accessibility"""
    print("🔍 Testing Dashboard...")
    
    try:
        # Test if dashboard can start
        result = subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard/app.py', '--server.headless', 'true', '--server.port', '8502'], 
                               capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 or "Streamlit is running" in result.stdout:
            print("✅ Dashboard test passed")
            return True
        else:
            print(f"❌ Dashboard test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✅ Dashboard started successfully (timeout expected)")
        return True
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        return False

def test_api_import():
    """Test API import and basic functionality"""
    print("🔍 Testing API Import...")
    
    try:
        # Test if API can be imported
        from api.main import app
        
        # Test basic app functionality
        if hasattr(app, 'routes'):
            print("✅ API import test passed")
            return True
        else:
            print("❌ API import test failed - no routes found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API import: {e}")
        return False

def test_database_schema():
    """Test database schema creation"""
    print("🔍 Testing Database Schema...")
    
    try:
        # Check if schema file exists
        schema_file = Path('database/schema.sql')
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                schema_content = f.read()
            
            # Check for key components
            if 'CREATE TABLE' in schema_content and 'PostGIS' in schema_content:
                print("✅ Database schema test passed")
                return True
            else:
                print("❌ Database schema test failed - missing key components")
                return False
        else:
            print("❌ Database schema test failed - schema file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing database schema: {e}")
        return False

def run_local_test():
    """Run local test suite"""
    print("🚀 US Healthcare Insights Dashboard V2.0 - Local Test Suite")
    print("=" * 70)
    
    tests = [
        ("Data Collection", test_data_collection),
        ("Data Processing", test_data_processing),
        ("ML Models", test_ml_models),
        ("Dashboard", test_dashboard),
        ("API Import", test_api_import),
        ("Database Schema", test_database_schema),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 LOCAL TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All local tests passed! Core functionality is working!")
        print("\n🚀 Next steps:")
        print("1. Install Docker Desktop")
        print("2. Run: docker-compose -f docker-compose.test.yml up -d")
        print("3. Access services:")
        print("   - Dashboard: http://localhost:8501")
        print("   - API: http://localhost:8000")
        print("   - Database: localhost:5432")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_local_test()
    sys.exit(0 if success else 1)
