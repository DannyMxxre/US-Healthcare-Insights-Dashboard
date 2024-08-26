#!/usr/bin/env python3
"""
Test script for US Healthcare Insights Dashboard V2.0
Tests all components: ETL, ML, API, Database
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
            print("⚠️ No processed data found, skipping ML test")
            return True
        
        result = subprocess.run([sys.executable, 'ml/models.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ML models test passed")
            return True
        else:
            print(f"❌ ML models test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing ML models: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("🔍 Testing Database Connection...")
    
    try:
        import psycopg2
        
        # Try to connect to PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="healthcare_insights",
            user="healthcare_user",
            password="healthcare_password"
        )
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        print(f"✅ Database connection test passed: {version[0]}")
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        print("💡 Make sure PostgreSQL is running: docker-compose up postgres")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("🔍 Testing API Endpoints...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            print("✅ API health endpoint test passed")
            
            # Test metrics endpoint
            response = requests.get("http://localhost:8000/metrics", timeout=10)
            if response.status_code == 200:
                print("✅ API metrics endpoint test passed")
                return True
            else:
                print("❌ API metrics endpoint test failed")
                return False
        else:
            print(f"❌ API health endpoint test failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ API connection test failed - API not running")
        print("💡 Start API with: docker-compose up api")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_dashboard():
    """Test dashboard accessibility"""
    print("🔍 Testing Dashboard...")
    
    try:
        # Test dashboard health
        response = requests.get("http://localhost:8501/_stcore/health", timeout=10)
        
        if response.status_code == 200:
            print("✅ Dashboard test passed")
            return True
        else:
            print(f"❌ Dashboard test failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Dashboard connection test failed - Dashboard not running")
        print("💡 Start dashboard with: docker-compose up dashboard")
        return False
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        return False

def test_docker_services():
    """Test Docker services"""
    print("🔍 Testing Docker Services...")
    
    try:
        # Check if Docker is running
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Check for our services
            services = ['healthcare-postgres-test', 'healthcare-redis-test', 
                       'healthcare-dashboard-test', 'healthcare-api-test']
            
            running_services = []
            for service in services:
                result = subprocess.run(['docker', 'ps', '--filter', f'name={service}'], 
                                      capture_output=True, text=True)
                if service in result.stdout:
                    running_services.append(service)
            
            if running_services:
                print(f"✅ Docker services running: {', '.join(running_services)}")
                return True
            else:
                print("⚠️ No Docker services found")
                print("💡 Start services with: docker-compose -f docker-compose.test.yml up -d")
                return False
        else:
            print("❌ Docker not running")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Docker services: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 US Healthcare Insights Dashboard V2.0 - Comprehensive Test Suite")
    print("=" * 70)
    
    tests = [
        ("Docker Services", test_docker_services),
        ("Database Connection", test_database_connection),
        ("Data Collection", test_data_collection),
        ("Data Processing", test_data_processing),
        ("ML Models", test_ml_models),
        ("API Endpoints", test_api_endpoints),
        ("Dashboard", test_dashboard),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
