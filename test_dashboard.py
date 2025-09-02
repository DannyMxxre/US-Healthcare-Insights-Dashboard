#!/usr/bin/env python3
"""
Test dashboard data loading
"""

from pathlib import Path
import pandas as pd
import json

def test_data_loading():
    """Test if dashboard can load data"""
    print("🔍 Testing dashboard data loading...")
    
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Processed directory: {processed_dir}")
    print(f"📁 Processed exists: {processed_dir.exists()}")
    
    if not processed_dir.exists():
        print("❌ Processed directory not found!")
        return False
    
    # Check for required files
    hospital_files = list(processed_dir.glob("national_hospital_summary_*.csv"))
    cost_files = list(processed_dir.glob("national_cost_trends_*.csv"))
    insights_files = list(processed_dir.glob("national_insights_*.json"))
    
    print(f"🏥 Hospital files found: {len(hospital_files)}")
    print(f"💰 Cost files found: {len(cost_files)}")
    print(f"📊 Insights files found: {len(insights_files)}")
    
    if not hospital_files:
        print("❌ No hospital files found!")
        return False
    
    if not cost_files:
        print("❌ No cost files found!")
        return False
    
    # Try to load latest files
    try:
        latest_hospital = max(hospital_files, key=lambda x: x.stat().st_mtime)
        latest_cost = max(cost_files, key=lambda x: x.stat().st_mtime)
        
        print(f"🏥 Loading hospital data: {latest_hospital}")
        hospital_data = pd.read_csv(latest_hospital)
        print(f"✅ Hospital data loaded: {hospital_data.shape}")
        
        print(f"💰 Loading cost data: {latest_cost}")
        cost_data = pd.read_csv(latest_cost)
        print(f"✅ Cost data loaded: {cost_data.shape}")
        
        if insights_files:
            latest_insights = max(insights_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Loading insights: {latest_insights}")
            with open(latest_insights, 'r') as f:
                insights = json.load(f)
            print(f"✅ Insights loaded: {len(insights)} keys")
        
        print("🎉 All data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

if __name__ == "__main__":
    test_data_loading()
