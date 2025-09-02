#!/usr/bin/env python3
"""
US Healthcare Insights Dashboard V1.0
Main project runner script
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'plotly', 'folium', 
        'numpy', 'tqdm', 'streamlit-folium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
    else:
        print("✅ All dependencies are installed")

def run_data_collection():
    """Run data collection process"""
    print("\n📊 Starting data collection...")
    
    try:
        result = subprocess.run([sys.executable, 'etl/data_collector.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data collection completed successfully")
            print(result.stdout)
        else:
            print("❌ Data collection failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        return False
    
    return True

def run_data_processing():
    """Run data processing (ETL)"""
    print("\n🔧 Starting data processing...")
    
    try:
        result = subprocess.run([sys.executable, 'etl/data_processor.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data processing completed successfully")
            print(result.stdout)
        else:
            print("❌ Data processing failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during data processing: {e}")
        return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🚀 Launching dashboard...")
    print("📍 Dashboard will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard/app.py'])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main project runner"""
    print("🏥 US Healthcare Insights Dashboard V1.0")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('etl').exists() or not Path('dashboard').exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    check_dependencies()
    
    # Ask user what to do
    print("\n🎯 What would you like to do?")
    print("1. Run complete pipeline (collect + process + dashboard)")
    print("2. Collect data only")
    print("3. Process data only")
    print("4. Launch dashboard only")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Complete pipeline
        if run_data_collection() and run_data_processing():
            launch_dashboard()
        else:
            print("❌ Pipeline failed. Please check the errors above.")
    
    elif choice == '2':
        # Data collection only
        run_data_collection()
    
    elif choice == '3':
        # Data processing only
        run_data_processing()
    
    elif choice == '4':
        # Dashboard only
        launch_dashboard()
    
    elif choice == '5':
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
