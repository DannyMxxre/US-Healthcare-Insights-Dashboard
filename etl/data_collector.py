"""
National Data Collector for US Healthcare Insights Dashboard
Collects comprehensive healthcare data for all 50 states
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
from tqdm import tqdm
import urllib.request
import json
import numpy as np

class NationalDataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
    
    def create_national_hospital_data(self):
        """Create comprehensive national hospital data"""
        print("🏥 Creating National Hospital Data...")
        
        # Generate sample data for all states
        states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        ]
        
        hospitals = []
        for state in states:
            # Generate 10-50 hospitals per state
            num_hospitals = np.random.randint(10, 50)
            
            for i in range(num_hospitals):
                hospital = {
                    'hospital_name': f'Hospital {i+1} - {state}',
                    'city': f'City {i+1}',
                    'state': state,
                    'zip_code': f'{np.random.randint(10000, 99999)}',
                    'hospital_type': np.random.choice(['Acute Care', 'Specialty', 'Community', 'Teaching']),
                    'beds': np.random.randint(50, 1000),
                    'rating': round(np.random.uniform(2.5, 5.0), 1),
                    'latitude': np.random.uniform(25, 49),  # US latitude range
                    'longitude': np.random.uniform(-125, -66),  # US longitude range
                    'medicare_rating': np.random.randint(1, 6),
                    'safety_rating': np.random.randint(1, 6),
                    'patient_satisfaction': round(np.random.uniform(60, 95), 1)
                }
                hospitals.append(hospital)
        
        df = pd.DataFrame(hospitals)
        print(f"✅ Created {len(df)} hospitals across {len(states)} states")
        return df
    
    def create_national_census_data(self):
        """Create national demographic data"""
        print("👥 Creating National Census Data...")
        
        states = [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
            'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
            'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
            'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
            'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
        ]
        
        census_data = []
        for state in states:
            # Generate realistic demographic data
            population = np.random.randint(500000, 40000000)
            median_income = np.random.randint(45000, 85000)
            poverty_rate = round(np.random.uniform(0.08, 0.25), 3)
            unemployment_rate = round(np.random.uniform(0.03, 0.12), 3)
            education_bachelors = round(np.random.uniform(0.20, 0.45), 3)
            healthcare_coverage = round(np.random.uniform(0.85, 0.95), 3)
            
            census_record = {
                'state': state,
                'population': population,
                'median_income': median_income,
                'poverty_rate': poverty_rate,
                'unemployment_rate': unemployment_rate,
                'education_bachelors': education_bachelors,
                'healthcare_coverage': healthcare_coverage,
                'uninsured_rate': round(1 - healthcare_coverage, 3)
            }
            census_data.append(census_record)
        
        df = pd.DataFrame(census_data)
        print(f"✅ Created census data for {len(df)} states")
        return df
    
    def create_national_healthcare_costs(self):
        """Create national healthcare cost data"""
        print("💰 Creating National Healthcare Cost Data...")
        
        states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        ]
        
        years = [2020, 2021, 2022, 2023, 2024]
        cost_data = []
        
        for state in states:
            for year in years:
                # Generate realistic cost data with trends
                base_premium = np.random.randint(350, 650)
                base_deductible = np.random.randint(1000, 2000)
                
                # Add yearly inflation
                inflation_factor = 1 + (year - 2020) * 0.05
                
                cost_record = {
                    'state': state,
                    'year': year,
                    'avg_premium': round(base_premium * inflation_factor),
                    'avg_deductible': round(base_deductible * inflation_factor),
                    'out_of_pocket_max': round((base_premium + base_deductible) * inflation_factor),
                    'medicare_spending_per_capita': np.random.randint(8000, 15000),
                    'medicaid_enrollment_rate': round(np.random.uniform(0.15, 0.35), 3)
                }
                cost_data.append(cost_record)
        
        df = pd.DataFrame(cost_data)
        print(f"✅ Created cost data for {len(df)} state-year combinations")
        return df
    
    def create_healthcare_quality_metrics(self):
        """Create healthcare quality metrics by state"""
        print("🏆 Creating Healthcare Quality Metrics...")
        
        states = [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
            'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
            'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
            'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
            'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
        ]
        
        quality_data = []
        for state in states:
            quality_record = {
                'state': state,
                'life_expectancy': round(np.random.uniform(75, 82), 1),
                'infant_mortality_rate': round(np.random.uniform(4, 8), 1),
                'preventable_deaths_per_100k': np.random.randint(50, 150),
                'hospital_readmission_rate': round(np.random.uniform(0.15, 0.25), 3),
                'patient_safety_score': np.random.randint(1, 6),
                'access_to_care_score': round(np.random.uniform(60, 95), 1),
                'healthcare_affordability_score': round(np.random.uniform(40, 85), 1)
            }
            quality_data.append(quality_record)
        
        df = pd.DataFrame(quality_data)
        print(f"✅ Created quality metrics for {len(df)} states")
        return df
    
    def collect_national_data(self):
        """Collect all national datasets"""
        print("🚀 National Healthcare Data Collection")
        print("=" * 60)
        
        datasets = {}
        
        # Create comprehensive datasets
        datasets['hospitals'] = self.create_national_hospital_data()
        datasets['census'] = self.create_national_census_data()
        datasets['costs'] = self.create_national_healthcare_costs()
        datasets['quality'] = self.create_healthcare_quality_metrics()
        
        # Save datasets
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, data in datasets.items():
            filename = self.raw_dir / f"national_{name}_{timestamp}.csv"
            data.to_csv(filename, index=False)
            print(f"✅ Saved {name}: {filename}")
        
        # Summary
        total_hospitals = len(datasets['hospitals'])
        total_states = len(datasets['census'])
        total_cost_records = len(datasets['costs'])
        
        print("\n" + "=" * 60)
        print("📊 NATIONAL DATA COLLECTION SUMMARY")
        print("=" * 60)
        print(f"🏥 Total Hospitals: {total_hospitals:,}")
        print(f"🗺️ Total States: {total_states}")
        print(f"💰 Cost Records: {total_cost_records:,}")
        print(f"📈 Quality Metrics: {len(datasets['quality'])}")
        
        return datasets

if __name__ == "__main__":
    collector = NationalDataCollector()
    datasets = collector.collect_national_data()
