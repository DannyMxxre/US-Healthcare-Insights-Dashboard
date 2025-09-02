"""
National Data Processor for US Healthcare Insights Dashboard
Processes comprehensive healthcare data for all 50 states
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import folium
from folium import plugins

class NationalDataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
    
    def load_national_data(self):
        """Load national data files"""
        print("📊 Loading national data...")
        
        # Find latest national files
        hospital_files = list(self.raw_dir.glob("national_hospitals_*.csv"))
        census_files = list(self.raw_dir.glob("national_census_*.csv"))
        cost_files = list(self.raw_dir.glob("national_costs_*.csv"))
        quality_files = list(self.raw_dir.glob("national_quality_*.csv"))
        
        if not hospital_files or not census_files:
            print("❌ National data files not found")
            return None
        
        # Load latest files
        self.hospitals = pd.read_csv(max(hospital_files, key=lambda x: x.stat().st_mtime))
        self.census = pd.read_csv(max(census_files, key=lambda x: x.stat().st_mtime))
        self.costs = pd.read_csv(max(cost_files, key=lambda x: x.stat().st_mtime))
        self.quality = pd.read_csv(max(quality_files, key=lambda x: x.stat().st_mtime))
        
        print(f"✅ Loaded national data:")
        print(f"   🏥 Hospitals: {self.hospitals.shape}")
        print(f"   👥 Census: {self.census.shape}")
        print(f"   💰 Costs: {self.costs.shape}")
        print(f"   🏆 Quality: {self.quality.shape}")
        
        return True
    
    def process_hospital_data(self):
        """Process national hospital data"""
        print("🏥 Processing national hospital data...")
        
        # Add calculated fields
        self.hospitals['beds_per_100k'] = (self.hospitals['beds'] / 100000).round(2)
        self.hospitals['rating_category'] = pd.cut(
            self.hospitals['rating'], 
            bins=[0, 3.0, 3.5, 4.0, 4.5, 5.0], 
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        # Hospital summary by state
        hospital_summary = self.hospitals.groupby('state').agg({
            'hospital_name': 'count',
            'beds': 'sum',
            'rating': 'mean',
            'medicare_rating': 'mean',
            'safety_rating': 'mean',
            'patient_satisfaction': 'mean'
        }).round(2)
        
        hospital_summary.columns = ['total_hospitals', 'total_beds', 'avg_rating', 'avg_medicare_rating', 'avg_safety_rating', 'avg_patient_satisfaction']
        hospital_summary = hospital_summary.reset_index()
        
        return hospital_summary
    
    def process_census_data(self):
        """Process national census data"""
        print("👥 Processing national census data...")
        
        # Add calculated fields
        self.census['poverty_population'] = (self.census['population'] * self.census['poverty_rate']).round(0)
        self.census['unemployed_population'] = (self.census['population'] * self.census['unemployment_rate']).round(0)
        self.census['educated_population'] = (self.census['population'] * self.census['education_bachelors']).round(0)
        self.census['uninsured_population'] = (self.census['population'] * self.census['uninsured_rate']).round(0)
        
        # Create demographic insights
        demographic_insights = {
            'total_population': self.census['population'].sum(),
            'avg_median_income': self.census['median_income'].mean(),
            'avg_poverty_rate': self.census['poverty_rate'].mean(),
            'avg_unemployment_rate': self.census['unemployment_rate'].mean(),
            'avg_healthcare_coverage': self.census['healthcare_coverage'].mean(),
            'richest_state': self.census.loc[self.census['median_income'].idxmax(), 'state'],
            'poorest_state': self.census.loc[self.census['median_income'].idxmin(), 'state'],
            'highest_uninsured_rate': self.census.loc[self.census['uninsured_rate'].idxmax(), 'state'],
            'lowest_uninsured_rate': self.census.loc[self.census['uninsured_rate'].idxmin(), 'state']
        }
        
        return demographic_insights
    
    def process_cost_data(self):
        """Process national healthcare cost data"""
        print("💰 Processing national healthcare cost data...")
        
        # Calculate year-over-year changes
        self.costs['premium_change'] = self.costs.groupby('state')['avg_premium'].pct_change() * 100
        self.costs['deductible_change'] = self.costs.groupby('state')['avg_deductible'].pct_change() * 100
        
        # Cost trends by state
        cost_trends = self.costs.groupby('state').agg({
            'avg_premium': ['mean', 'std'],
            'avg_deductible': ['mean', 'std'],
            'medicare_spending_per_capita': 'mean',
            'medicaid_enrollment_rate': 'mean'
        }).round(2)
        
        cost_trends.columns = ['avg_premium', 'premium_std', 'avg_deductible', 'deductible_std', 'avg_medicare_spending', 'avg_medicaid_enrollment']
        cost_trends = cost_trends.reset_index()
        
        return cost_trends
    
    def process_quality_data(self):
        """Process healthcare quality data"""
        print("🏆 Processing healthcare quality data...")
        
        # Quality insights
        quality_insights = {
            'avg_life_expectancy': self.quality['life_expectancy'].mean(),
            'avg_infant_mortality': self.quality['infant_mortality_rate'].mean(),
            'avg_preventable_deaths': self.quality['preventable_deaths_per_100k'].mean(),
            'avg_readmission_rate': self.quality['hospital_readmission_rate'].mean(),
            'best_life_expectancy_state': self.quality.loc[self.quality['life_expectancy'].idxmax(), 'state'],
            'worst_life_expectancy_state': self.quality.loc[self.quality['life_expectancy'].idxmin(), 'state'],
            'best_access_to_care_state': self.quality.loc[self.quality['access_to_care_score'].idxmax(), 'state'],
            'worst_access_to_care_state': self.quality.loc[self.quality['access_to_care_score'].idxmin(), 'state']
        }
        
        return quality_insights
    
    def create_national_correlation_analysis(self):
        """Create comprehensive correlation analysis"""
        print("📈 Creating national correlation analysis...")
        
        # Merge datasets for correlation analysis
        merged_data = self.census.merge(self.quality, on='state', how='inner')
        
        correlations = {
            'income_vs_life_expectancy': merged_data['median_income'].corr(merged_data['life_expectancy']),
            'income_vs_healthcare_coverage': merged_data['median_income'].corr(merged_data['healthcare_coverage']),
            'poverty_vs_infant_mortality': merged_data['poverty_rate'].corr(merged_data['infant_mortality_rate']),
            'education_vs_healthcare_coverage': merged_data['education_bachelors'].corr(merged_data['healthcare_coverage']),
            'unemployment_vs_uninsured': merged_data['unemployment_rate'].corr(merged_data['uninsured_rate'])
        }
        
        return correlations
    
    def create_national_map(self):
        """Create comprehensive national map"""
        print("🗺️ Creating national healthcare map...")
        
        # Create a map centered on the US
        us_map = folium.Map(
            location=[39.8283, -98.5795], 
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        
        # Add hospital markers (sample for performance)
        sample_hospitals = self.hospitals.sample(n=100)  # Show 100 random hospitals
        
        for idx, row in sample_hospitals.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"""
                <b>{row['hospital_name']}</b><br>
                State: {row['state']}<br>
                Beds: {row['beds']}<br>
                Rating: {row['rating']}<br>
                Type: {row['hospital_type']}
                """,
                icon=folium.Icon(color='red', icon='hospital-o')
            ).add_to(us_map)
        
        # Add state-level heatmap
        state_centers = self.census[['state']].copy()
        state_centers['latitude'] = np.random.uniform(25, 49, len(state_centers))
        state_centers['longitude'] = np.random.uniform(-125, -66, len(state_centers))
        
        # Add state markers with uninsured rate
        for idx, row in self.census.iterrows():
            folium.CircleMarker(
                location=[state_centers.iloc[idx]['latitude'], state_centers.iloc[idx]['longitude']],
                radius=row['uninsured_rate'] * 100,  # Scale for visibility
                popup=f"""
                <b>{row['state']}</b><br>
                Uninsured Rate: {row['uninsured_rate']:.1%}<br>
                Population: {row['population']:,}<br>
                Median Income: ${row['median_income']:,}
                """,
                color='blue',
                fill=True,
                fillOpacity=0.6
            ).add_to(us_map)
        
        return us_map
    
    def process_all_national_data(self):
        """Process all national datasets"""
        print("🚀 National Healthcare Data Processing")
        print("=" * 60)
        
        # Load data
        if not self.load_national_data():
            return None
        
        processed_data = {}
        insights = {}
        
        # Process each dataset
        processed_data['hospital_summary'] = self.process_hospital_data()
        insights['demographic_insights'] = self.process_census_data()
        processed_data['cost_trends'] = self.process_cost_data()
        insights['quality_insights'] = self.process_quality_data()
        insights['correlation_analysis'] = self.create_national_correlation_analysis()
        
        # Create national map
        processed_data['national_map'] = self.create_national_map()
        
        # Save processed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, data in processed_data.items():
            if name != 'national_map':  # Skip map object
                output_file = self.processed_dir / f"national_{name}_{timestamp}.csv"
                data.to_csv(output_file, index=False)
                print(f"✅ Saved {name}: {output_file}")
        
        # Save map as HTML
        map_file = self.processed_dir / f"national_healthcare_map_{timestamp}.html"
        processed_data['national_map'].save(str(map_file))
        print(f"✅ Saved national map: {map_file}")
        
        # Save insights
        insights_file = self.processed_dir / f"national_insights_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert_numpy(d)
        
        converted_insights = convert_dict(insights)
        
        with open(insights_file, 'w') as f:
            json.dump(converted_insights, f, indent=2)
        print(f"✅ Saved national insights: {insights_file}")
        
        # Print summary
        print("\n📊 NATIONAL PROCESSING SUMMARY:")
        print(f"🏥 Hospitals processed: {len(self.hospitals):,}")
        print(f"🗺️ States analyzed: {len(self.census)}")
        print(f"💰 Cost records: {len(self.costs):,}")
        print(f"🏆 Quality metrics: {len(self.quality)}")
        print(f"🗺️ National map created: {map_file}")
        
        return processed_data, insights

if __name__ == "__main__":
    processor = NationalDataProcessor()
    processed_data, insights = processor.process_all_national_data()
