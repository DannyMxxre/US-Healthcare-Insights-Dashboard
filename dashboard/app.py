"""
Ultimate US Healthcare Insights Dashboard
Comprehensive national healthcare analysis for all 50 states
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
import folium
from streamlit_folium import folium_static

# Page config
st.set_page_config(
    page_title="Ultimate US Healthcare Insights Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class UltimateHealthcareDashboard:
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # State name mapping
        self.state_names = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
        }
        
    def load_national_data(self):
        """Load national processed data files"""
        try:
            # Find latest national files
            hospital_files = list(self.processed_dir.glob("national_hospital_summary_*.csv"))
            cost_files = list(self.processed_dir.glob("national_cost_trends_*.csv"))
            insights_files = list(self.processed_dir.glob("national_insights_*.json"))
            map_files = list(self.processed_dir.glob("national_healthcare_map_*.html"))
            
            if not hospital_files or not cost_files:
                st.error("❌ No national data found. Please run the national ETL pipeline first.")
                return None
            
            # Load latest files
            self.hospital_data = pd.read_csv(max(hospital_files, key=lambda x: x.stat().st_mtime))
            self.cost_data = pd.read_csv(max(cost_files, key=lambda x: x.stat().st_mtime))
            
            if insights_files:
                with open(max(insights_files, key=lambda x: x.stat().st_mtime), 'r') as f:
                    self.insights = json.load(f)
            else:
                self.insights = {}
            
            if map_files:
                self.map_file = max(map_files, key=lambda x: x.stat().st_mtime)
            else:
                self.map_file = None
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading national data: {e}")
            return None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🏥 Ultimate US Healthcare Insights Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Comprehensive National Healthcare Analysis - All 50 States")
        st.markdown("---")
    
    def render_national_metrics(self):
        """Render national key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_hospitals = self.hospital_data['total_hospitals'].sum()
            st.metric("🏥 Total Hospitals", f"{total_hospitals:,}")
        
        with col2:
            total_beds = self.hospital_data['total_beds'].sum()
            st.metric("🛏️ Total Beds", f"{total_beds:,}")
        
        with col3:
            avg_rating = self.hospital_data['avg_rating'].mean()
            st.metric("⭐ Avg Hospital Rating", f"{avg_rating:.1f}")
        
        with col4:
            total_states = len(self.hospital_data)
            st.metric("🗺️ States Analyzed", total_states)
    
    def render_hospital_analysis(self):
        """Render national hospital analysis"""
        st.subheader("🏥 National Hospital Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top states by hospital count
            top_hospital_states = self.hospital_data.nlargest(10, 'total_hospitals').copy()
            top_hospital_states['state_full'] = top_hospital_states['state'].map(self.state_names)
            fig_hospitals = px.bar(
                top_hospital_states,
                x='state_full',
                y='total_hospitals',
                title="Top 10 States by Hospital Count",
                color='total_hospitals',
                color_continuous_scale='Blues'
            )
            fig_hospitals.update_layout(height=400)
            st.plotly_chart(fig_hospitals, use_container_width=True)
        
        with col2:
            # Hospital ratings by state
            top_rated_states = self.hospital_data.nlargest(10, 'avg_rating').copy()
            top_rated_states['state_full'] = top_rated_states['state'].map(self.state_names)
            fig_ratings = px.bar(
                top_rated_states,
                x='state_full',
                y='avg_rating',
                title="Top 10 States by Hospital Rating",
                color='avg_rating',
                color_continuous_scale='RdYlGn'
            )
            fig_ratings.update_layout(height=400)
            st.plotly_chart(fig_ratings, use_container_width=True)
    
    def render_cost_analysis(self):
        """Render national cost analysis"""
        st.subheader("💰 National Healthcare Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Premium costs by state
            top_cost_states = self.cost_data.nlargest(10, 'avg_premium').copy()
            top_cost_states['state_full'] = top_cost_states['state'].map(self.state_names)
            fig_premium = px.bar(
                top_cost_states,
                x='state_full',
                y='avg_premium',
                title="Top 10 States by Average Premium",
                color='avg_premium',
                color_continuous_scale='Reds'
            )
            fig_premium.update_layout(height=400)
            st.plotly_chart(fig_premium, use_container_width=True)
        
        with col2:
            # Medicare spending by state
            top_medicare_states = self.cost_data.nlargest(10, 'avg_medicare_spending').copy()
            top_medicare_states['state_full'] = top_medicare_states['state'].map(self.state_names)
            fig_medicare = px.bar(
                top_medicare_states,
                x='state_full',
                y='avg_medicare_spending',
                title="Top 10 States by Medicare Spending",
                color='avg_medicare_spending',
                color_continuous_scale='Greens'
            )
            fig_medicare.update_layout(height=400)
            st.plotly_chart(fig_medicare, use_container_width=True)
    
    def render_geospatial_analysis(self):
        """Render national geospatial analysis"""
        st.subheader("🗺️ National Geospatial Analysis")
        
        if self.map_file:
            # Display the national map
            with open(self.map_file, 'r') as f:
                map_html = f.read()
            
            st.components.v1.html(map_html, height=600)
            
            st.markdown("**📍 National Hospital Distribution:**")
            hospital_summary = self.hospital_data[['state', 'total_hospitals', 'total_beds', 'avg_rating']].copy()
            hospital_summary['state_full'] = hospital_summary['state'].map(self.state_names)
            hospital_summary = hospital_summary[['state_full', 'total_hospitals', 'total_beds', 'avg_rating']].rename(columns={'state_full': 'State'})
            st.dataframe(hospital_summary, use_container_width=True)
        else:
            st.warning("🗺️ National map file not found")
    
    def render_correlation_analysis(self):
        """Render national correlation analysis"""
        st.subheader("📈 National Correlation Analysis")
        
        if 'correlation_analysis' in self.insights:
            correlations = self.insights['correlation_analysis']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Income vs Life Expectancy",
                    f"{correlations['income_vs_life_expectancy']:.3f}",
                    help="Correlation between median income and life expectancy"
                )
            
            with col2:
                st.metric(
                    "Income vs Healthcare Coverage",
                    f"{correlations['income_vs_healthcare_coverage']:.3f}",
                    help="Correlation between income and healthcare coverage"
                )
            
            with col3:
                st.metric(
                    "Poverty vs Infant Mortality",
                    f"{correlations['poverty_vs_infant_mortality']:.3f}",
                    help="Correlation between poverty and infant mortality"
                )
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.metric(
                    "Education vs Healthcare Coverage",
                    f"{correlations['education_vs_healthcare_coverage']:.3f}",
                    help="Correlation between education and healthcare coverage"
                )
            
            with col5:
                st.metric(
                    "Unemployment vs Uninsured",
                    f"{correlations['unemployment_vs_uninsured']:.3f}",
                    help="Correlation between unemployment and uninsured rate"
                )
    
    def render_insights(self):
        """Render national insights"""
        st.subheader("🔍 National Key Insights")
        
        if self.insights:
            # Demographic insights
            if 'demographic_insights' in self.insights:
                with st.expander("👥 Demographic Insights"):
                    demo_insights = self.insights['demographic_insights']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Total Population:** {demo_insights['total_population']:,}")
                        st.write(f"**Average Median Income:** ${demo_insights['avg_median_income']:,.0f}")
                        st.write(f"**Average Poverty Rate:** {demo_insights['avg_poverty_rate']:.1%}")
                        st.write(f"**Richest State:** {demo_insights['richest_state']}")
                    
                    with col2:
                        st.write(f"**Average Unemployment Rate:** {demo_insights['avg_unemployment_rate']:.1%}")
                        st.write(f"**Average Healthcare Coverage:** {demo_insights['avg_healthcare_coverage']:.1%}")
                        st.write(f"**Poorest State:** {demo_insights['poorest_state']}")
                        st.write(f"**Highest Uninsured Rate:** {demo_insights['highest_uninsured_rate']}")
            
            # Quality insights
            if 'quality_insights' in self.insights:
                with st.expander("🏆 Healthcare Quality Insights"):
                    quality_insights = self.insights['quality_insights']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Average Life Expectancy:** {quality_insights['avg_life_expectancy']:.1f} years")
                        st.write(f"**Average Infant Mortality:** {quality_insights['avg_infant_mortality']:.1f} per 1,000")
                        st.write(f"**Best Life Expectancy:** {quality_insights['best_life_expectancy_state']}")
                        st.write(f"**Best Access to Care:** {quality_insights['best_access_to_care_state']}")
                    
                    with col2:
                        st.write(f"**Average Preventable Deaths:** {quality_insights['avg_preventable_deaths']:.0f} per 100k")
                        st.write(f"**Average Readmission Rate:** {quality_insights['avg_readmission_rate']:.1%}")
                        st.write(f"**Worst Life Expectancy:** {quality_insights['worst_life_expectancy_state']}")
                        st.write(f"**Worst Access to Care:** {quality_insights['worst_access_to_care_state']}")
    
    def render_state_comparison(self):
        """Render state comparison tool"""
        st.subheader("🔍 State Comparison Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_states = st.multiselect(
                "Select states to compare:",
                options=self.hospital_data['state'].tolist(),
                default=['CA', 'TX', 'NY', 'FL']
            )
        
        with col2:
            metric = st.selectbox(
                "Select metric:",
                options=['total_hospitals', 'total_beds', 'avg_rating', 'avg_medicare_rating']
            )
        
        if selected_states:
            comparison_data = self.hospital_data[self.hospital_data['state'].isin(selected_states)].copy()
            comparison_data['state_full'] = comparison_data['state'].map(self.state_names)
            
            fig_comparison = px.bar(
                comparison_data,
                x='state_full',
                y=metric,
                title=f"State Comparison: {metric.replace('_', ' ').title()}",
                color=metric,
                color_continuous_scale='Viridis'
            )
            fig_comparison.update_layout(height=400)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    def run(self):
        """Run the ultimate dashboard"""
        self.render_header()
        
        if not self.load_national_data():
            st.stop()
        
        # Render dashboard sections
        self.render_national_metrics()
        st.markdown("---")
        
        self.render_hospital_analysis()
        st.markdown("---")
        
        self.render_cost_analysis()
        st.markdown("---")
        
        self.render_geospatial_analysis()
        st.markdown("---")
        
        self.render_correlation_analysis()
        st.markdown("---")
        
        self.render_state_comparison()
        st.markdown("---")
        
        self.render_insights()

if __name__ == "__main__":
    dashboard = UltimateHealthcareDashboard()
    dashboard.run()
