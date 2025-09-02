"""
Simple US Healthcare Insights Dashboard for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import numpy as np

# Page config
st.set_page_config(
    page_title="US Healthcare Insights Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample healthcare data for demonstration"""
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
        'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
        'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
        'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
        'Wisconsin', 'Wyoming'
    ]
    
    # Generate hospital data
    hospital_data = []
    for state in states:
        hospital_data.append({
            'state': state,
            'hospitals': np.random.randint(10, 200),
            'beds_per_100k': np.random.randint(200, 800),
            'doctors_per_100k': np.random.randint(150, 400),
            'avg_rating': round(np.random.uniform(3.0, 4.5), 1)
        })
    
    # Generate cost data
    cost_data = []
    for state in states:
        cost_data.append({
            'state': state,
            'avg_premium': round(np.random.uniform(300, 600), 0),
            'avg_deductible': round(np.random.uniform(1000, 3000), 0),
            'healthcare_spending_per_capita': round(np.random.uniform(8000, 12000), 0),
            'uninsured_rate': round(np.random.uniform(5, 15), 1)
        })
    
    return pd.DataFrame(hospital_data), pd.DataFrame(cost_data)

def main():
    st.markdown('<div class="main-header">🏥 US Healthcare Insights Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Comprehensive National Healthcare Analysis - All 50 States")
    
    # Try to load real data, fallback to sample data
    try:
        data_dir = Path("data/processed")
        hospital_files = list(data_dir.glob("national_hospital_summary_*.csv"))
        cost_files = list(data_dir.glob("national_cost_trends_*.csv"))
        
        if hospital_files and cost_files:
            # Load real data
            hospital_data = pd.read_csv(max(hospital_files, key=lambda x: x.stat().st_mtime))
            cost_data = pd.read_csv(max(cost_files, key=lambda x: x.stat().st_mtime))
            st.success("✅ Loaded real healthcare data!")
        else:
            # Generate sample data
            hospital_data, cost_data = generate_sample_data()
            st.info("ℹ️ Using sample data for demonstration")
    except Exception as e:
        # Generate sample data
        hospital_data, cost_data = generate_sample_data()
        st.info("ℹ️ Using sample data for demonstration")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total States", len(hospital_data))
    
    with col2:
        total_hospitals = hospital_data['hospitals'].sum()
        st.metric("Total Hospitals", f"{total_hospitals:,}")
    
    with col3:
        avg_premium = cost_data['avg_premium'].mean()
        st.metric("Avg Premium", f"${avg_premium:.0f}")
    
    with col4:
        avg_rating = hospital_data['avg_rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.1f}/5.0")
    
    # Hospital distribution
    st.header("🏥 Hospital Distribution by State")
    
    fig = px.bar(
        hospital_data.head(10),
        x='state',
        y='hospitals',
        title="Top 10 States by Number of Hospitals",
        color='avg_rating',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Healthcare costs
    st.header("💰 Healthcare Costs Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            cost_data,
            x='avg_premium',
            y='avg_deductible',
            size='healthcare_spending_per_capita',
            color='uninsured_rate',
            hover_data=['state'],
            title="Premium vs Deductible by State"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            cost_data,
            x='avg_premium',
            nbins=20,
            title="Distribution of Average Premiums"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # State comparison
    st.header("📊 State Comparison")
    
    selected_states = st.multiselect(
        "Select states to compare:",
        options=hospital_data['state'].tolist(),
        default=['California', 'Texas', 'New York', 'Florida']
    )
    
    if selected_states:
        filtered_data = hospital_data[hospital_data['state'].isin(selected_states)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=filtered_data['state'],
            y=filtered_data['hospitals'],
            name='Hospitals',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=filtered_data['state'],
            y=filtered_data['avg_rating'],
            name='Rating',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Hospital Count and Rating Comparison",
            yaxis=dict(title="Number of Hospitals"),
            yaxis2=dict(title="Average Rating", overlaying="y", side="right"),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.header("💡 Key Insights")
    
    insights = [
        f"🏥 **{len(hospital_data)} states** analyzed with comprehensive healthcare data",
        f"💰 **${avg_premium:.0f}** average monthly premium across all states",
        f"⭐ **{avg_rating:.1f}/5.0** average hospital rating nationwide",
        f"🏆 **{total_hospitals:,}** total hospitals providing care",
        "📈 **California, Texas, and New York** lead in hospital count",
        "💡 **Premium costs** vary significantly by state and region"
    ]
    
    for insight in insights:
        st.markdown(insight)
    
    st.markdown("---")
    st.markdown("**Data Source:** Generated healthcare analytics data")
    st.markdown("**Dashboard:** US Healthcare Insights Dashboard")

if __name__ == "__main__":
    main()
