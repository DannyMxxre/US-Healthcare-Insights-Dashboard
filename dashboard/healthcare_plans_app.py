"""
Healthcare Plans Dashboard
Best in Class Healthcare Plans for Every State
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
import ast
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Healthcare Plans Dashboard",
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
    .plan-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .plan-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .plan-details {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        text-align: center;
        flex: 1;
        margin: 0 0.25rem;
    }
    .review-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        border-left: 3px solid #007bff;
    }
    .feature-tag {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .score-badge {
        background-color: #ffc107;
        color: #212529;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

class HealthcarePlansDashboard:
    def __init__(self):
        self.data_dir = Path("data/raw")
        self.load_data()
        
    def load_data(self):
        """Load healthcare plans data"""
        try:
            # Find latest files
            plans_files = list(self.data_dir.glob("healthcare_plans_*.csv"))
            best_plans_files = list(self.data_dir.glob("best_healthcare_plans_*.csv"))
            summary_files = list(self.data_dir.glob("healthcare_plans_summary_*.json"))
            
            if not plans_files or not best_plans_files:
                st.error("❌ No healthcare plans data found. Please run the healthcare plans collector first.")
                return None
            
            # Load latest files
            self.plans_df = pd.read_csv(max(plans_files, key=lambda x: x.stat().st_mtime))
            self.best_plans_df = pd.read_csv(max(best_plans_files, key=lambda x: x.stat().st_mtime))
            
            if summary_files:
                with open(max(summary_files, key=lambda x: x.stat().st_mtime), 'r') as f:
                    self.summary_stats = json.load(f)
            else:
                self.summary_stats = {}
            
            # Convert string representations back to lists
            self.best_plans_df['features'] = self.best_plans_df['features'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            self.best_plans_df['reviews'] = self.best_plans_df['reviews'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading healthcare plans data: {e}")
            return None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<div class="main-header">🏥 Best in Class Healthcare Plans</div>', unsafe_allow_html=True)
        st.markdown("### **Find the Best Healthcare Plans for Every State**")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Plans", f"{self.summary_stats.get('total_plans', len(self.plans_df)):,}")
        
        with col2:
            st.metric("States Covered", f"{self.summary_stats.get('states_covered', self.plans_df['state'].nunique())}")
        
        with col3:
            avg_premium = self.summary_stats.get('avg_premium', self.plans_df['monthly_premium'].mean())
            st.metric("Avg Monthly Premium", f"${avg_premium:.0f}")
        
        with col4:
            avg_rating = self.summary_stats.get('avg_rating', self.plans_df['overall_rating'].mean())
            st.metric("Avg Rating", f"{avg_rating:.1f}/5.0")
    
    def render_state_selector(self):
        """Render state selection"""
        st.sidebar.header("📍 Select State")
        
        states = sorted(self.best_plans_df['state'].unique())
        selected_state = st.sidebar.selectbox(
            "Choose a state:",
            states,
            index=states.index('California') if 'California' in states else 0
        )
        
        return selected_state
    
    def render_plan_filters(self):
        """Render plan filters"""
        st.sidebar.header("🔍 Filter Plans")
        
        # Plan type filter
        plan_types = ['All'] + sorted(self.plans_df['plan_type'].unique().tolist())
        selected_plan_type = st.sidebar.selectbox("Plan Type:", plan_types)
        
        # Price range filter
        min_price = int(self.plans_df['monthly_premium'].min())
        max_price = int(self.plans_df['monthly_premium'].max())
        price_range = st.sidebar.slider(
            "Monthly Premium Range ($):",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )
        
        # Rating filter
        min_rating = float(self.plans_df['overall_rating'].min())
        max_rating = float(self.plans_df['overall_rating'].max())
        rating_range = st.sidebar.slider(
            "Rating Range:",
            min_value=min_rating,
            max_value=max_rating,
            value=(min_rating, max_rating)
        )
        
        return selected_plan_type, price_range, rating_range
    
    def render_best_plan_for_state(self, state):
        """Render the best plan for selected state"""
        st.header(f"🏆 Best Healthcare Plan for {state}")
        
        state_best = self.best_plans_df[self.best_plans_df['state'] == state].iloc[0]
        
        # Plan header
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="plan-card">
                <div class="plan-header">{state_best['plan_name']}</div>
                <div style="color: #666; margin-bottom: 1rem;">
                    {state_best['insurance_company']} • {state_best['plan_type']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="score-badge" style="text-align: center; font-size: 2rem;">
                {state_best['overall_score']}
            </div>
            <div style="text-align: center; color: #666;">Overall Score</div>
            """, unsafe_allow_html=True)
        
        # Plan metrics
        st.markdown("### 📊 Plan Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Monthly Premium", f"${state_best['monthly_premium']:.0f}")
        
        with col2:
            st.metric("Annual Deductible", f"${state_best['annual_deductible']:,.0f}")
        
        with col3:
            st.metric("Max Out-of-Pocket", f"${state_best['max_out_of_pocket']:,.0f}")
        
        with col4:
            st.metric("Network Size", f"{state_best['network_size']:,} providers")
        
        # Ratings breakdown
        st.markdown("### ⭐ Ratings Breakdown")
        
        ratings_data = {
            'Overall Rating': state_best['overall_rating'],
            'Customer Satisfaction': state_best['customer_satisfaction']
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(ratings_data.keys()),
                y=list(ratings_data.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )
        ])
        
        fig.update_layout(
            title="Plan Ratings by Category",
            yaxis_title="Rating (1-5)",
            yaxis_range=[0, 5],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Features
        st.markdown("### 🎯 Plan Features")
        
        features = state_best['features']
        cols = st.columns(3)
        
        for i, feature in enumerate(features):
            col_idx = i % 3
            cols[col_idx].markdown(f'<div class="feature-tag">{feature}</div>', unsafe_allow_html=True)
        
        # Recommendation reason
        st.markdown("### 💡 Why This Plan?")
        st.info(f"**{state_best['recommendation_reason']}**")
        
        # Value proposition
        st.metric("Value Proposition Score", f"{state_best['value_proposition']}")
    
    def render_plan_reviews(self, state):
        """Render plan reviews"""
        st.markdown("### 💬 Customer Reviews")
        
        state_best = self.best_plans_df[self.best_plans_df['state'] == state].iloc[0]
        reviews = state_best['reviews']
        
        # Show top 5 reviews
        for i, review in enumerate(reviews[:5]):
            st.markdown(f"""
            <div class="review-box">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <strong>Rating: {review['rating']}/5.0</strong>
                    <span style="color: #666;">{review['date']}</span>
                </div>
                <div>{review['text']}</div>
                <div style="margin-top: 0.5rem; color: #666; font-size: 0.9rem;">
                    👍 {review['helpful_votes']} helpful • 
                    {'✅ Verified' if review['verified'] else '❌ Not verified'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_comparison_charts(self):
        """Render comparison charts"""
        st.markdown("### 📈 Plan Comparisons")
        
        # Premium vs Rating scatter plot
        fig = px.scatter(
            self.best_plans_df,
            x='monthly_premium',
            y='overall_rating',
            color='plan_type',
            size='overall_score',
            hover_data=['state', 'plan_name', 'insurance_company'],
            title="Premium vs Rating by Plan Type"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top plans by score
        st.markdown("### 🏆 Top 10 Plans by Overall Score")
        
        top_plans = self.best_plans_df.nlargest(10, 'overall_score')
        
        fig = px.bar(
            top_plans,
            x='overall_score',
            y='plan_name',
            orientation='h',
            color='state',
            title="Top 10 Healthcare Plans"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Plan type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            plan_type_counts = self.best_plans_df['plan_type'].value_counts()
            fig = px.pie(
                values=plan_type_counts.values,
                names=plan_type_counts.index,
                title="Best Plans by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            company_counts = self.best_plans_df['insurance_company'].value_counts().head(8)
            fig = px.bar(
                x=company_counts.values,
                y=company_counts.index,
                orientation='h',
                title="Top Insurance Companies"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_state_comparison(self):
        """Render state comparison"""
        st.markdown("### 🗺️ State-by-State Comparison")
        
        # Average metrics by state
        state_metrics = self.plans_df.groupby('state').agg({
            'monthly_premium': 'mean',
            'annual_deductible': 'mean',
            'overall_rating': 'mean',
            'overall_score': 'mean'
        }).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.choropleth(
                state_metrics.reset_index(),
                locations='state',
                locationmode='country names',
                color='monthly_premium',
                title="Average Monthly Premium by State",
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.choropleth(
                state_metrics.reset_index(),
                locations='state',
                locationmode='country names',
                color='overall_rating',
                title="Average Rating by State",
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the dashboard"""
        if not self.load_data():
            return
        
        self.render_header()
        
        # Sidebar
        selected_state = self.render_state_selector()
        selected_plan_type, price_range, rating_range = self.render_plan_filters()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Best Plan", "💬 Reviews", "📊 Comparisons", "🗺️ State Analysis"
        ])
        
        with tab1:
            self.render_best_plan_for_state(selected_state)
        
        with tab2:
            self.render_plan_reviews(selected_state)
        
        with tab3:
            self.render_comparison_charts()
        
        with tab4:
            self.render_state_comparison()

if __name__ == "__main__":
    dashboard = HealthcarePlansDashboard()
    dashboard.run()
