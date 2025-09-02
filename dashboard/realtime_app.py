#!/usr/bin/env python3
"""
Real-time Healthcare Dashboard V2.1
Live updates with WebSocket support and real-time analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import Dict, List, Optional
import websockets
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeHealthcareDashboard:
    """Real-time healthcare dashboard with live data updates"""
    
    def __init__(self):
        st.set_page_config(
            page_title="US Healthcare Insights - Real-time",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.data_dir = Path('data/realtime')
        self.data_dir.mkdir(exist_ok=True)
        
        # Real-time data cache
        self.cache = {}
        self.last_update = {}
        
        # WebSocket connection status
        self.ws_connected = False
        self.data_queue = queue.Queue()
        
        # Initialize session state
        if 'realtime_data' not in st.session_state:
            st.session_state.realtime_data = {}
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
    
    def load_realtime_data(self) -> Dict[str, pd.DataFrame]:
        """Load latest real-time data"""
        data = {}
        
        try:
            # Find latest data files
            for data_type in ['covid_data', 'weather_data', 'health_news', 'hospital_status', 'emergency_alerts']:
                files = list(self.data_dir.glob(f"{data_type}_*.csv"))
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    df = pd.read_csv(latest_file)
                    data[data_type] = df
                    self.last_update[data_type] = datetime.fromtimestamp(latest_file.stat().st_mtime)
            
            st.session_state.realtime_data = data
            st.session_state.last_refresh = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading real-time data: {e}")
        
        return data
    
    def render_header(self):
        """Render dashboard header with real-time status"""
        st.title("🏥 US Healthcare Insights Dashboard")
        st.markdown("### Real-time Analytics & Live Monitoring")
        
        # Real-time status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔄 Last Update",
                value=st.session_state.last_refresh.strftime("%H:%M:%S"),
                delta="Live"
            )
        
        with col2:
            data_count = len(st.session_state.realtime_data)
            st.metric(
                label="📊 Data Sources",
                value=data_count,
                delta="Active"
            )
        
        with col3:
            total_records = sum(len(df) for df in st.session_state.realtime_data.values())
            st.metric(
                label="📈 Total Records",
                value=f"{total_records:,}",
                delta="Real-time"
            )
        
        with col4:
            # Connection status
            status = "🟢 Connected" if self.ws_connected else "🔴 Offline"
            st.metric(
                label="🌐 Connection",
                value=status,
                delta="WebSocket"
            )
    
    def render_covid_tracker(self, covid_data: pd.DataFrame):
        """Render real-time COVID-19 tracker"""
        if covid_data.empty:
            return
        
        st.header("🦠 Real-time COVID-19 Tracker")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # COVID cases by state
            fig_cases = px.bar(
                covid_data.head(10),
                x='state',
                y='positive',
                title="Active COVID Cases by State",
                color='positive_rate',
                color_continuous_scale='Reds'
            )
            fig_cases.update_layout(height=400)
            st.plotly_chart(fig_cases, use_container_width=True)
        
        with col2:
            # Hospitalization rates
            fig_hosp = px.scatter(
                covid_data,
                x='positive_rate',
                y='hospitalization_rate',
                size='positive',
                color='state',
                title="Hospitalization vs Positive Rate",
                hover_data=['state', 'positive', 'hospitalizedCurrently']
            )
            fig_hosp.update_layout(height=400)
            st.plotly_chart(fig_hosp, use_container_width=True)
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_positive = covid_data['positive'].sum()
            st.metric(
                label="Total Positive Cases",
                value=f"{total_positive:,}",
                delta=f"{covid_data['positive'].iloc[-1] - covid_data['positive'].iloc[-2] if len(covid_data) > 1 else 0:,}"
            )
        
        with col2:
            avg_rate = covid_data['positive_rate'].mean()
            st.metric(
                label="Average Positive Rate",
                value=f"{avg_rate:.1%}",
                delta=f"{avg_rate - covid_data['positive_rate'].iloc[-2] if len(covid_data) > 1 else 0:.1%}"
            )
        
        with col3:
            total_hospitalized = covid_data['hospitalizedCurrently'].sum()
            st.metric(
                label="Currently Hospitalized",
                value=f"{total_hospitalized:,}",
                delta="Live"
            )
        
        with col4:
            avg_hosp_rate = covid_data['hospitalization_rate'].mean()
            st.metric(
                label="Avg Hospitalization Rate",
                value=f"{avg_hosp_rate:.1%}",
                delta="Real-time"
            )
    
    def render_weather_impact(self, weather_data: pd.DataFrame):
        """Render weather impact on healthcare"""
        if weather_data.empty:
            return
        
        st.header("🌤️ Weather Impact on Healthcare")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature distribution
            fig_temp = px.histogram(
                weather_data,
                x='temperature',
                title="Temperature Distribution",
                nbins=20,
                color_discrete_sequence=['lightblue']
            )
            fig_temp.update_layout(height=300)
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            # Weather conditions
            weather_counts = weather_data['weather_condition'].value_counts()
            fig_weather = px.pie(
                values=weather_counts.values,
                names=weather_counts.index,
                title="Weather Conditions"
            )
            fig_weather.update_layout(height=300)
            st.plotly_chart(fig_weather, use_container_width=True)
        
        # Weather metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_temp = weather_data['temperature'].mean()
            st.metric(
                label="Average Temperature",
                value=f"{avg_temp:.1f}°C",
                delta="Live"
            )
        
        with col2:
            avg_humidity = weather_data['humidity'].mean()
            st.metric(
                label="Average Humidity",
                value=f"{avg_humidity:.1f}%",
                delta="Real-time"
            )
        
        with col3:
            most_common_weather = weather_data['weather_condition'].mode().iloc[0]
            st.metric(
                label="Most Common Weather",
                value=most_common_weather,
                delta="Current"
            )
    
    def render_health_news(self, news_data: pd.DataFrame):
        """Render real-time health news and sentiment"""
        if news_data.empty:
            return
        
        st.header("📰 Real-time Health News & Sentiment")
        
        # Sentiment analysis
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = news_data['sentiment'].value_counts()
            fig_sentiment = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="News Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
            )
            fig_sentiment.update_layout(height=300)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # News timeline
            news_data['published_at'] = pd.to_datetime(news_data['published_at'])
            fig_timeline = px.scatter(
                news_data.head(10),
                x='published_at',
                y='sentiment',
                size=[1] * len(news_data.head(10)),
                color='sentiment',
                title="News Timeline by Sentiment",
                hover_data=['title', 'source']
            )
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Latest news
        st.subheader("📋 Latest Health News")
        
        for idx, row in news_data.head(5).iterrows():
            with st.expander(f"{row['title']} - {row['source']}"):
                st.write(f"**Published:** {row['published_at']}")
                st.write(f"**Sentiment:** {row['sentiment']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**URL:** {row['url']}")
    
    def render_hospital_status(self, hospital_data: pd.DataFrame):
        """Render real-time hospital status"""
        if hospital_data.empty:
            return
        
        st.header("🏥 Real-time Hospital Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ICU occupancy by state
            fig_icu = px.bar(
                hospital_data,
                x='state',
                y='icu_occupancy',
                title="ICU Occupancy by State",
                color='icu_occupancy',
                color_continuous_scale='RdYlGn_r'
            )
            fig_icu.update_layout(height=400)
            st.plotly_chart(fig_icu, use_container_width=True)
        
        with col2:
            # Emergency wait times
            fig_wait = px.scatter(
                hospital_data,
                x='emergency_wait_time',
                y='staff_shortage',
                size='total_hospitals',
                color='state',
                title="Emergency Wait Time vs Staff Shortage",
                hover_data=['state', 'available_beds']
            )
            fig_wait.update_layout(height=400)
            st.plotly_chart(fig_wait, use_container_width=True)
        
        # Hospital metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_hospitals = hospital_data['total_hospitals'].sum()
            st.metric(
                label="Total Hospitals",
                value=f"{total_hospitals:,}",
                delta="Live"
            )
        
        with col2:
            avg_icu_occupancy = hospital_data['icu_occupancy'].mean()
            st.metric(
                label="Average ICU Occupancy",
                value=f"{avg_icu_occupancy:.1%}",
                delta="Real-time"
            )
        
        with col3:
            avg_wait_time = hospital_data['emergency_wait_time'].mean()
            st.metric(
                label="Avg Emergency Wait",
                value=f"{avg_wait_time:.1f} min",
                delta="Current"
            )
        
        with col4:
            avg_staff_shortage = hospital_data['staff_shortage'].mean()
            st.metric(
                label="Avg Staff Shortage",
                value=f"{avg_staff_shortage:.1%}",
                delta="Live"
            )
    
    def render_emergency_alerts(self, alerts_data: pd.DataFrame):
        """Render real-time emergency alerts"""
        if alerts_data.empty:
            return
        
        st.header("🚨 Real-time Emergency Alerts")
        
        # Alert severity distribution
        col1, col2 = st.columns(2)
        
        with col1:
            severity_counts = alerts_data['severity'].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Alert Severity Distribution",
                color_discrete_sequence=['red', 'orange', 'yellow', 'green']
            )
            fig_severity.update_layout(height=300)
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            # Alert types
            type_counts = alerts_data['type'].value_counts()
            fig_types = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title="Emergency Alert Types",
                color=type_counts.values,
                color_continuous_scale='Reds'
            )
            fig_types.update_layout(height=300)
            st.plotly_chart(fig_types, use_container_width=True)
        
        # Active alerts
        st.subheader("🚨 Active Emergency Alerts")
        
        for idx, row in alerts_data.iterrows():
            severity_color = {
                'Critical': '🔴',
                'High': '🟠',
                'Medium': '🟡',
                'Low': '🟢'
            }
            
            with st.expander(f"{severity_color.get(row['severity'], '⚪')} {row['type']} - {row['state']}"):
                st.write(f"**Alert ID:** {row['alert_id']}")
                st.write(f"**Severity:** {row['severity']}")
                st.write(f"**State:** {row['state']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Issued:** {row['issued_at']}")
                st.write(f"**Expires:** {row['expires_at']}")
                st.write(f"**Affected Hospitals:** {row['affected_hospitals']}")
    
    def render_realtime_analytics(self):
        """Render real-time analytics dashboard"""
        st.header("📊 Real-time Analytics Dashboard")
        
        # Auto-refresh
        if st.button("🔄 Refresh Data", type="primary"):
            self.load_realtime_data()
            st.rerun()
        
        # Auto-refresh every 30 seconds
        if st.checkbox("🔄 Enable Auto-refresh (30s)"):
            time.sleep(30)
            self.load_realtime_data()
            st.rerun()
        
        # Load data
        data = self.load_realtime_data()
        
        if not data:
            st.warning("⚠️ No real-time data available. Run the real-time collector first.")
            return
        
        # Render sections
        self.render_covid_tracker(data.get('covid_data', pd.DataFrame()))
        self.render_weather_impact(data.get('weather_data', pd.DataFrame()))
        self.render_health_news(data.get('health_news', pd.DataFrame()))
        self.render_hospital_status(data.get('hospital_status', pd.DataFrame()))
        self.render_emergency_alerts(data.get('emergency_alerts', pd.DataFrame()))
    
    def run(self):
        """Run the real-time dashboard"""
        self.render_header()
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Dashboard Controls")
            
            # Data source selection
            st.subheader("📊 Data Sources")
            data_sources = st.multiselect(
                "Select data sources to display:",
                ["COVID-19", "Weather", "Health News", "Hospital Status", "Emergency Alerts"],
                default=["COVID-19", "Weather", "Health News", "Hospital Status", "Emergency Alerts"]
            )
            
            # Refresh controls
            st.subheader("🔄 Refresh Controls")
            if st.button("🔄 Manual Refresh"):
                self.load_realtime_data()
                st.rerun()
            
            auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
            refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)
            
            # Connection status
            st.subheader("🌐 Connection Status")
            st.info(f"WebSocket: {'🟢 Connected' if self.ws_connected else '🔴 Disconnected'}")
            st.info(f"Last Update: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        # Main content
        self.render_realtime_analytics()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray;'>
                <p>US Healthcare Insights Dashboard V2.1 - Real-time Analytics</p>
                <p>Powered by Streamlit, Plotly, and Real-time APIs</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Main function"""
    dashboard = RealTimeHealthcareDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
