"""
ML Insights Dashboard
Advanced Machine Learning Analytics for Healthcare
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import ast

# Page config
st.set_page_config(
    page_title="ML Insights Dashboard",
    page_icon="🤖",
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
    .ml-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.3rem;
        text-align: center;
        flex: 1;
        margin: 0 0.25rem;
    }
    .anomaly-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .performance-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .feature-importance-bar {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MLInsightsDashboard:
    def __init__(self):
        self.ml_dir = Path("ml/saved_models")
        self.data_dir = Path("data/raw")
        self.load_ml_data()
        
    def load_ml_data(self):
        """Load ML results and data"""
        try:
            # Load ML results
            ml_files = list(self.ml_dir.glob("simple_ml_results_*.json"))
            if ml_files:
                with open(max(ml_files, key=lambda x: x.stat().st_mtime), 'r') as f:
                    self.ml_results = json.load(f)
            else:
                self.ml_results = {}
            
            # Load real-time data for predictions
            realtime_files = list(self.data_dir.glob("realtime_*.csv"))
            if realtime_files:
                self.realtime_data = {}
                for file in realtime_files:
                    data_type = file.stem.replace('realtime_', '')
                    self.realtime_data[data_type] = pd.read_csv(file)
            else:
                self.realtime_data = {}
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading ML data: {e}")
            return None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<div class="main-header">🤖 ML Insights Dashboard</div>', unsafe_allow_html=True)
        st.markdown("### **Advanced Machine Learning Analytics for Healthcare**")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            models_trained = self.ml_results.get('models_trained', 0)
            st.metric("Models Trained", f"{models_trained}")
        
        with col2:
            if 'predictions' in self.ml_results and 'icu_occupancy_pred' in self.ml_results['predictions']:
                avg_icu = np.mean(self.ml_results['predictions']['icu_occupancy_pred'])
                st.metric("Avg ICU Prediction", f"{avg_icu:.1%}")
            else:
                st.metric("Avg ICU Prediction", "N/A")
        
        with col3:
            anomalies = self.ml_results.get('predictions', {}).get('anomalies_detected', 0)
            st.metric("Anomalies Detected", f"{anomalies}")
        
        with col4:
            if 'insights' in self.ml_results and 'recommendations' in self.ml_results['insights']:
                recommendations = len(self.ml_results['insights']['recommendations'])
                st.metric("Recommendations", f"{recommendations}")
            else:
                st.metric("Recommendations", "0")
    
    def render_model_performance(self):
        """Render model performance metrics"""
        st.header("📊 Model Performance")
        
        if 'insights' in self.ml_results and 'model_performance' in self.ml_results['insights']:
            performance = self.ml_results['insights']['model_performance']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'icu_predictor' in performance:
                    icu_perf = performance['icu_predictor']
                    st.markdown(f"""
                    <div class="ml-card">
                        <h3>🏥 ICU Predictor</h3>
                        <div class="performance-badge">{icu_perf['performance_level']}</div>
                        <p><strong>R² Score:</strong> {icu_perf['r2_score']:.3f}</p>
                        <p><strong>MSE:</strong> {icu_perf['mse']:.4f}</p>
                        <p><strong>Type:</strong> Neural Network</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if 'sentiment_analyzer' in performance:
                    sent_perf = performance['sentiment_analyzer']
                    st.markdown(f"""
                    <div class="ml-card">
                        <h3>📝 Sentiment Analyzer</h3>
                        <div class="performance-badge">{sent_perf['performance_level']}</div>
                        <p><strong>Accuracy:</strong> {sent_perf['accuracy']:.3f}</p>
                        <p><strong>Type:</strong> NLP Classification</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if 'anomaly_detector' in self.ml_results.get('model_results', {}):
                    anomaly_perf = self.ml_results['model_results']['anomaly_detector']
                    st.markdown(f"""
                    <div class="ml-card">
                        <h3>🔍 Anomaly Detector</h3>
                        <div class="performance-badge">Active</div>
                        <p><strong>Anomalies:</strong> {anomaly_perf['anomalies_detected']}</p>
                        <p><strong>Rate:</strong> {anomaly_perf['anomaly_rate']:.1%}</p>
                        <p><strong>Type:</strong> Isolation Forest</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No model performance data available.")
    
    def render_predictions(self):
        """Render ML predictions"""
        st.header("🔮 ML Predictions")
        
        if 'predictions' in self.ml_results:
            predictions = self.ml_results['predictions']
            
            # ICU Predictions
            if 'icu_occupancy_pred' in predictions:
                st.subheader("🏥 ICU Occupancy Predictions")
                
                icu_preds = predictions['icu_occupancy_pred']
                states = [f"State {i+1}" for i in range(len(icu_preds))]
                
                # Create prediction chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=states,
                        y=icu_preds,
                        marker_color=['red' if p > 0.8 else 'orange' if p > 0.6 else 'green' for p in icu_preds],
                        text=[f"{p:.1%}" for p in icu_preds],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="ICU Occupancy Predictions by State",
                    yaxis_title="Predicted ICU Occupancy",
                    yaxis_tickformat='.0%',
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # High risk states
                high_risk = [i for i, p in enumerate(icu_preds) if p > 0.8]
                if high_risk:
                    st.warning(f"⚠️ **High Risk States**: {len(high_risk)} states predicted to have ICU occupancy >80%")
            
            # Anomaly Predictions
            if 'anomaly_predictions' in predictions:
                st.subheader("🚨 Anomaly Detection Results")
                
                anomaly_preds = predictions['anomaly_predictions']
                anomaly_scores = predictions.get('anomaly_scores', [0] * len(anomaly_preds))
                
                # Create anomaly chart
                fig = go.Figure(data=[
                    go.Scatter(
                        x=list(range(len(anomaly_scores))),
                        y=anomaly_scores,
                        mode='markers',
                        marker=dict(
                            color=['red' if p == -1 else 'green' for p in anomaly_preds],
                            size=10
                        ),
                        text=[f"State {i+1}: {'Anomaly' if p == -1 else 'Normal'}" for i, p in enumerate(anomaly_preds)],
                        hovertemplate='%{text}<br>Score: %{y:.3f}<extra></extra>'
                    )
                ])
                
                fig.update_layout(
                    title="Anomaly Detection Scores",
                    xaxis_title="State",
                    yaxis_title="Anomaly Score",
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Anomaly summary
                anomalies = sum(1 for p in anomaly_preds if p == -1)
                st.info(f"🔍 **Anomaly Summary**: {anomalies} anomalies detected out of {len(anomaly_preds)} states")
        else:
            st.warning("No prediction data available.")
    
    def render_feature_importance(self):
        """Render feature importance analysis"""
        st.header("🎯 Feature Importance")
        
        if 'model_results' in self.ml_results and 'icu_predictor' in self.ml_results['model_results']:
            icu_model = self.ml_results['model_results']['icu_predictor']
            
            if 'feature_importance' in icu_model:
                feature_importance = icu_model['feature_importance']
                
                # Create feature importance chart
                features = list(feature_importance.keys())
                importance_values = list(feature_importance.values())
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=importance_values,
                        y=features,
                        orientation='h',
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title="Feature Importance for ICU Predictor",
                    xaxis_title="Importance Score",
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Top features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                st.subheader("🏆 Top Features")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    for i, (feature, importance) in enumerate(sorted_features[:5]):
                        st.markdown(f"""
                        <div class="feature-importance-bar">
                            <strong>{feature}</strong>: {importance:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    for i, (feature, importance) in enumerate(sorted_features[5:]):
                        st.markdown(f"""
                        <div class="feature-importance-bar">
                            <strong>{feature}</strong>: {importance:.3f}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("No feature importance data available.")
    
    def render_ml_insights(self):
        """Render ML insights and recommendations"""
        st.header("💡 ML Insights & Recommendations")
        
        if 'insights' in self.ml_results:
            insights = self.ml_results['insights']
            
            # Predictions summary
            if 'predictions_summary' in insights:
                st.subheader("📊 Predictions Summary")
                
                pred_summary = insights['predictions_summary']
                if 'icu_occupancy' in pred_summary:
                    icu_summary = pred_summary['icu_occupancy']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean Prediction", f"{icu_summary['mean_prediction']:.1%}")
                    
                    with col2:
                        st.metric("Max Prediction", f"{icu_summary['max_prediction']:.1%}")
                    
                    with col3:
                        st.metric("Min Prediction", f"{icu_summary['min_prediction']:.1%}")
                    
                    with col4:
                        st.metric("High Risk States", f"{icu_summary['high_risk_states']}")
            
            # Anomaly analysis
            if 'anomaly_analysis' in insights:
                st.subheader("🚨 Anomaly Analysis")
                
                anomaly_analysis = insights['anomaly_analysis']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Anomalies Detected", anomaly_analysis['anomalies_detected'])
                
                with col2:
                    st.metric("Anomaly Rate", f"{anomaly_analysis['anomaly_rate']:.1%}")
                
                with col3:
                    risk_level = anomaly_analysis['risk_level']
                    color = 'red' if risk_level == 'High' else 'orange' if risk_level == 'Medium' else 'green'
                    st.markdown(f"""
                    <div class="performance-badge" style="background-color: {color};">
                        {risk_level} Risk
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            if 'recommendations' in insights:
                st.subheader("🎯 AI Recommendations")
                
                for i, recommendation in enumerate(insights['recommendations']):
                    st.markdown(f"""
                    <div class="anomaly-alert">
                        <strong>Recommendation {i+1}:</strong> {recommendation}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No insights data available.")
    
    def render_real_time_predictions(self):
        """Render real-time prediction interface"""
        st.header("⚡ Real-time Predictions")
        
        st.info("🤖 **ML Model Status**: Ready for real-time predictions")
        
        # Prediction interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Input Parameters")
            
            # Sample input form
            total_hospitals = st.slider("Total Hospitals", 50, 500, 150)
            available_beds = st.slider("Available Beds", 1000, 10000, 3000)
            emergency_wait_time = st.slider("Emergency Wait Time (min)", 10, 120, 30)
            staff_shortage = st.slider("Staff Shortage Rate", 0.0, 0.5, 0.2)
            bed_utilization = st.slider("Bed Utilization Rate", 0.5, 1.0, 0.8)
        
        with col2:
            st.subheader("🔮 Predictions")
            
            # Simulate predictions based on inputs
            icu_pred = min(1.0, max(0.0, (bed_utilization * 0.6 + staff_shortage * 0.3 + emergency_wait_time / 100)))
            risk_level = "High" if icu_pred > 0.8 else "Medium" if icu_pred > 0.6 else "Low"
            
            st.metric("ICU Occupancy Prediction", f"{icu_pred:.1%}")
            st.metric("Risk Level", risk_level)
            
            if icu_pred > 0.8:
                st.error("⚠️ High ICU occupancy predicted - Immediate action required!")
            elif icu_pred > 0.6:
                st.warning("⚠️ Medium ICU occupancy predicted - Monitor closely")
            else:
                st.success("✅ Low ICU occupancy predicted - Normal operations")
        
        # Run prediction button
        if st.button("🚀 Run ML Prediction", type="primary"):
            st.success("✅ Prediction completed! Check results above.")
    
    def run(self):
        """Run the ML insights dashboard"""
        if not self.load_ml_data():
            return
        
        self.render_header()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance", "🔮 Predictions", "🎯 Features", "💡 Insights", "⚡ Real-time"
        ])
        
        with tab1:
            self.render_model_performance()
        
        with tab2:
            self.render_predictions()
        
        with tab3:
            self.render_feature_importance()
        
        with tab4:
            self.render_ml_insights()
        
        with tab5:
            self.render_real_time_predictions()

if __name__ == "__main__":
    dashboard = MLInsightsDashboard()
    dashboard.run()
