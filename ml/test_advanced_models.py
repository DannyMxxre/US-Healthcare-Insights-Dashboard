#!/usr/bin/env python3
"""
Simplified Advanced ML Test V2.2
Test basic functionality without complex dependencies
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Basic ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAdvancedML:
    """Simplified Advanced ML for testing"""
    
    def __init__(self):
        self.models_dir = Path('ml/saved_models')
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.results = {}
    
    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare data for ML"""
        logger.info("🔧 Preparing data for ML...")
        
        # Start with hospital data
        if 'hospital_status' in data and not data['hospital_status'].empty:
            df = data['hospital_status'].copy()
        else:
            # Create sample data
            df = pd.DataFrame({
                'state': ['CA', 'TX', 'NY', 'FL', 'IL'],
                'total_hospitals': [150, 200, 180, 120, 160],
                'available_beds': [3000, 4000, 3500, 2500, 3200],
                'icu_occupancy': [0.75, 0.85, 0.70, 0.80, 0.90],
                'emergency_wait_time': [25, 45, 30, 40, 50],
                'staff_shortage': [0.15, 0.25, 0.20, 0.30, 0.35]
            })
        
        # Add derived features
        df['bed_utilization'] = 1 - (df['available_beds'] / (df['total_hospitals'] * 50))
        df['staff_efficiency'] = 1 - df['staff_shortage']
        df['emergency_load'] = df['emergency_wait_time'] * df['icu_occupancy']
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        logger.info(f"✅ Prepared dataset: {df.shape}")
        return df
    
    def train_icu_predictor(self, data: pd.DataFrame):
        """Train ICU occupancy predictor"""
        logger.info("🤖 Training ICU occupancy predictor...")
        
        # Prepare features
        feature_cols = ['total_hospitals', 'available_beds', 'emergency_wait_time', 
                       'staff_shortage', 'bed_utilization', 'staff_efficiency', 'emergency_load']
        
        X = data[feature_cols].fillna(0)
        y = data['icu_occupancy'].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        model_name = "icu_predictor"
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        self.results[model_name] = {
            'mse': mse,
            'r2': r2,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }
        
        logger.info(f"✅ ICU predictor trained - R²: {r2:.3f}, MSE: {mse:.4f}")
        return model, scaler
    
    def train_anomaly_detector(self, data: pd.DataFrame):
        """Train anomaly detection model"""
        logger.info("🔍 Training anomaly detector...")
        
        # Prepare features
        feature_cols = ['icu_occupancy', 'emergency_wait_time', 'staff_shortage', 
                       'bed_utilization', 'emergency_load']
        
        X = data[feature_cols].fillna(0)
        
        # Train Isolation Forest
        anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        anomaly_detector.fit(X)
        
        # Detect anomalies
        anomaly_scores = anomaly_detector.decision_function(X)
        anomaly_predictions = anomaly_detector.predict(X)
        
        # Find anomalies
        anomalies = data[anomaly_predictions == -1]
        
        # Save model
        model_name = "anomaly_detector"
        self.models[model_name] = anomaly_detector
        
        self.results[model_name] = {
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / len(data),
            'anomaly_scores': anomaly_scores.tolist()
        }
        
        logger.info(f"✅ Anomaly detector trained - {len(anomalies)} anomalies detected")
        return anomaly_detector
    
    def train_sentiment_analyzer(self, news_data: pd.DataFrame):
        """Train sentiment analysis model"""
        logger.info("📝 Training sentiment analyzer...")
        
        if news_data.empty or 'title' not in news_data.columns or 'sentiment' not in news_data.columns:
            logger.warning("⚠️ No news data available for sentiment analysis")
            return None
        
        # Prepare text data
        texts = news_data['title'].fillna("").tolist()
        labels = news_data['sentiment'].fillna("Neutral").tolist()
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Train classifier
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X, labels)
        
        # Evaluate
        y_pred = classifier.predict(X)
        accuracy = np.mean(y_pred == labels)
        
        # Save model
        model_name = "sentiment_analyzer"
        self.models[model_name] = classifier
        self.scalers[model_name] = vectorizer  # Reusing scalers dict for vectorizer
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'unique_sentiments': list(set(labels))
        }
        
        logger.info(f"✅ Sentiment analyzer trained - Accuracy: {accuracy:.3f}")
        return classifier, vectorizer
    
    def make_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using trained models"""
        logger.info("⚡ Making predictions...")
        
        predictions = {}
        
        # ICU predictions
        if 'icu_predictor' in self.models:
            model = self.models['icu_predictor']
            scaler = self.scalers['icu_predictor']
            
            feature_cols = ['total_hospitals', 'available_beds', 'emergency_wait_time', 
                           'staff_shortage', 'bed_utilization', 'staff_efficiency', 'emergency_load']
            
            X_pred = data[feature_cols].fillna(0)
            X_pred_scaled = scaler.transform(X_pred)
            
            icu_predictions = model.predict(X_pred_scaled)
            predictions['icu_occupancy_pred'] = icu_predictions.tolist()
        
        # Anomaly detection
        if 'anomaly_detector' in self.models:
            anomaly_detector = self.models['anomaly_detector']
            
            feature_cols = ['icu_occupancy', 'emergency_wait_time', 'staff_shortage', 
                          'bed_utilization', 'emergency_load']
            
            X_anomaly = data[feature_cols].fillna(0)
            anomaly_scores = anomaly_detector.decision_function(X_anomaly)
            anomaly_predictions = anomaly_detector.predict(X_anomaly)
            
            predictions['anomaly_scores'] = anomaly_scores.tolist()
            predictions['anomaly_predictions'] = anomaly_predictions.tolist()
            predictions['anomalies_detected'] = int(np.sum(anomaly_predictions == -1))
        
        logger.info(f"✅ Predictions completed: {len(predictions)} models")
        return predictions
    
    def generate_insights(self, data: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from predictions"""
        logger.info("💡 Generating insights...")
        
        insights = {
            'model_performance': {},
            'predictions_summary': {},
            'anomaly_analysis': {},
            'recommendations': []
        }
        
        # Model performance
        for model_name, result in self.results.items():
            if 'r2' in result:
                insights['model_performance'][model_name] = {
                    'r2_score': result['r2'],
                    'mse': result['mse'],
                    'performance_level': 'Excellent' if result['r2'] > 0.8 else 'Good' if result['r2'] > 0.6 else 'Fair'
                }
            elif 'accuracy' in result:
                insights['model_performance'][model_name] = {
                    'accuracy': result['accuracy'],
                    'performance_level': 'Excellent' if result['accuracy'] > 0.9 else 'Good' if result['accuracy'] > 0.7 else 'Fair'
                }
        
        # Predictions summary
        if 'icu_occupancy_pred' in predictions:
            icu_preds = predictions['icu_occupancy_pred']
            insights['predictions_summary']['icu_occupancy'] = {
                'mean_prediction': np.mean(icu_preds),
                'max_prediction': np.max(icu_preds),
                'min_prediction': np.min(icu_preds),
                'high_risk_states': len([p for p in icu_preds if p > 0.8])
            }
        
        # Anomaly analysis
        if 'anomaly_predictions' in predictions:
            anomaly_count = predictions['anomalies_detected']
            total_records = len(predictions['anomaly_predictions'])
            anomaly_rate = anomaly_count / total_records
            
            insights['anomaly_analysis'] = {
                'anomalies_detected': anomaly_count,
                'anomaly_rate': anomaly_rate,
                'risk_level': 'High' if anomaly_rate > 0.15 else 'Medium' if anomaly_rate > 0.05 else 'Low'
            }
        
        # Generate recommendations
        recommendations = []
        
        if 'icu_occupancy_pred' in predictions:
            high_icu_states = [i for i, p in enumerate(predictions['icu_occupancy_pred']) if p > 0.8]
            if high_icu_states:
                recommendations.append(f"⚠️ {len(high_icu_states)} states predicted to have high ICU occupancy (>80%)")
        
        if 'anomaly_predictions' in predictions:
            if predictions['anomalies_detected'] > 0:
                recommendations.append(f"🚨 {predictions['anomalies_detected']} anomalies detected - requires immediate attention")
        
        insights['recommendations'] = recommendations
        
        logger.info(f"✅ Generated {len(recommendations)} recommendations")
        return insights
    
    def train_all_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all models"""
        logger.info("🚀 Training all models...")
        
        # Prepare data
        healthcare_data = self.prepare_data(data)
        
        # Train models
        models_trained = {}
        
        # ICU predictor
        models_trained['icu_predictor'] = self.train_icu_predictor(healthcare_data)
        
        # Anomaly detector
        models_trained['anomaly_detector'] = self.train_anomaly_detector(healthcare_data)
        
        # Sentiment analyzer
        if 'health_news' in data and not data['health_news'].empty:
            models_trained['sentiment_analyzer'] = self.train_sentiment_analyzer(data['health_news'])
        
        # Make predictions
        predictions = self.make_predictions(healthcare_data)
        
        # Generate insights
        insights = self.generate_insights(healthcare_data, predictions)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_summary = {
            'timestamp': timestamp,
            'models_trained': len(models_trained),
            'predictions': predictions,
            'insights': insights,
            'model_results': self.results
        }
        
        # Save to file
        with open(self.models_dir / f'simple_ml_results_{timestamp}.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"🎉 ML training completed - {len(models_trained)} models trained")
        return results_summary

def main():
    """Main function"""
    ml_system = SimpleAdvancedML()
    
    # Load data
    data_dir = Path('data/realtime')
    data = {}
    
    for data_type in ['covid_data', 'weather_data', 'health_news', 'hospital_status', 'emergency_alerts']:
        files = list(data_dir.glob(f"{data_type}_*.csv"))
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            data[data_type] = pd.read_csv(latest_file)
            print(f"✅ Loaded {data_type}: {len(data[data_type])} records")
    
    # Train models
    results = ml_system.train_all_models(data)
    
    print("\n🎉 Advanced ML training completed!")
    print(f"📊 Models trained: {results['models_trained']}")
    print(f"💡 Insights generated: {len(results['insights']['recommendations'])} recommendations")
    
    # Print some results
    if 'predictions' in results:
        if 'icu_occupancy_pred' in results['predictions']:
            icu_preds = results['predictions']['icu_occupancy_pred']
            print(f"🏥 ICU predictions: {len(icu_preds)} states, avg: {np.mean(icu_preds):.3f}")
        
        if 'anomalies_detected' in results['predictions']:
            print(f"🚨 Anomalies detected: {results['predictions']['anomalies_detected']}")

if __name__ == "__main__":
    main()
