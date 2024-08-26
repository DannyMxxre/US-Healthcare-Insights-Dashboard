#!/usr/bin/env python3
"""
Working Advanced ML Models V2.2
Deep Learning alternatives, NLP, Anomaly Detection, and Real-time Predictions
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingAdvancedHealthcareML:
    """Working Advanced ML models for healthcare analytics"""
    
    def __init__(self):
        self.models_dir = Path('ml/saved_models')
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.tokenizers = {}
        self.results = {}
        
        # Model configurations
        self.model_configs = {
            'neural_network': {
                'hidden_layer_sizes': (100, 50, 25),
                'max_iter': 1000,
                'learning_rate_init': 0.001,
                'early_stopping': True
            },
            'nlp': {
                'max_words': 10000,
                'max_length': 100,
                'embedding_dim': 16
            },
            'anomaly_detection': {
                'contamination': 0.1,
                'n_estimators': 100
            }
        }
    
    def prepare_healthcare_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare comprehensive healthcare dataset for ML"""
        logger.info("🔧 Preparing healthcare data for ML...")
        
        # Start with hospital data
        if 'hospital_status' in data and not data['hospital_status'].empty:
            df = data['hospital_status'].copy()
        else:
            # Create sample data if not available
            df = pd.DataFrame({
                'state': ['CA', 'TX', 'NY', 'FL', 'IL'],
                'total_hospitals': [150, 200, 180, 120, 160],
                'available_beds': [3000, 4000, 3500, 2500, 3200],
                'icu_occupancy': [0.75, 0.85, 0.70, 0.80, 0.90],
                'emergency_wait_time': [25, 45, 30, 40, 50],
                'staff_shortage': [0.15, 0.25, 0.20, 0.30, 0.35]
            })
        
        # Add COVID data if available
        if 'covid_data' in data and not data['covid_data'].empty:
            covid_df = data['covid_data'].copy()
            # Merge on state
            df = df.merge(covid_df[['state', 'positive_rate', 'hospitalization_rate']], 
                         on='state', how='left')
        
        # Add weather data if available
        if 'weather_data' in data and not data['weather_data'].empty:
            weather_df = data['weather_data'].copy()
            # Aggregate weather data by state (simplified)
            weather_agg = weather_df.groupby('city').agg({
                'temperature': 'mean',
                'humidity': 'mean',
                'pressure': 'mean'
            }).reset_index()
            weather_agg['state'] = weather_agg['city'].map({
                'New York': 'NY', 'Los Angeles': 'CA', 'Chicago': 'IL',
                'Houston': 'TX', 'Phoenix': 'AZ'
            })
            df = df.merge(weather_agg[['state', 'temperature', 'humidity', 'pressure']], 
                         on='state', how='left')
        
        # Add derived features
        df['bed_utilization'] = 1 - (df['available_beds'] / (df['total_hospitals'] * 50))
        df['staff_efficiency'] = 1 - df['staff_shortage']
        df['emergency_load'] = df['emergency_wait_time'] * df['icu_occupancy']
        
        # Handle missing values - only numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        logger.info(f"✅ Prepared dataset: {df.shape}")
        return df
    
    def build_neural_network_model(self, input_dim: int, output_dim: int = 1):
        """Build neural network model using sklearn MLP"""
        config = self.model_configs['neural_network']
        
        if output_dim == 1:
            # Regression
            model = MLPRegressor(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                max_iter=config['max_iter'],
                learning_rate_init=config['learning_rate_init'],
                early_stopping=config['early_stopping'],
                random_state=42
            )
        else:
            # Classification
            model = MLPClassifier(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                max_iter=config['max_iter'],
                learning_rate_init=config['learning_rate_init'],
                early_stopping=config['early_stopping'],
                random_state=42
            )
        
        return model
    
    def train_neural_network_predictor(self, data: pd.DataFrame, target_col: str = 'icu_occupancy'):
        """Train neural network model for healthcare predictions"""
        logger.info(f"🤖 Training Neural Network model for {target_col}...")
        
        # Prepare features
        feature_cols = ['total_hospitals', 'available_beds', 'emergency_wait_time', 
                       'staff_shortage', 'bed_utilization', 'staff_efficiency', 'emergency_load']
        
        # Add COVID features if available
        if 'positive_rate' in data.columns:
            feature_cols.append('positive_rate')
        if 'hospitalization_rate' in data.columns:
            feature_cols.append('hospitalization_rate')
        
        # Add weather features if available
        if 'temperature' in data.columns:
            feature_cols.extend(['temperature', 'humidity', 'pressure'])
        
        X = data[feature_cols].fillna(0)
        y = data[target_col].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build and train model
        model = self.build_neural_network_model(X_train.shape[1])
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model and results
        model_name = f"neural_network_{target_col}"
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        self.results[model_name] = {
            'mse': mse,
            'r2': r2,
            'feature_importance': dict(zip(feature_cols, np.ones(len(feature_cols)))),  # Simplified
            'model_type': 'Neural Network (MLP)'
        }
        
        # Save model
        joblib.dump(model, self.models_dir / f"{model_name}.pkl")
        joblib.dump(scaler, self.models_dir / f"{model_name}_scaler.pkl")
        
        logger.info(f"✅ Neural Network model trained - R²: {r2:.3f}, MSE: {mse:.4f}")
        return model, scaler
    
    def build_nlp_model(self, texts: List[str], labels: List[str]):
        """Build NLP model for text analysis"""
        logger.info("📝 Building NLP model for text analysis...")
        
        # Prepare text data
        processed_texts = []
        for text in texts:
            if pd.isna(text):
                processed_texts.append("")
                continue
            
            # Basic text preprocessing
            text_lower = str(text).lower()
            # Simple tokenization (split by spaces)
            tokens = text_lower.split()
            processed_texts.append(" ".join(tokens))
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=self.model_configs['nlp']['max_words'],
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(processed_texts)
        
        # Train classifier
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X, labels)
        
        # Save model
        model_name = "nlp_sentiment"
        self.models[model_name] = classifier
        self.tokenizers[model_name] = vectorizer
        
        # Evaluate
        y_pred = classifier.predict(X)
        accuracy = np.mean(y_pred == labels)
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(labels, y_pred, output_dict=True),
            'model_type': 'NLP Sentiment Analysis'
        }
        
        # Save model
        joblib.dump(classifier, self.models_dir / f"{model_name}.pkl")
        joblib.dump(vectorizer, self.models_dir / f"{model_name}_vectorizer.pkl")
        
        logger.info(f"✅ NLP model trained - Accuracy: {accuracy:.3f}")
        return classifier, vectorizer
    
    def train_anomaly_detector(self, data: pd.DataFrame):
        """Train anomaly detection model"""
        logger.info("🔍 Training Anomaly Detection model...")
        
        # Prepare features for anomaly detection
        feature_cols = ['icu_occupancy', 'emergency_wait_time', 'staff_shortage', 
                       'bed_utilization', 'emergency_load']
        
        # Add COVID features if available
        if 'positive_rate' in data.columns:
            feature_cols.append('positive_rate')
        if 'hospitalization_rate' in data.columns:
            feature_cols.append('hospitalization_rate')
        
        X = data[feature_cols].fillna(0)
        
        # Train Isolation Forest
        config = self.model_configs['anomaly_detection']
        anomaly_detector = IsolationForest(
            contamination=config['contamination'],
            n_estimators=config['n_estimators'],
            random_state=42
        )
        
        anomaly_detector.fit(X)
        
        # Detect anomalies
        anomaly_scores = anomaly_detector.decision_function(X)
        anomaly_predictions = anomaly_detector.predict(X)
        
        # Find anomalies (predictions == -1)
        anomalies = data[anomaly_predictions == -1]
        
        # Save model
        model_name = "anomaly_detector"
        self.models[model_name] = anomaly_detector
        
        self.results[model_name] = {
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / len(data),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_data': anomalies.to_dict('records'),
            'model_type': 'Isolation Forest Anomaly Detection'
        }
        
        # Save model
        joblib.dump(anomaly_detector, self.models_dir / f"{model_name}.pkl")
        
        logger.info(f"✅ Anomaly detector trained - {len(anomalies)} anomalies detected")
        return anomaly_detector
    
    def build_automl_pipeline(self, data: pd.DataFrame, target_col: str):
        """Build AutoML pipeline for automatic model selection"""
        logger.info(f"🎯 Building AutoML pipeline for {target_col}...")
        
        # Prepare features
        feature_cols = [col for col in data.columns if col != target_col and col != 'state']
        X = data[feature_cols].fillna(0)
        y = data[target_col].fillna(0)
        
        # Define models to try
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'SVR': SVR(kernel='rbf'),
            'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        results = {}
        
        # Test each model
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                mean_score = cv_scores.mean()
                
                results[name] = {
                    'mean_score': mean_score,
                    'std_score': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                
                logger.info(f"  {name}: R² = {mean_score:.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                logger.warning(f"  {name}: Error - {e}")
        
        # Train best model on full dataset
        if best_model is not None:
            best_model.fit(X, y)
            
            # Save best model
            model_name = f"automl_{target_col}"
            self.models[model_name] = best_model
            
            self.results[model_name] = {
                'best_model': type(best_model).__name__,
                'best_score': best_score,
                'all_results': results,
                'model_type': 'AutoML Pipeline'
            }
            
            # Save model
            joblib.dump(best_model, self.models_dir / f"{model_name}.pkl")
            
            logger.info(f"✅ AutoML completed - Best: {type(best_model).__name__} (R² = {best_score:.3f})")
            return best_model
        
        return None
    
    def make_realtime_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make real-time predictions using trained models"""
        logger.info("⚡ Making real-time predictions...")
        
        predictions = {}
        
        # ICU Occupancy Prediction
        if 'neural_network_icu_occupancy' in self.models:
            model = self.models['neural_network_icu_occupancy']
            scaler = self.scalers['neural_network_icu_occupancy']
            
            feature_cols = ['total_hospitals', 'available_beds', 'emergency_wait_time', 
                           'staff_shortage', 'bed_utilization', 'staff_efficiency', 'emergency_load']
            
            # Add additional features if available
            for col in ['positive_rate', 'hospitalization_rate', 'temperature', 'humidity', 'pressure']:
                if col in data.columns:
                    feature_cols.append(col)
            
            X_pred = data[feature_cols].fillna(0)
            X_pred_scaled = scaler.transform(X_pred)
            
            icu_predictions = model.predict(X_pred_scaled)
            predictions['icu_occupancy_pred'] = icu_predictions.tolist()
        
        # Anomaly Detection
        if 'anomaly_detector' in self.models:
            anomaly_detector = self.models['anomaly_detector']
            
            feature_cols = ['icu_occupancy', 'emergency_wait_time', 'staff_shortage', 
                          'bed_utilization', 'emergency_load']
            
            for col in ['positive_rate', 'hospitalization_rate']:
                if col in data.columns:
                    feature_cols.append(col)
            
            X_anomaly = data[feature_cols].fillna(0)
            anomaly_scores = anomaly_detector.decision_function(X_anomaly)
            anomaly_predictions = anomaly_detector.predict(X_anomaly)
            
            predictions['anomaly_scores'] = anomaly_scores.tolist()
            predictions['anomaly_predictions'] = anomaly_predictions.tolist()
            predictions['anomalies_detected'] = int(np.sum(anomaly_predictions == -1))
        
        # AutoML Predictions
        for model_name in self.models:
            if model_name.startswith('automl_'):
                target_col = model_name.replace('automl_', '')
                if target_col in data.columns:
                    model = self.models[model_name]
                    
                    feature_cols = [col for col in data.columns if col != target_col and col != 'state']
                    X_pred = data[feature_cols].fillna(0)
                    
                    pred_values = model.predict(X_pred)
                    predictions[f'{target_col}_pred'] = pred_values.tolist()
        
        logger.info(f"✅ Real-time predictions completed: {len(predictions)} models")
        return predictions
    
    def generate_ml_insights(self, data: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from ML models and predictions"""
        logger.info("💡 Generating ML insights...")
        
        insights = {
            'model_performance': {},
            'predictions_summary': {},
            'anomaly_analysis': {},
            'feature_importance': {},
            'recommendations': []
        }
        
        # Model performance insights
        for model_name, result in self.results.items():
            if 'r2' in result:
                insights['model_performance'][model_name] = {
                    'r2_score': result['r2'],
                    'mse': result['mse'],
                    'performance_level': 'Excellent' if result['r2'] > 0.8 else 'Good' if result['r2'] > 0.6 else 'Fair',
                    'model_type': result.get('model_type', 'Unknown')
                }
            elif 'accuracy' in result:
                insights['model_performance'][model_name] = {
                    'accuracy': result['accuracy'],
                    'performance_level': 'Excellent' if result['accuracy'] > 0.9 else 'Good' if result['accuracy'] > 0.7 else 'Fair',
                    'model_type': result.get('model_type', 'Unknown')
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
    
    def train_all_advanced_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all advanced ML models"""
        logger.info("🚀 Training all advanced ML models...")
        
        # Prepare data
        healthcare_data = self.prepare_healthcare_data(data)
        
        # Train models
        models_trained = {}
        
        # Neural Network models
        models_trained['neural_network_icu'] = self.train_neural_network_predictor(
            healthcare_data, 'icu_occupancy'
        )
        
        # NLP model (if news data available)
        if 'health_news' in data and not data['health_news'].empty:
            news_data = data['health_news']
            if 'title' in news_data.columns and 'sentiment' in news_data.columns:
                models_trained['nlp_sentiment'] = self.build_nlp_model(
                    news_data['title'].tolist(),
                    news_data['sentiment'].tolist()
                )
        
        # Anomaly detection
        models_trained['anomaly_detector'] = self.train_anomaly_detector(healthcare_data)
        
        # AutoML pipelines
        for target_col in ['icu_occupancy', 'emergency_wait_time', 'staff_shortage']:
            if target_col in healthcare_data.columns:
                models_trained[f'automl_{target_col}'] = self.build_automl_pipeline(
                    healthcare_data, target_col
                )
        
        # Make predictions
        predictions = self.make_realtime_predictions(healthcare_data)
        
        # Generate insights
        insights = self.generate_ml_insights(healthcare_data, predictions)
        
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
        with open(self.models_dir / f'advanced_ml_results_{timestamp}.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"🎉 Advanced ML training completed - {len(models_trained)} models trained")
        return results_summary

def main():
    """Main function for testing advanced ML models"""
    ml_system = WorkingAdvancedHealthcareML()
    
    # Load sample data
    data_dir = Path('data/realtime')
    data = {}
    
    for data_type in ['covid_data', 'weather_data', 'health_news', 'hospital_status', 'emergency_alerts']:
        files = list(data_dir.glob(f"{data_type}_*.csv"))
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            data[data_type] = pd.read_csv(latest_file)
            print(f"✅ Loaded {data_type}: {len(data[data_type])} records")
    
    # Train all models
    results = ml_system.train_all_advanced_models(data)
    
    print("\n🎉 Advanced ML training completed!")
    print(f"📊 Models trained: {results['models_trained']}")
    print(f"💡 Insights generated: {len(results['insights']['recommendations'])} recommendations")
    
    # Print model performance
    print("\n📈 Model Performance:")
    for model_name, perf in results['insights']['model_performance'].items():
        if 'r2_score' in perf:
            print(f"  {model_name}: R² = {perf['r2_score']:.3f} ({perf['performance_level']})")
        elif 'accuracy' in perf:
            print(f"  {model_name}: Accuracy = {perf['accuracy']:.3f} ({perf['performance_level']})")

if __name__ == "__main__":
    main()
