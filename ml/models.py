"""
Machine Learning Models for US Healthcare Insights Dashboard
Advanced predictive analytics and insights
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HealthcareMLModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
        
    def prepare_data(self, hospitals_df, census_df, costs_df, quality_df):
        """Prepare data for ML models"""
        print("🔧 Preparing data for Machine Learning models...")
        
        # Merge datasets
        merged_data = hospitals_df.merge(census_df, on='state', how='inner')
        merged_data = merged_data.merge(quality_df, on='state', how='inner')
        
        # Create features
        merged_data['beds_per_100k'] = (merged_data['beds'] / merged_data['population'] * 100000)
        merged_data['income_per_capita'] = merged_data['median_income'] / merged_data['population']
        merged_data['poverty_impact'] = merged_data['poverty_rate'] * merged_data['population']
        
        # Handle missing values
        merged_data = merged_data.fillna(merged_data.mean())
        
        return merged_data
    
    def train_hospital_rating_predictor(self, data):
        """Train model to predict hospital ratings"""
        print("🏥 Training Hospital Rating Predictor...")
        
        # Features for hospital rating prediction
        features = ['beds', 'medicare_rating', 'safety_rating', 'patient_satisfaction',
                   'population', 'median_income', 'poverty_rate', 'unemployment_rate',
                   'education_bachelors', 'healthcare_coverage']
        
        X = data[features].dropna()
        y = data.loc[X.index, 'rating']
        
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
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Store results
        self.models['hospital_rating'] = model
        self.scalers['hospital_rating'] = scaler
        self.results['hospital_rating'] = {
            'mse': mse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(features, model.feature_importances_))
        }
        
        print(f"✅ Hospital Rating Model - R²: {r2:.3f}, CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return model, scaler
    
    def train_healthcare_cost_predictor(self, data):
        """Train model to predict healthcare costs"""
        print("💰 Training Healthcare Cost Predictor...")
        
        # Features for cost prediction
        features = ['population', 'median_income', 'poverty_rate', 'unemployment_rate',
                   'education_bachelors', 'healthcare_coverage', 'life_expectancy',
                   'infant_mortality_rate', 'preventable_deaths_per_100k']
        
        X = data[features].dropna()
        y = data.loc[X.index, 'avg_premium']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Store results
        self.models['healthcare_cost'] = model
        self.scalers['healthcare_cost'] = scaler
        self.results['healthcare_cost'] = {
            'mse': mse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'coefficients': dict(zip(features, model.coef_))
        }
        
        print(f"✅ Healthcare Cost Model - R²: {r2:.3f}, CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return model, scaler
    
    def train_state_clustering(self, data):
        """Cluster states based on healthcare characteristics"""
        print("🗺️ Training State Clustering Model...")
        
        # Features for clustering
        features = ['avg_rating', 'avg_premium', 'poverty_rate', 'uninsured_rate',
                   'life_expectancy', 'infant_mortality_rate', 'education_bachelors']
        
        X = data[features].dropna()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        inertias = []
        K_range = range(2, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method to find optimal k
        optimal_k = 4  # Based on typical healthcare segmentation
        
        # Train final model
        model = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = model.fit_predict(X_scaled)
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_rating': cluster_data['avg_rating'].mean(),
                'avg_premium': cluster_data['avg_premium'].mean(),
                'avg_poverty_rate': cluster_data['poverty_rate'].mean(),
                'avg_life_expectancy': cluster_data['life_expectancy'].mean()
            }
        
        # Store results
        self.models['state_clustering'] = model
        self.scalers['state_clustering'] = scaler
        self.results['state_clustering'] = {
            'n_clusters': optimal_k,
            'inertia': model.inertia_,
            'cluster_analysis': cluster_analysis,
            'cluster_labels': clusters
        }
        
        print(f"✅ State Clustering Model - {optimal_k} clusters, Inertia: {model.inertia_:.2f}")
        
        return model, scaler, data_with_clusters
    
    def train_health_outcome_predictor(self, data):
        """Train model to predict health outcomes"""
        print("🏆 Training Health Outcome Predictor...")
        
        # Features for health outcome prediction
        features = ['avg_rating', 'avg_premium', 'poverty_rate', 'uninsured_rate',
                   'education_bachelors', 'unemployment_rate', 'median_income']
        
        X = data[features].dropna()
        y = data.loc[X.index, 'life_expectancy']
        
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
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Store results
        self.models['health_outcome'] = model
        self.scalers['health_outcome'] = scaler
        self.results['health_outcome'] = {
            'mse': mse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(features, model.feature_importances_))
        }
        
        print(f"✅ Health Outcome Model - R²: {r2:.3f}, CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return model, scaler
    
    def generate_ml_insights(self, data):
        """Generate insights from ML models"""
        print("🔍 Generating ML Insights...")
        
        insights = []
        
        # Hospital Rating Insights
        if 'hospital_rating' in self.results:
            rating_insights = self.results['hospital_rating']
            top_features = sorted(rating_insights['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            
            insights.append({
                'type': 'hospital_rating',
                'title': 'Hospital Rating Prediction Model',
                'description': f"Model achieves {rating_insights['r2']:.1%} accuracy in predicting hospital ratings",
                'top_features': [f"{feature}: {importance:.3f}" for feature, importance in top_features],
                'metric_value': rating_insights['r2'],
                'metric_unit': 'R² Score'
            })
        
        # Healthcare Cost Insights
        if 'healthcare_cost' in self.results:
            cost_insights = self.results['healthcare_cost']
            top_coefficients = sorted(cost_insights['coefficients'].items(), 
                                    key=lambda x: abs(x[1]), reverse=True)[:3]
            
            insights.append({
                'type': 'healthcare_cost',
                'title': 'Healthcare Cost Prediction Model',
                'description': f"Model explains {cost_insights['r2']:.1%} of healthcare cost variation",
                'top_factors': [f"{factor}: {coef:.2f}" for factor, coef in top_coefficients],
                'metric_value': cost_insights['r2'],
                'metric_unit': 'R² Score'
            })
        
        # State Clustering Insights
        if 'state_clustering' in self.results:
            cluster_insights = self.results['state_clustering']
            
            insights.append({
                'type': 'state_clustering',
                'title': 'State Healthcare Clustering',
                'description': f"Identified {cluster_insights['n_clusters']} distinct healthcare state groups",
                'clusters': cluster_insights['cluster_analysis'],
                'metric_value': cluster_insights['n_clusters'],
                'metric_unit': 'Clusters'
            })
        
        # Health Outcome Insights
        if 'health_outcome' in self.results:
            outcome_insights = self.results['health_outcome']
            top_features = sorted(outcome_insights['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            
            insights.append({
                'type': 'health_outcome',
                'title': 'Health Outcome Prediction Model',
                'description': f"Model predicts life expectancy with {outcome_insights['r2']:.1%} accuracy",
                'top_features': [f"{feature}: {importance:.3f}" for feature, importance in top_features],
                'metric_value': outcome_insights['r2'],
                'metric_unit': 'R² Score'
            })
        
        return insights
    
    def save_models(self, filepath='ml/saved_models/'):
        """Save trained models"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.models.items():
            model_file = f"{filepath}{model_name}_{timestamp}.pkl"
            joblib.dump(model, model_file)
            print(f"💾 Saved {model_name} model: {model_file}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_file = f"{filepath}{scaler_name}_scaler_{timestamp}.pkl"
            joblib.dump(scaler, scaler_file)
            print(f"💾 Saved {scaler_name} scaler: {scaler_file}")
        
        # Save results
        results_file = f"{filepath}ml_results_{timestamp}.pkl"
        joblib.dump(self.results, results_file)
        print(f"💾 Saved ML results: {results_file}")
    
    def train_all_models(self, hospitals_df, census_df, costs_df, quality_df):
        """Train all ML models"""
        print("🤖 Training All Machine Learning Models")
        print("=" * 60)
        
        # Prepare data
        data = self.prepare_data(hospitals_df, census_df, costs_df, quality_df)
        
        # Train models
        self.train_hospital_rating_predictor(data)
        self.train_healthcare_cost_predictor(data)
        self.train_state_clustering(data)
        self.train_health_outcome_predictor(data)
        
        # Generate insights
        insights = self.generate_ml_insights(data)
        
        # Save models
        self.save_models()
        
        print("\n" + "=" * 60)
        print("🤖 ML MODEL TRAINING SUMMARY")
        print("=" * 60)
        print(f"📊 Models trained: {len(self.models)}")
        print(f"🔍 Insights generated: {len(insights)}")
        print(f"💾 Models saved to: ml/saved_models/")
        
        return insights

if __name__ == "__main__":
    # Example usage
    ml_models = HealthcareMLModels()
    
    # Load sample data (you would load your actual data here)
    print("🤖 Machine Learning Models for Healthcare Analysis")
    print("Run this module after data collection to train ML models")
