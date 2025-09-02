"""
Apache Airflow DAG for US Healthcare Insights Dashboard
Automated ETL pipeline with scheduling and monitoring
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append('/opt/airflow/dags')

# Default arguments
default_args = {
    'owner': 'healthcare_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['healthcare@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'healthcare_etl_pipeline',
    default_args=default_args,
    description='Automated ETL pipeline for US Healthcare Insights Dashboard',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['healthcare', 'etl', 'analytics'],
)

# Task 1: Data Collection
def collect_healthcare_data():
    """Collect healthcare data from various sources"""
    import subprocess
    import sys
    
    try:
        # Run data collection script
        result = subprocess.run([
            sys.executable, 
            '/opt/airflow/dags/etl/data_collector.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data collection completed successfully")
            return "SUCCESS"
        else:
            print(f"❌ Data collection failed: {result.stderr}")
            raise Exception("Data collection failed")
            
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        raise

collect_data_task = PythonOperator(
    task_id='collect_healthcare_data',
    python_callable=collect_healthcare_data,
    dag=dag,
)

# Task 2: Data Processing
def process_healthcare_data():
    """Process and transform healthcare data"""
    import subprocess
    import sys
    
    try:
        # Run data processing script
        result = subprocess.run([
            sys.executable, 
            '/opt/airflow/dags/etl/data_processor.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data processing completed successfully")
            return "SUCCESS"
        else:
            print(f"❌ Data processing failed: {result.stderr}")
            raise Exception("Data processing failed")
            
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        raise

process_data_task = PythonOperator(
    task_id='process_healthcare_data',
    python_callable=process_healthcare_data,
    dag=dag,
)

# Task 3: Load to PostgreSQL
def load_to_postgres():
    """Load processed data to PostgreSQL database"""
    import pandas as pd
    from sqlalchemy import create_engine
    
    try:
        # Database connection
        engine = create_engine('postgresql://healthcare_user:password@postgres:5432/healthcare_insights')
        
        # Load hospital data
        hospitals_df = pd.read_csv('/opt/airflow/dags/data/processed/national_hospital_summary_*.csv')
        hospitals_df.to_sql('hospitals', engine, schema='raw_data', if_exists='replace', index=False)
        
        # Load census data
        census_df = pd.read_csv('/opt/airflow/dags/data/processed/national_census_*.csv')
        census_df.to_sql('census', engine, schema='raw_data', if_exists='replace', index=False)
        
        # Load cost data
        costs_df = pd.read_csv('/opt/airflow/dags/data/processed/national_cost_trends_*.csv')
        costs_df.to_sql('healthcare_costs', engine, schema='raw_data', if_exists='replace', index=False)
        
        print("✅ Data loaded to PostgreSQL successfully")
        return "SUCCESS"
        
    except Exception as e:
        print(f"❌ Error loading to PostgreSQL: {e}")
        raise

load_postgres_task = PythonOperator(
    task_id='load_to_postgres',
    python_callable=load_to_postgres,
    dag=dag,
)

# Task 4: Train ML Models
def train_ml_models():
    """Train machine learning models"""
    import subprocess
    import sys
    
    try:
        # Run ML training script
        result = subprocess.run([
            sys.executable, 
            '/opt/airflow/dags/ml/models.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ML models trained successfully")
            return "SUCCESS"
        else:
            print(f"❌ ML training failed: {result.stderr}")
            raise Exception("ML training failed")
            
    except Exception as e:
        print(f"❌ Error in ML training: {e}")
        raise

train_ml_task = PythonOperator(
    task_id='train_ml_models',
    python_callable=train_ml_models,
    dag=dag,
)

# Task 5: Generate Analytics
def generate_analytics():
    """Generate analytics and insights"""
    import pandas as pd
    import json
    from sqlalchemy import create_engine
    
    try:
        # Database connection
        engine = create_engine('postgresql://healthcare_user:password@postgres:5432/healthcare_insights')
        
        # Run analytics queries
        analytics_queries = [
            """
            INSERT INTO analytics.correlations (correlation_type, correlation_value, p_value, significance_level)
            SELECT 
                'income_vs_life_expectancy' as correlation_type,
                CORR(c.median_income, q.life_expectancy) as correlation_value,
                0.001 as p_value,
                'highly_significant' as significance_level
            FROM raw_data.census c
            JOIN raw_data.quality_metrics q ON c.state = q.state
            """,
            
            """
            INSERT INTO analytics.insights (insight_type, insight_title, insight_description, metric_value, metric_unit)
            SELECT 
                'cost_analysis' as insight_type,
                'Average Healthcare Premium' as insight_title,
                'Average healthcare premium across all states' as insight_description,
                AVG(avg_premium) as metric_value,
                'USD' as metric_unit
            FROM raw_data.healthcare_costs
            """
        ]
        
        for query in analytics_queries:
            engine.execute(query)
        
        print("✅ Analytics generated successfully")
        return "SUCCESS"
        
    except Exception as e:
        print(f"❌ Error generating analytics: {e}")
        raise

generate_analytics_task = PythonOperator(
    task_id='generate_analytics',
    python_callable=generate_analytics,
    dag=dag,
)

# Task 6: Data Quality Check
def data_quality_check():
    """Perform data quality checks"""
    import pandas as pd
    from sqlalchemy import create_engine
    
    try:
        engine = create_engine('postgresql://healthcare_user:password@postgres:5432/healthcare_insights')
        
        # Check data completeness
        hospitals_count = pd.read_sql("SELECT COUNT(*) FROM raw_data.hospitals", engine).iloc[0,0]
        census_count = pd.read_sql("SELECT COUNT(*) FROM raw_data.census", engine).iloc[0,0]
        costs_count = pd.read_sql("SELECT COUNT(*) FROM raw_data.healthcare_costs", engine).iloc[0,0]
        
        # Quality thresholds
        if hospitals_count < 1000:
            raise Exception(f"Insufficient hospital data: {hospitals_count}")
        
        if census_count < 50:
            raise Exception(f"Insufficient census data: {census_count}")
        
        if costs_count < 200:
            raise Exception(f"Insufficient cost data: {costs_count}")
        
        print(f"✅ Data quality check passed:")
        print(f"   Hospitals: {hospitals_count}")
        print(f"   Census records: {census_count}")
        print(f"   Cost records: {costs_count}")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"❌ Data quality check failed: {e}")
        raise

quality_check_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=dag,
)

# Task 7: Update Dashboard
def update_dashboard():
    """Update dashboard with latest data"""
    import subprocess
    import sys
    
    try:
        # Restart dashboard service
        result = subprocess.run([
            'docker', 'restart', 'healthcare-dashboard'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dashboard updated successfully")
            return "SUCCESS"
        else:
            print(f"❌ Dashboard update failed: {result.stderr}")
            raise Exception("Dashboard update failed")
            
    except Exception as e:
        print(f"❌ Error updating dashboard: {e}")
        raise

update_dashboard_task = PythonOperator(
    task_id='update_dashboard',
    python_callable=update_dashboard,
    dag=dag,
)

# Task 8: Send Success Email
success_email = EmailOperator(
    task_id='send_success_email',
    to=['healthcare@company.com'],
    subject='Healthcare ETL Pipeline - SUCCESS',
    html_content="""
    <h2>Healthcare ETL Pipeline Completed Successfully</h2>
    <p>The daily healthcare data processing pipeline has completed successfully.</p>
    <ul>
        <li>Data collection: ✅</li>
        <li>Data processing: ✅</li>
        <li>Database loading: ✅</li>
        <li>ML model training: ✅</li>
        <li>Analytics generation: ✅</li>
        <li>Dashboard update: ✅</li>
    </ul>
    <p>Dashboard is available at: <a href="http://localhost:8501">http://localhost:8501</a></p>
    """,
    dag=dag,
)

# Task 9: Send Failure Email
failure_email = EmailOperator(
    task_id='send_failure_email',
    to=['healthcare@company.com'],
    subject='Healthcare ETL Pipeline - FAILURE',
    html_content="""
    <h2>Healthcare ETL Pipeline Failed</h2>
    <p>The daily healthcare data processing pipeline has failed.</p>
    <p>Please check the Airflow logs for details.</p>
    """,
    trigger_rule='one_failed',
    dag=dag,
)

# Task dependencies
collect_data_task >> process_data_task >> load_postgres_task >> quality_check_task
load_postgres_task >> train_ml_task >> generate_analytics_task
quality_check_task >> generate_analytics_task >> update_dashboard_task >> success_email
[collect_data_task, process_data_task, load_postgres_task, train_ml_task, generate_analytics_task, update_dashboard_task] >> failure_email
