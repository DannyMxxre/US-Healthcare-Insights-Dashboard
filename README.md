# 🏥 US Healthcare Insights Dashboard V2.0

## 🎯 **Project Objective**

**US Healthcare Insights Dashboard V2.0** is a comprehensive, enterprise-grade data engineering project for analyzing the US healthcare system. This project demonstrates advanced data engineering skills including ETL pipelines, machine learning, real-time APIs, containerization, and automated workflows.

### **🏆 Key Achievements:**

- **📊 National Analysis:** Data processing across all 50 US states
- **🏥 1,542 Hospitals:** Comprehensive analysis of medical facilities
- **💰 250 Records:** Healthcare cost analysis (2020-2024)
- **🗺️ Geospatial Analysis:** Interactive maps with hospital markers
- **📈 Correlation Analysis:** Identifying relationships between social and medical indicators
- **🤖 Machine Learning:** Predictive models for healthcare outcomes
- **🗄️ PostgreSQL Database:** Production-ready data storage with PostGIS
- **⚡ Apache Airflow:** Automated ETL pipelines with scheduling
- **🐳 Docker Containerization:** Full application containerization
- **🔌 REST API:** FastAPI-based API with authentication and monitoring
- **📊 Monitoring:** Prometheus and Grafana for observability

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Start with Docker (Recommended)**
```bash
# Start all services
docker-compose up -d

# Access services:
# Dashboard: http://localhost:8501
# Airflow: http://localhost:8080
# API: http://localhost:8000
# Grafana: http://localhost:3000
```

### **3. Manual Setup**
```bash
# Collect data
python3 etl/data_collector.py

# Process data
python3 etl/data_processor.py

# Train ML models
python3 ml/models.py

# Start dashboard
streamlit run dashboard/app.py

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ETL Pipeline  │    │   PostgreSQL    │
│                 │    │                 │    │   Database      │
│ • data.gov      │───▶│ • Airflow DAGs  │───▶│ • Raw Data      │
│ • CDC API       │    │ • Data Quality  │    │ • Processed     │
│ • CMS API       │    │ • Validation   │    │ • Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Models     │    │   REST API       │    │   Dashboard     │
│                 │    │                 │    │                 │
│ • Predictions  │◀───│ • FastAPI        │◀───│ • Streamlit     │
│ • Clustering   │    │ • Authentication │    │ • Interactive   │
│ • Analytics    │    │ • Monitoring     │    │ • Real-time     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 **Data Sources**

### **🏥 National Hospital Data**
- **1,542 hospitals** across all 50 states
- Quality and safety ratings
- Bed capacity and facility types
- Geographic coordinates with PostGIS

### **👥 Demographic Data**
- Population by state
- Median income and poverty levels
- Education and unemployment rates
- Healthcare insurance coverage

### **💰 Healthcare Cost Data**
- Average insurance premiums (2020-2024)
- Deductibles and out-of-pocket maximums
- Medicare spending per capita
- Medicaid enrollment rates

### **🏆 Healthcare Quality Metrics**
- Life expectancy
- Infant mortality rates
- Preventable deaths
- Hospital readmission rates

## 🛠️ **Technology Stack**

### **Backend & Data Processing**
- **Python 3.11** - primary development language
- **Pandas & NumPy** - data processing and analysis
- **PostgreSQL + PostGIS** - production database with geospatial support
- **SQLAlchemy** - ORM and database management
- **Alembic** - database migrations

### **Machine Learning**
- **Scikit-learn** - traditional ML models
- **TensorFlow & PyTorch** - deep learning capabilities
- **XGBoost & LightGBM** - gradient boosting
- **Joblib** - model serialization

### **Web Framework & API**
- **FastAPI** - high-performance REST API
- **Streamlit** - interactive dashboard
- **Uvicorn** - ASGI server
- **Pydantic** - data validation

### **Data Pipeline & Orchestration**
- **Apache Airflow** - workflow orchestration
- **Redis** - caching and session management
- **Prometheus** - metrics collection
- **Grafana** - monitoring dashboards

### **Containerization & Deployment**
- **Docker** - application containerization
- **Docker Compose** - multi-service orchestration
- **Nginx** - reverse proxy and load balancing

### **Monitoring & Observability**
- **Prometheus** - metrics collection
- **Grafana** - visualization and alerting
- **Structlog** - structured logging
- **Health checks** - service monitoring

## 📁 **Project Structure**

```
US Healthcare Insights Dashboard/
├── 📊 data/
│   ├── raw/                    # Raw data storage
│   └── processed/              # Processed data
├── 🔧 etl/
│   ├── data_collector.py       # Data collection
│   └── data_processor.py       # ETL processing
├── 🤖 ml/
│   ├── models.py               # ML model training
│   └── saved_models/           # Trained models
├── 🎨 dashboard/
│   └── app.py                  # Streamlit dashboard
├── 🔌 api/
│   └── main.py                 # FastAPI REST API
├── ⚡ airflow/
│   └── dags/                   # Airflow DAGs
├── 🗄️ database/
│   └── schema.sql              # PostgreSQL schema
├── 🐳 docker/
│   ├── docker-compose.yml      # Multi-service setup
│   └── Dockerfile.dashboard    # Dashboard container
├── 📊 monitoring/
│   ├── prometheus.yml          # Prometheus config
│   └── grafana/                # Grafana dashboards
├── 📋 requirements.txt         # Python dependencies
├── 🚀 run_project.py          # Project runner
└── 📖 README.md               # Documentation
```

## 🤖 **Machine Learning Models**

### **🏥 Hospital Rating Predictor**
- **Algorithm:** Random Forest Regressor
- **Features:** Hospital characteristics, demographic data
- **Accuracy:** R² score with cross-validation
- **Use Case:** Predict hospital quality ratings

### **💰 Healthcare Cost Predictor**
- **Algorithm:** Linear Regression
- **Features:** Economic indicators, population data
- **Accuracy:** Cost trend prediction
- **Use Case:** Forecast healthcare costs

### **🗺️ State Clustering**
- **Algorithm:** K-Means Clustering
- **Features:** Healthcare metrics by state
- **Clusters:** 4 distinct healthcare state groups
- **Use Case:** State segmentation analysis

### **🏆 Health Outcome Predictor**
- **Algorithm:** Random Forest Regressor
- **Features:** Social determinants, healthcare access
- **Target:** Life expectancy prediction
- **Use Case:** Health outcome forecasting

## 🔌 **REST API Endpoints**

### **🏥 Hospital Endpoints**
- `GET /api/v1/hospitals` - List hospitals with filtering
- `GET /api/v1/hospitals/{id}` - Get specific hospital
- `POST /api/v1/predict/hospital-rating` - Predict hospital rating

### **🗺️ State Endpoints**
- `GET /api/v1/states` - Get state metrics
- `GET /api/v1/states/{state}` - Get detailed state data

### **📊 Analytics Endpoints**
- `GET /api/v1/analytics/correlations` - Get correlation analysis
- `GET /api/v1/analytics/insights` - Get generated insights

### **📈 Dashboard Endpoints**
- `GET /api/v1/dashboard/summary` - Get summary metrics
- `GET /api/v1/export/hospitals` - Export hospital data

## ⚡ **Apache Airflow DAGs**

### **Daily ETL Pipeline**
- **Schedule:** Daily at 2 AM
- **Tasks:**
  1. Data Collection
  2. Data Processing
  3. Database Loading
  4. ML Model Training
  5. Analytics Generation
  6. Dashboard Update
  7. Email Notifications

### **Data Quality Checks**
- Completeness validation
- Data type verification
- Range validation
- Cross-reference checks

### **Monitoring & Alerting**
- Success/failure notifications
- Performance metrics
- Error tracking
- Retry mechanisms

## 🐳 **Docker Services**

### **Core Services**
- **PostgreSQL + PostGIS** - Database with geospatial support
- **Redis** - Caching and session management
- **Streamlit Dashboard** - Interactive web interface
- **FastAPI** - REST API service

### **Orchestration**
- **Apache Airflow** - Workflow management
- **Nginx** - Reverse proxy and load balancing

### **Monitoring**
- **Prometheus** - Metrics collection
- **Grafana** - Visualization and alerting

## 📊 **Key Insights**

### **🏥 Hospital Analysis**
- **Top 10 states** by hospital count
- **Top 10 states** by quality ratings
- **Hospital distribution** by facility type
- **Geographic accessibility** of medical care

### **💰 Cost Analysis**
- **Insurance premium trends** (2020-2024)
- **Regional variations** in healthcare costs
- **Correlation** between income and medical expenses
- **Accessibility** of medical services

### **👥 Demographic Analysis**
- **Social determinants** of health
- **Correlation** between education and healthcare access
- **Impact of poverty** on health indicators
- **Geographic disparities** in medical care

### **🏆 Healthcare Quality**
- **Life expectancy** by state
- **Infant mortality** and preventable deaths
- **Access to medical care**
- **Healthcare system efficiency**

## 🚀 **Deployment Options**

### **Local Development**
```bash
# Clone and setup
git clone <repository-url>
cd US-Healthcare-Insights-Dashboard
pip install -r requirements.txt

# Run with Docker
docker-compose up -d
```

### **Production Deployment**
- **AWS ECS** - Container orchestration
- **Google Cloud Run** - Serverless containers
- **Azure Container Instances** - Managed containers
- **Kubernetes** - Enterprise orchestration

### **Cloud Services**
- **Streamlit Cloud** - Dashboard hosting
- **Heroku** - Application hosting
- **AWS RDS** - Managed PostgreSQL
- **Google Cloud SQL** - Managed database

## 📋 **Portfolio Highlights**

### **🔧 Technical Skills**
- **Data Engineering** - Complete ETL pipeline development
- **Machine Learning** - Predictive modeling and analytics
- **Database Design** - PostgreSQL schema design and optimization
- **API Development** - RESTful API with authentication
- **Containerization** - Docker and Docker Compose
- **Workflow Orchestration** - Apache Airflow DAGs
- **Monitoring** - Prometheus and Grafana setup

### **📊 Analytical Skills**
- **Healthcare Analytics** - Medical data analysis
- **Demographic Analysis** - Social determinants study
- **Cost Analysis** - Economic indicators analysis
- **Quality Metrics** - Healthcare quality assessment
- **Predictive Modeling** - ML model development

### **🎯 Business Value**
- **Data-Driven Insights** - Evidence-based solutions
- **Healthcare Optimization** - System efficiency analysis
- **Policy Recommendations** - Data-backed recommendations
- **Public Health Awareness** - Health awareness improvement

## 🔮 **Future Enhancements**

### **V3.0 Plans**
- **Real-time Data Streaming** - Apache Kafka integration
- **Advanced ML Models** - Deep learning for predictions
- **Natural Language Processing** - Text analysis of medical reports
- **Blockchain Integration** - Secure health data sharing

### **Advanced Features**
- **Real-time Alerts** - Healthcare system monitoring
- **Predictive Analytics** - Disease outbreak prediction
- **Personalized Insights** - Individual health recommendations
- **Mobile Application** - iOS/Android apps

## 📄 **License**

This project is created for demonstrating advanced data engineering skills. Used for educational and portfolio purposes.

## 👨‍💻 **Author**

**Danny Covellie** - Senior Data Engineer & Healthcare Analytics Specialist

---

**🏥 US Healthcare Insights Dashboard V2.0** - Enterprise-Grade Healthcare Analytics Platform
