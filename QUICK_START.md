# 🚀 Quick Start Guide - US Healthcare Insights Dashboard

## ⚡ **5-Minute Setup**

### 1️⃣ **Prerequisites Check**
```bash
# Check Python version (3.11+ required)
python3 --version

# Check if pip is installed
pip --version

# Check if git is installed
git --version
```

### 2️⃣ **Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python3 -c "import streamlit, pandas, plotly, folium; print('✅ All packages installed successfully!')"
```

### 3️⃣ **Generate Sample Data**
```bash
# Generate healthcare plans data
python3 etl/healthcare_plans_collector.py

# Run ML models to generate insights
python3 ml/test_advanced_models.py
```

### 4️⃣ **Launch Dashboards**
```bash
# Main Healthcare Analytics Dashboard
streamlit run dashboard/app.py --server.port 8501

# Healthcare Plans Dashboard (in new terminal)
streamlit run dashboard/healthcare_plans_app.py --server.port 8502

# ML Insights Dashboard (in new terminal)
streamlit run dashboard/ml_insights_app.py --server.port 8503
```

### 5️⃣ **Access Your Dashboards**
- **Main Dashboard:** http://localhost:8501
- **Healthcare Plans:** http://localhost:8502
- **ML Insights:** http://localhost:8503

---

## 🐳 **Docker Setup (Alternative)**

### **Quick Docker Launch**
```bash
# Start all services with Docker
docker-compose up -d

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:8000
# Database: localhost:5432
```

### **Testing Environment**
```bash
# Start testing environment
docker-compose -f docker-compose.test.yml up -d
```

---

## 📊 **What You'll See**

### 🏥 **Main Dashboard (Port 8501)**
- **Healthcare Analytics:** Hospital distribution, costs, demographics
- **Interactive Maps:** State-level geospatial analysis
- **Correlation Analysis:** Healthcare metrics relationships
- **State Comparisons:** Multi-state benchmarking

### 💼 **Healthcare Plans Dashboard (Port 8502)**
- **Best Plans:** Top healthcare plans for each state
- **Price Analysis:** Premium and deductible comparisons
- **Customer Reviews:** Real user feedback and ratings
- **Plan Features:** Comprehensive benefit analysis

### 🤖 **ML Insights Dashboard (Port 8503)**
- **Predictive Analytics:** ICU occupancy predictions
- **Anomaly Detection:** Healthcare data anomaly identification
- **Model Performance:** ML model metrics and insights
- **Real-time Predictions:** Live forecasting interface

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Port Already in Use**
```bash
# Find and kill process using port
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run dashboard/app.py --server.port 8504
```

#### **Missing Dependencies**
```bash
# Reinstall specific package
pip install streamlit-folium

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

#### **Data Not Loading**
```bash
# Regenerate data
python3 etl/healthcare_plans_collector.py
python3 ml/test_advanced_models.py

# Check data files exist
ls -la data/processed/
ls -la ml/saved_models/
```

#### **ML Models Not Working**
```bash
# Run simplified ML test
python3 ml/test_advanced_models.py

# Check for errors
python3 test_simple.py
```

### **Performance Optimization**

#### **Enable Caching**
```bash
# Install watchdog for better performance
pip install watchdog

# On macOS, install Xcode command line tools
xcode-select --install
```

#### **Memory Optimization**
```bash
# Set environment variables for better performance
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_SERVER_ENABLE_CORS=false
```

---

## 🧪 **Testing Your Setup**

### **Run All Tests**
```bash
# Test complete project
python3 test_project.py

# Test local environment
python3 test_local.py

# Simple functionality test
python3 test_simple.py
```

### **Verify Data Quality**
```bash
# Check data files
ls -la data/processed/
ls -la ml/saved_models/

# Verify data integrity
python3 -c "
import pandas as pd
import json

# Check healthcare plans
plans = pd.read_csv('data/processed/best_healthcare_plans_2024.csv')
print(f'✅ Healthcare Plans: {len(plans)} records')

# Check ML results
with open('ml/saved_models/simple_ml_results_2024.json', 'r') as f:
    ml_results = json.load(f)
print(f'✅ ML Results: {len(ml_results)} models trained')
"
```

---

## 📈 **Next Steps**

### **Explore Features**
1. **Interactive Maps:** Click on states to see detailed information
2. **Data Filtering:** Use dropdowns to filter by state, plan type, etc.
3. **ML Predictions:** Try the real-time prediction interface
4. **Export Data:** Download charts and data for further analysis

### **Customize Dashboard**
1. **Add New Data:** Modify `etl/` scripts to include additional sources
2. **Create New Visualizations:** Add custom charts in `dashboard/` files
3. **Train New ML Models:** Extend `ml/` scripts with additional algorithms
4. **Deploy to Cloud:** Use Streamlit Cloud or other cloud platforms

### **Advanced Usage**
1. **API Integration:** Use the FastAPI endpoints for custom applications
2. **Database Access:** Connect directly to PostgreSQL for advanced queries
3. **Real-time Alerts:** Configure the alert system for monitoring
4. **Production Deployment:** Set up Docker containers for production use

---

## 🆘 **Need Help?**

### **Documentation**
- **README.md:** Complete project overview
- **PROJECT_STRUCTURE.md:** Detailed file structure
- **FEATURES.md:** Comprehensive feature list
- **NEXT_STEPS.md:** Future development plans

### **Common Commands**
```bash
# Check project status
git status

# View logs
docker-compose logs

# Restart services
docker-compose restart

# Clean up
docker-compose down
```

### **Support**
- Check the troubleshooting section above
- Review error messages in terminal output
- Verify all prerequisites are installed
- Ensure data files are generated correctly

---

**🎉 Congratulations! You're now running the US Healthcare Insights Dashboard!**

**🏥 US Healthcare Insights Dashboard** - Enterprise-Grade Healthcare Analytics Platform
