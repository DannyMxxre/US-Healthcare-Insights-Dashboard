# 🏥 US Healthcare Insights Dashboard V1.0

## 🎯 **Project Objective**

**US Healthcare Insights Dashboard** is a comprehensive ETL project for analyzing the US healthcare system using open data sources. The project demonstrates a complete data processing cycle: from collection and transformation to insight visualization.

### **🏆 Key Achievements:**

- **📊 National Analysis:** Data processing across all 50 US states
- **🏥 1,542 Hospitals:** Comprehensive analysis of medical facilities
- **💰 250 Records:** Healthcare cost analysis (2020-2024)
- **🗺️ Geospatial Analysis:** Interactive maps with hospital markers
- **📈 Correlation Analysis:** Identifying relationships between social and medical indicators

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Collect Data**
```bash
# Create national datasets
python3 etl/data_collector.py
```

### **3. Process Data (ETL)**
```bash
# Process and analyze data
python3 etl/data_processor.py
```

### **4. Launch Dashboard**
```bash
# Launch interactive dashboard
streamlit run dashboard/app.py
```

The dashboard will be available at: `http://localhost:8501`

## 📊 **Data Sources**

### **🏥 National Hospital Data**
- **1,542 hospitals** across all 50 states
- Quality and safety ratings
- Bed capacity and facility types
- Geographic coordinates

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
- **Python 3.13** - primary development language
- **Pandas** - data processing and analysis
- **NumPy** - numerical computations
- **Folium** - interactive map creation

### **Frontend & Visualization**
- **Streamlit** - interactive web interface
- **Plotly** - interactive charts and graphs
- **Streamlit-Folium** - map integration with Streamlit

### **Data Management**
- **CSV** - structured data storage
- **JSON** - metadata and insights
- **HTML** - interactive maps

## 📁 **Project Structure**

```
US Healthcare Insights Dashboard/
├── 📊 data/
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── 🔧 etl/
│   ├── data_collector.py       # National data collection
│   └── data_processor.py       # ETL data processing
├── 🎨 dashboard/
│   └── app.py                  # Main dashboard
├── 📋 requirements.txt         # Project dependencies
├── 🚀 run_project.py          # Project runner script
└── 📖 README.md               # Project documentation
```

## 📈 **Key Insights**

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

## 🗺️ **Geospatial Analysis**

### **Interactive US Map**
- **Hospital markers** with detailed information
- **State circle markers** with uninsured rates
- **Heatmap** of medical facility density
- **Interactive popups** with data

## 📊 **Analysis Tools**

### **State Comparison**
- Interactive state selection for comparison
- Multiple metrics for analysis
- Visualization of regional differences

### **Correlation Analysis**
- Relationship between income and life expectancy
- Correlation between education and healthcare coverage
- Impact of poverty on health indicators

## 🚀 **Deployment**

### **Local Deployment**
```bash
# Clone repository
git clone <repository-url>
cd US-Healthcare-Insights-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run project
python3 run_project.py
```

### **Cloud Deployment**
The project is ready for deployment on:
- **Streamlit Cloud** - for dashboard demonstration
- **Heroku** - for full deployment
- **Docker + VPS** - for production environment
- **Render.com** - for automatic deployment

## 📋 **Portfolio Highlights**

### **🔧 Technical Skills**
- **ETL Pipeline Development** - complete data processing cycle
- **Data Visualization** - interactive dashboard creation
- **Geospatial Analysis** - geographic data processing
- **Statistical Analysis** - correlation analysis and insights

### **📊 Analytical Skills**
- **Healthcare Analytics** - medical data analysis
- **Demographic Analysis** - social determinants study
- **Cost Analysis** - economic indicators analysis
- **Quality Metrics** - healthcare quality assessment

### **🎯 Business Value**
- **Data-Driven Insights** - evidence-based solutions
- **Healthcare Optimization** - healthcare system optimization
- **Policy Recommendations** - recommendations for policymakers
- **Public Health Awareness** - health awareness improvement

## 🔮 **Future Enhancements**

### **V2.0 Plans**
- **Real-time Data Integration** - real-time API integration
- **Machine Learning Models** - health trend prediction
- **Advanced Geospatial Features** - 3D maps and clustering
- **Mobile Optimization** - mobile device adaptation

### **Data Expansion**
- **COVID-19 Impact Analysis** - pandemic impact analysis
- **Chronic Disease Tracking** - chronic disease monitoring
- **Mental Health Metrics** - mental health indicators
- **Environmental Health** - environment-health connection

## 📄 **License**

This project is created for demonstrating skills in data analysis and healthcare. Used only for educational and portfolio purposes.

## 👨‍💻 **Author**

**Danny Covellie** - Data Scientist & Healthcare Analytics Specialist

---

**🏥 US Healthcare Insights Dashboard V1.0** - Comprehensive US Healthcare System Analysis
