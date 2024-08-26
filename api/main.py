"""
FastAPI REST API for US Healthcare Insights Dashboard
Production-ready API with authentication, validation, and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import redis
import json
import logging
from datetime import datetime, timedelta
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="US Healthcare Insights API",
    description="REST API for US Healthcare Analytics and Insights",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://healthcare_user:healthcare_password@localhost:5432/healthcare_insights")
engine = create_engine(DATABASE_URL)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')

# Pydantic models
class HospitalData(BaseModel):
    hospital_name: str
    city: str
    state: str
    beds: int
    rating: float
    hospital_type: str

class StateMetrics(BaseModel):
    state: str
    total_hospitals: int
    avg_rating: float
    avg_premium: float
    life_expectancy: float

class PredictionRequest(BaseModel):
    state: str
    population: int
    median_income: int
    poverty_rate: float
    unemployment_rate: float

class PredictionResponse(BaseModel):
    predicted_rating: float
    confidence: float
    factors: List[str]

# Authentication middleware
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    # In production, verify against database or external service
    if token != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return token

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return JSONResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Hospital endpoints
@app.get("/api/v1/hospitals", response_model=List[HospitalData])
async def get_hospitals(
    state: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    token: str = Depends(verify_token)
):
    """Get hospitals with optional filtering"""
    REQUEST_COUNT.labels(method='GET', endpoint='/api/v1/hospitals').inc()
    
    try:
        # Check cache first
        cache_key = f"hospitals:{state}:{limit}:{offset}"
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        # Build query
        query = "SELECT hospital_name, city, state, beds, rating, hospital_type FROM raw_data.hospitals"
        params = {}
        
        if state:
            query += " WHERE state = :state"
            params['state'] = state
        
        query += " LIMIT :limit OFFSET :offset"
        params['limit'] = limit
        params['offset'] = offset
        
        # Execute query
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            hospitals = [dict(row) for row in result]
        
        # Cache results
        redis_client.setex(cache_key, 3600, json.dumps(hospitals))
        
        return hospitals
        
    except Exception as e:
        logger.error(f"Error fetching hospitals: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/hospitals/{hospital_id}")
async def get_hospital(hospital_id: int, token: str = Depends(verify_token)):
    """Get specific hospital by ID"""
    REQUEST_COUNT.labels(method='GET', endpoint='/api/v1/hospitals/{id}').inc()
    
    try:
        query = """
        SELECT * FROM raw_data.hospitals 
        WHERE id = :hospital_id
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"hospital_id": hospital_id})
            hospital = result.fetchone()
        
        if not hospital:
            raise HTTPException(status_code=404, detail="Hospital not found")
        
        return dict(hospital)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching hospital {hospital_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# State metrics endpoints
@app.get("/api/v1/states", response_model=List[StateMetrics])
async def get_state_metrics(token: str = Depends(verify_token)):
    """Get healthcare metrics by state"""
    REQUEST_COUNT.labels(method='GET', endpoint='/api/v1/states').inc()
    
    try:
        query = """
        SELECT 
            h.state,
            COUNT(h.id) as total_hospitals,
            AVG(h.rating) as avg_rating,
            AVG(c.avg_premium) as avg_premium,
            AVG(q.life_expectancy) as life_expectancy
        FROM raw_data.hospitals h
        LEFT JOIN raw_data.healthcare_costs c ON h.state = c.state
        LEFT JOIN raw_data.quality_metrics q ON h.state = q.state
        GROUP BY h.state
        ORDER BY total_hospitals DESC
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            states = [dict(row) for row in result]
        
        return states
        
    except Exception as e:
        logger.error(f"Error fetching state metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/states/{state}")
async def get_state_detail(state: str, token: str = Depends(verify_token)):
    """Get detailed metrics for specific state"""
    REQUEST_COUNT.labels(method='GET', endpoint='/api/v1/states/{state}').inc()
    
    try:
        # Get comprehensive state data
        query = """
        SELECT 
            c.state,
            c.population,
            c.median_income,
            c.poverty_rate,
            c.unemployment_rate,
            c.healthcare_coverage,
            c.uninsured_rate,
            q.life_expectancy,
            q.infant_mortality_rate,
            q.preventable_deaths_per_100k,
            AVG(h.rating) as avg_hospital_rating,
            COUNT(h.id) as hospital_count,
            AVG(cost.avg_premium) as avg_premium,
            AVG(cost.avg_deductible) as avg_deductible
        FROM raw_data.census c
        LEFT JOIN raw_data.quality_metrics q ON c.state = q.state
        LEFT JOIN raw_data.hospitals h ON c.state = h.state
        LEFT JOIN raw_data.healthcare_costs cost ON c.state = cost.state
        WHERE c.state = :state
        GROUP BY c.state, c.population, c.median_income, c.poverty_rate, 
                 c.unemployment_rate, c.healthcare_coverage, c.uninsured_rate,
                 q.life_expectancy, q.infant_mortality_rate, q.preventable_deaths_per_100k
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"state": state})
            state_data = result.fetchone()
        
        if not state_data:
            raise HTTPException(status_code=404, detail="State not found")
        
        return dict(state_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching state detail for {state}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Analytics endpoints
@app.get("/api/v1/analytics/correlations")
async def get_correlations(token: str = Depends(verify_token)):
    """Get correlation analysis"""
    REQUEST_COUNT.labels(method='GET', endpoint='/api/v1/analytics/correlations').inc()
    
    try:
        query = """
        SELECT correlation_type, correlation_value, p_value, significance_level
        FROM analytics.correlations
        ORDER BY abs(correlation_value) DESC
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            correlations = [dict(row) for row in result]
        
        return correlations
        
    except Exception as e:
        logger.error(f"Error fetching correlations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/analytics/insights")
async def get_insights(token: str = Depends(verify_token)):
    """Get generated insights"""
    REQUEST_COUNT.labels(method='GET', endpoint='/api/v1/analytics/insights').inc()
    
    try:
        query = """
        SELECT insight_type, insight_title, insight_description, metric_value, metric_unit
        FROM analytics.insights
        ORDER BY created_at DESC
        LIMIT 20
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            insights = [dict(row) for row in result]
        
        return insights
        
    except Exception as e:
        logger.error(f"Error fetching insights: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ML Prediction endpoint
@app.post("/api/v1/predict/hospital-rating", response_model=PredictionResponse)
async def predict_hospital_rating(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """Predict hospital rating based on state characteristics"""
    REQUEST_COUNT.labels(method='POST', endpoint='/api/v1/predict/hospital-rating').inc()
    
    try:
        # Load ML model (in production, load from saved model)
        # For demo, use simple prediction
        features = [
            request.population,
            request.median_income,
            request.poverty_rate,
            request.unemployment_rate
        ]
        
        # Simple prediction model (replace with actual ML model)
        predicted_rating = 3.5 + (request.median_income / 10000) * 0.1 - request.poverty_rate * 2
        
        # Ensure rating is within bounds
        predicted_rating = max(1.0, min(5.0, predicted_rating))
        
        # Calculate confidence based on data quality
        confidence = 0.8 - (request.poverty_rate * 0.3)
        confidence = max(0.1, min(1.0, confidence))
        
        # Identify key factors
        factors = []
        if request.median_income > 60000:
            factors.append("High median income")
        if request.poverty_rate < 0.1:
            factors.append("Low poverty rate")
        if request.unemployment_rate < 0.05:
            factors.append("Low unemployment rate")
        
        return PredictionResponse(
            predicted_rating=round(predicted_rating, 2),
            confidence=round(confidence, 2),
            factors=factors
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Data export endpoints
@app.get("/api/v1/export/hospitals")
async def export_hospitals(
    format: str = "csv",
    token: str = Depends(verify_token)
):
    """Export hospital data"""
    REQUEST_COUNT.labels(method='GET', endpoint='/api/v1/export/hospitals').inc()
    
    try:
        query = "SELECT * FROM raw_data.hospitals"
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if format.lower() == "json":
            return JSONResponse(content=df.to_dict(orient='records'))
        elif format.lower() == "csv":
            return JSONResponse(content=df.to_csv(index=False))
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
    except Exception as e:
        logger.error(f"Error exporting hospitals: {e}")
        raise HTTPException(status_code=500, detail="Export failed")

# Dashboard endpoints
@app.get("/api/v1/dashboard/summary")
async def get_dashboard_summary(token: str = Depends(verify_token)):
    """Get dashboard summary metrics"""
    REQUEST_COUNT.labels(method='GET', endpoint='/api/v1/dashboard/summary').inc()
    
    try:
        # Get summary statistics
        queries = {
            "total_hospitals": "SELECT COUNT(*) FROM raw_data.hospitals",
            "total_states": "SELECT COUNT(DISTINCT state) FROM raw_data.hospitals",
            "avg_rating": "SELECT AVG(rating) FROM raw_data.hospitals",
            "total_population": "SELECT SUM(population) FROM raw_data.census",
            "avg_premium": "SELECT AVG(avg_premium) FROM raw_data.healthcare_costs"
        }
        
        summary = {}
        with engine.connect() as conn:
            for key, query in queries.items():
                result = conn.execute(text(query))
                summary[key] = result.scalar()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error fetching dashboard summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
