-- US Healthcare Insights Dashboard Database Schema
-- PostgreSQL Database Design

-- Create database
CREATE DATABASE healthcare_insights;

-- Connect to database
\c healthcare_insights;

-- Enable PostGIS for geospatial data
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS processed_data;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Raw data tables
CREATE TABLE raw_data.hospitals (
    id SERIAL PRIMARY KEY,
    hospital_name VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(2) NOT NULL,
    zip_code VARCHAR(10),
    hospital_type VARCHAR(50),
    beds INTEGER,
    rating DECIMAL(3,1),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    medicare_rating INTEGER,
    safety_rating INTEGER,
    patient_satisfaction DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE raw_data.census (
    id SERIAL PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    population INTEGER,
    median_income INTEGER,
    poverty_rate DECIMAL(5,4),
    unemployment_rate DECIMAL(5,4),
    education_bachelors DECIMAL(5,4),
    healthcare_coverage DECIMAL(5,4),
    uninsured_rate DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE raw_data.healthcare_costs (
    id SERIAL PRIMARY KEY,
    state VARCHAR(2) NOT NULL,
    year INTEGER NOT NULL,
    avg_premium INTEGER,
    avg_deductible INTEGER,
    out_of_pocket_max INTEGER,
    medicare_spending_per_capita INTEGER,
    medicaid_enrollment_rate DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE raw_data.quality_metrics (
    id SERIAL PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    life_expectancy DECIMAL(4,1),
    infant_mortality_rate DECIMAL(4,1),
    preventable_deaths_per_100k INTEGER,
    hospital_readmission_rate DECIMAL(5,4),
    patient_safety_score INTEGER,
    access_to_care_score DECIMAL(5,2),
    healthcare_affordability_score DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processed data tables
CREATE TABLE processed_data.hospital_summary (
    id SERIAL PRIMARY KEY,
    state VARCHAR(2) NOT NULL,
    total_hospitals INTEGER,
    total_beds INTEGER,
    avg_rating DECIMAL(3,2),
    avg_medicare_rating DECIMAL(3,2),
    avg_safety_rating DECIMAL(3,2),
    avg_patient_satisfaction DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE processed_data.cost_trends (
    id SERIAL PRIMARY KEY,
    state VARCHAR(2) NOT NULL,
    avg_premium DECIMAL(10,2),
    premium_std DECIMAL(10,2),
    avg_deductible DECIMAL(10,2),
    deductible_std DECIMAL(10,2),
    avg_medicare_spending INTEGER,
    avg_medicaid_enrollment DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analytics tables
CREATE TABLE analytics.correlations (
    id SERIAL PRIMARY KEY,
    correlation_type VARCHAR(50) NOT NULL,
    correlation_value DECIMAL(5,4),
    p_value DECIMAL(10,8),
    significance_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE analytics.insights (
    id SERIAL PRIMARY KEY,
    insight_type VARCHAR(50) NOT NULL,
    insight_title VARCHAR(255) NOT NULL,
    insight_description TEXT,
    metric_value DECIMAL(10,4),
    metric_unit VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Geospatial table with PostGIS
CREATE TABLE analytics.hospital_locations (
    id SERIAL PRIMARY KEY,
    hospital_id INTEGER REFERENCES raw_data.hospitals(id),
    geom GEOMETRY(POINT, 4326),
    state VARCHAR(2),
    hospital_name VARCHAR(255),
    rating DECIMAL(3,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_hospitals_state ON raw_data.hospitals(state);
CREATE INDEX idx_hospitals_rating ON raw_data.hospitals(rating);
CREATE INDEX idx_census_state ON raw_data.census(state);
CREATE INDEX idx_costs_state_year ON raw_data.healthcare_costs(state, year);
CREATE INDEX idx_quality_state ON raw_data.quality_metrics(state);
CREATE INDEX idx_hospital_locations_geom ON analytics.hospital_locations USING GIST(geom);

-- Views for common queries
CREATE VIEW analytics.state_healthcare_summary AS
SELECT 
    h.state,
    COUNT(h.id) as hospital_count,
    AVG(h.rating) as avg_hospital_rating,
    AVG(h.beds) as avg_beds,
    c.population,
    c.median_income,
    c.poverty_rate,
    c.uninsured_rate,
    q.life_expectancy,
    q.infant_mortality_rate
FROM raw_data.hospitals h
LEFT JOIN raw_data.census c ON h.state = c.state
LEFT JOIN raw_data.quality_metrics q ON c.state = q.state
GROUP BY h.state, c.population, c.median_income, c.poverty_rate, c.uninsured_rate, q.life_expectancy, q.infant_mortality_rate;

-- Functions for data processing
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_hospitals_updated_at BEFORE UPDATE ON raw_data.hospitals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Stored procedure for data aggregation
CREATE OR REPLACE FUNCTION aggregate_state_data()
RETURNS VOID AS $$
BEGIN
    -- Clear existing summary data
    DELETE FROM processed_data.hospital_summary;
    
    -- Insert aggregated hospital data
    INSERT INTO processed_data.hospital_summary (
        state, total_hospitals, total_beds, avg_rating, 
        avg_medicare_rating, avg_safety_rating, avg_patient_satisfaction
    )
    SELECT 
        state,
        COUNT(*) as total_hospitals,
        SUM(beds) as total_beds,
        AVG(rating) as avg_rating,
        AVG(medicare_rating) as avg_medicare_rating,
        AVG(safety_rating) as avg_safety_rating,
        AVG(patient_satisfaction) as avg_patient_satisfaction
    FROM raw_data.hospitals
    GROUP BY state;
    
    RAISE NOTICE 'State data aggregation completed';
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE healthcare_insights TO healthcare_user;
GRANT ALL PRIVILEGES ON SCHEMA raw_data TO healthcare_user;
GRANT ALL PRIVILEGES ON SCHEMA processed_data TO healthcare_user;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO healthcare_user;
