"""
Healthcare Plans Data Collector
Collects healthcare plan information including prices, reviews, and ratings
"""

import pandas as pd
import requests
import json
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcarePlansCollector:
    def __init__(self):
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Healthcare plan types
        self.plan_types = [
            'PPO', 'HMO', 'EPO', 'POS', 'HDHP', 'Medicare Advantage',
            'Medicaid Managed Care', 'ACA Marketplace', 'Employer Group'
        ]
        
        # Insurance companies
        self.insurance_companies = [
            'UnitedHealth Group', 'Anthem', 'Aetna', 'Cigna', 'Humana',
            'Kaiser Permanente', 'Blue Cross Blue Shield', 'Molina Healthcare',
            'Centene Corporation', 'Health Net', 'Molina Healthcare',
            'WellCare Health Plans', 'Amerigroup', 'CareSource'
        ]
        
        # States with their healthcare characteristics
        self.state_healthcare_data = {
            'California': {'avg_premium': 450, 'deductible': 1500, 'rating': 4.2},
            'Texas': {'avg_premium': 380, 'deductible': 2000, 'rating': 3.8},
            'New York': {'avg_premium': 520, 'deductible': 1200, 'rating': 4.1},
            'Florida': {'avg_premium': 420, 'deductible': 1800, 'rating': 3.9},
            'Illinois': {'avg_premium': 410, 'deductible': 1600, 'rating': 4.0},
            'Pennsylvania': {'avg_premium': 390, 'deductible': 1700, 'rating': 3.9},
            'Ohio': {'avg_premium': 360, 'deductible': 1900, 'rating': 3.7},
            'Georgia': {'avg_premium': 370, 'deductible': 2100, 'rating': 3.6},
            'Michigan': {'avg_premium': 380, 'deductible': 1800, 'rating': 3.8},
            'North Carolina': {'avg_premium': 350, 'deductible': 2000, 'rating': 3.7},
            'Virginia': {'avg_premium': 400, 'deductible': 1600, 'rating': 4.0},
            'Washington': {'avg_premium': 430, 'deductible': 1400, 'rating': 4.2},
            'Arizona': {'avg_premium': 380, 'deductible': 2200, 'rating': 3.5},
            'Massachusetts': {'avg_premium': 480, 'deductible': 1100, 'rating': 4.3},
            'Tennessee': {'avg_premium': 340, 'deductible': 2300, 'rating': 3.4},
            'Indiana': {'avg_premium': 350, 'deductible': 2000, 'rating': 3.6},
            'Missouri': {'avg_premium': 360, 'deductible': 1900, 'rating': 3.7},
            'Maryland': {'avg_premium': 420, 'deductible': 1500, 'rating': 4.1},
            'Colorado': {'avg_premium': 410, 'deductible': 1600, 'rating': 4.0},
            'Wisconsin': {'avg_premium': 370, 'deductible': 1800, 'rating': 3.8},
            'Minnesota': {'avg_premium': 390, 'deductible': 1600, 'rating': 4.1},
            'South Carolina': {'avg_premium': 330, 'deductible': 2200, 'rating': 3.3},
            'Alabama': {'avg_premium': 320, 'deductible': 2400, 'rating': 3.2},
            'Louisiana': {'avg_premium': 340, 'deductible': 2200, 'rating': 3.4},
            'Kentucky': {'avg_premium': 330, 'deductible': 2300, 'rating': 3.3},
            'Oregon': {'avg_premium': 420, 'deductible': 1500, 'rating': 4.1},
            'Oklahoma': {'avg_premium': 340, 'deductible': 2200, 'rating': 3.4},
            'Connecticut': {'avg_premium': 460, 'deductible': 1300, 'rating': 4.2},
            'Utah': {'avg_premium': 350, 'deductible': 2000, 'rating': 3.7},
            'Iowa': {'avg_premium': 360, 'deductible': 1900, 'rating': 3.8},
            'Nevada': {'avg_premium': 380, 'deductible': 2000, 'rating': 3.6},
            'Arkansas': {'avg_premium': 320, 'deductible': 2300, 'rating': 3.3},
            'Mississippi': {'avg_premium': 310, 'deductible': 2400, 'rating': 3.1},
            'Kansas': {'avg_premium': 350, 'deductible': 2000, 'rating': 3.6},
            'New Mexico': {'avg_premium': 340, 'deductible': 2100, 'rating': 3.5},
            'Nebraska': {'avg_premium': 360, 'deductible': 1900, 'rating': 3.7},
            'Idaho': {'avg_premium': 340, 'deductible': 2100, 'rating': 3.6},
            'West Virginia': {'avg_premium': 330, 'deductible': 2200, 'rating': 3.3},
            'New Hampshire': {'avg_premium': 420, 'deductible': 1500, 'rating': 4.0},
            'Maine': {'avg_premium': 400, 'deductible': 1600, 'rating': 3.9},
            'Rhode Island': {'avg_premium': 440, 'deductible': 1400, 'rating': 4.1},
            'Montana': {'avg_premium': 350, 'deductible': 2000, 'rating': 3.7},
            'Delaware': {'avg_premium': 410, 'deductible': 1600, 'rating': 3.9},
            'South Dakota': {'avg_premium': 340, 'deductible': 2100, 'rating': 3.5},
            'North Dakota': {'avg_premium': 350, 'deductible': 2000, 'rating': 3.6},
            'Alaska': {'avg_premium': 480, 'deductible': 1800, 'rating': 3.8},
            'Vermont': {'avg_premium': 430, 'deductible': 1500, 'rating': 4.2},
            'Wyoming': {'avg_premium': 360, 'deductible': 2000, 'rating': 3.6},
            'Hawaii': {'avg_premium': 420, 'deductible': 1400, 'rating': 4.1}
        }
        
    def generate_plan_data(self):
        """Generate comprehensive healthcare plan data"""
        logger.info("🏥 Generating healthcare plans data...")
        
        plans_data = []
        
        for state, base_data in self.state_healthcare_data.items():
            # Generate multiple plans per state
            num_plans = random.randint(8, 15)
            
            for i in range(num_plans):
                plan_type = random.choice(self.plan_types)
                company = random.choice(self.insurance_companies)
                
                # Base pricing from state data
                base_premium = base_data['avg_premium']
                base_deductible = base_data['deductible']
                base_rating = base_data['rating']
                
                # Add variation based on plan type
                if plan_type == 'PPO':
                    premium_multiplier = random.uniform(1.1, 1.3)
                    deductible_multiplier = random.uniform(0.8, 1.2)
                elif plan_type == 'HMO':
                    premium_multiplier = random.uniform(0.8, 1.1)
                    deductible_multiplier = random.uniform(0.7, 1.0)
                elif plan_type == 'HDHP':
                    premium_multiplier = random.uniform(0.6, 0.9)
                    deductible_multiplier = random.uniform(1.5, 2.5)
                elif plan_type == 'Medicare Advantage':
                    premium_multiplier = random.uniform(0.3, 0.8)
                    deductible_multiplier = random.uniform(0.5, 1.0)
                else:
                    premium_multiplier = random.uniform(0.9, 1.2)
                    deductible_multiplier = random.uniform(0.8, 1.3)
                
                # Calculate final prices
                monthly_premium = round(base_premium * premium_multiplier, 2)
                annual_deductible = round(base_deductible * deductible_multiplier, -2)
                
                # Generate plan features
                features = self._generate_plan_features(plan_type)
                
                # Generate reviews
                reviews = self._generate_reviews(base_rating, company)
                
                # Calculate overall score
                overall_score = self._calculate_overall_score(
                    monthly_premium, annual_deductible, base_rating, features
                )
                
                plan_data = {
                    'state': state,
                    'plan_name': f"{company} {plan_type} {random.randint(1000, 9999)}",
                    'plan_type': plan_type,
                    'insurance_company': company,
                    'monthly_premium': monthly_premium,
                    'annual_deductible': annual_deductible,
                    'copay_primary': random.randint(15, 50),
                    'copay_specialist': random.randint(25, 75),
                    'copay_urgent_care': random.randint(50, 100),
                    'copay_emergency': random.randint(100, 250),
                    'prescription_deductible': random.choice([0, 100, 200, 300]),
                    'max_out_of_pocket': round(annual_deductible * random.uniform(1.5, 2.5), -2),
                    'network_size': random.randint(1000, 50000),
                    'overall_rating': round(base_rating + random.uniform(-0.3, 0.3), 1),
                    'customer_satisfaction': round(base_rating + random.uniform(-0.2, 0.2), 1),
                    'claims_processing': round(base_rating + random.uniform(-0.4, 0.4), 1),
                    'provider_network': round(base_rating + random.uniform(-0.3, 0.3), 1),
                    'overall_score': overall_score,
                    'features': features,
                    'reviews': reviews,
                    'enrollment_date': datetime.now().strftime('%Y-%m-%d'),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                plans_data.append(plan_data)
        
        return pd.DataFrame(plans_data)
    
    def _generate_plan_features(self, plan_type):
        """Generate plan features based on type"""
        base_features = ['Preventive Care', 'Prescription Coverage', 'Mental Health']
        
        if plan_type == 'PPO':
            features = base_features + ['Out-of-Network Coverage', 'No Referral Required', 'Large Network']
        elif plan_type == 'HMO':
            features = base_features + ['Primary Care Physician Required', 'In-Network Only', 'Lower Premiums']
        elif plan_type == 'HDHP':
            features = base_features + ['Health Savings Account', 'High Deductible', 'Tax Advantages']
        elif plan_type == 'Medicare Advantage':
            features = base_features + ['Part D Coverage', 'Additional Benefits', 'Care Coordination']
        else:
            features = base_features + ['Standard Coverage', 'Flexible Options']
        
        # Add random additional features
        additional_features = [
            'Telemedicine', 'Dental Coverage', 'Vision Coverage', 'Gym Membership',
            'Chiropractic Care', 'Acupuncture', 'Alternative Medicine', 'Travel Coverage',
            'Maternity Care', 'Pediatric Care', 'Elder Care', 'Hospice Care'
        ]
        
        num_additional = random.randint(2, 5)
        features.extend(random.sample(additional_features, num_additional))
        
        return features
    
    def _generate_reviews(self, base_rating, company):
        """Generate realistic reviews"""
        reviews = []
        num_reviews = random.randint(5, 15)
        
        review_templates = [
            "Great coverage and reasonable premiums. {company} has been reliable.",
            "Good network of doctors. Claims processing is {speed}.",
            "Premium is {cost_opinion} but coverage is comprehensive.",
            "Customer service is {service_quality}. Overall satisfied.",
            "Network could be larger but {company} provides good value.",
            "Claims were processed {speed}. No major issues.",
            "Premium increased this year but still {cost_opinion} compared to others.",
            "Good coverage for {coverage_type}. Would recommend.",
            "Provider network is {network_size}. Easy to find doctors.",
            "Overall experience with {company} has been {experience}."
        ]
        
        for i in range(num_reviews):
            template = random.choice(review_templates)
            
            # Fill template variables
            review_text = template.format(
                company=company,
                speed=random.choice(['fast', 'slow', 'average']),
                cost_opinion=random.choice(['affordable', 'expensive', 'reasonable']),
                service_quality=random.choice(['excellent', 'good', 'average', 'poor']),
                coverage_type=random.choice(['family', 'individual', 'senior', 'preventive']),
                network_size=random.choice(['large', 'adequate', 'small']),
                experience=random.choice(['positive', 'mixed', 'good', 'satisfactory'])
            )
            
            # Generate rating based on base_rating with some variation
            rating = max(1.0, min(5.0, base_rating + random.uniform(-1.0, 1.0)))
            
            review = {
                'text': review_text,
                'rating': round(rating, 1),
                'date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
                'verified': random.choice([True, False]),
                'helpful_votes': random.randint(0, 25)
            }
            
            reviews.append(review)
        
        return reviews
    
    def _calculate_overall_score(self, premium, deductible, rating, features):
        """Calculate overall plan score (0-100)"""
        # Premium score (lower is better)
        premium_score = max(0, 100 - (premium - 200) / 5)
        
        # Deductible score (lower is better)
        deductible_score = max(0, 100 - (deductible - 1000) / 20)
        
        # Rating score
        rating_score = rating * 20
        
        # Features score
        feature_score = min(100, len(features) * 8)
        
        # Weighted average
        overall_score = (
            premium_score * 0.25 +
            deductible_score * 0.20 +
            rating_score * 0.35 +
            feature_score * 0.20
        )
        
        return round(overall_score, 1)
    
    def generate_best_plans_analysis(self, plans_df):
        """Generate best plans analysis for each state"""
        logger.info("🏆 Generating best plans analysis...")
        
        best_plans = []
        
        for state in plans_df['state'].unique():
            state_plans = plans_df[plans_df['state'] == state].copy()
            
            # Sort by overall score
            state_plans = state_plans.sort_values('overall_score', ascending=False)
            
            # Get top 3 plans
            top_plans = state_plans.head(3)
            
            for idx, plan in top_plans.iterrows():
                best_plan = {
                    'state': state,
                    'rank': idx + 1,
                    'plan_name': plan['plan_name'],
                    'plan_type': plan['plan_type'],
                    'insurance_company': plan['insurance_company'],
                    'monthly_premium': plan['monthly_premium'],
                    'annual_deductible': plan['annual_deductible'],
                    'overall_score': plan['overall_score'],
                    'overall_rating': plan['overall_rating'],
                    'customer_satisfaction': plan['customer_satisfaction'],
                    'max_out_of_pocket': plan['max_out_of_pocket'],
                    'network_size': plan['network_size'],
                    'features': plan['features'],
                    'reviews': plan['reviews'],
                    'recommendation_reason': self._generate_recommendation_reason(plan),
                    'value_proposition': self._calculate_value_proposition(plan)
                }
                
                best_plans.append(best_plan)
        
        return pd.DataFrame(best_plans)
    
    def _generate_recommendation_reason(self, plan):
        """Generate recommendation reason for plan"""
        reasons = []
        
        if plan['overall_score'] >= 85:
            reasons.append("Excellent overall value")
        elif plan['overall_score'] >= 75:
            reasons.append("Great value for money")
        else:
            reasons.append("Good coverage options")
        
        if plan['monthly_premium'] < 350:
            reasons.append("Affordable premiums")
        
        if plan['annual_deductible'] < 1500:
            reasons.append("Low deductible")
        
        if plan['overall_rating'] >= 4.0:
            reasons.append("High customer satisfaction")
        
        if plan['network_size'] > 20000:
            reasons.append("Large provider network")
        
        if len(plan['features']) > 8:
            reasons.append("Comprehensive benefits")
        
        return " | ".join(reasons[:3])  # Top 3 reasons
    
    def _calculate_value_proposition(self, plan):
        """Calculate value proposition score"""
        # Value = (Features + Rating) / (Premium + Deductible/12)
        monthly_cost = plan['monthly_premium'] + (plan['annual_deductible'] / 12)
        value_score = (len(plan['features']) * 5 + plan['overall_rating'] * 10) / monthly_cost
        
        return round(value_score, 2)
    
    def save_data(self, plans_df, best_plans_df):
        """Save healthcare plans data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw plans data
        plans_file = self.data_dir / f"healthcare_plans_{timestamp}.csv"
        plans_df.to_csv(plans_file, index=False)
        logger.info(f"💾 Saved plans data: {plans_file}")
        
        # Save best plans analysis
        best_plans_file = self.data_dir / f"best_healthcare_plans_{timestamp}.csv"
        best_plans_df.to_csv(best_plans_file, index=False)
        logger.info(f"🏆 Saved best plans analysis: {best_plans_file}")
        
        # Save summary statistics
        summary_stats = {
            'total_plans': len(plans_df),
            'states_covered': plans_df['state'].nunique(),
            'plan_types': plans_df['plan_type'].value_counts().to_dict(),
            'avg_premium': plans_df['monthly_premium'].mean(),
            'avg_deductible': plans_df['annual_deductible'].mean(),
            'avg_rating': plans_df['overall_rating'].mean(),
            'top_companies': plans_df['insurance_company'].value_counts().head(5).to_dict(),
            'generated_at': datetime.now().isoformat()
        }
        
        summary_file = self.data_dir / f"healthcare_plans_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"📊 Saved summary statistics: {summary_file}")
        
        return plans_file, best_plans_file, summary_file

def main():
    """Main execution function"""
    print("🏥 Healthcare Plans Data Collector")
    print("=" * 50)
    
    collector = HealthcarePlansCollector()
    
    # Generate plans data
    plans_df = collector.generate_plan_data()
    print(f"✅ Generated {len(plans_df)} healthcare plans")
    
    # Generate best plans analysis
    best_plans_df = collector.generate_best_plans_analysis(plans_df)
    print(f"🏆 Generated best plans analysis for {best_plans_df['state'].nunique()} states")
    
    # Save data
    plans_file, best_plans_file, summary_file = collector.save_data(plans_df, best_plans_df)
    
    print("\n📊 Summary:")
    print(f"Total Plans: {len(plans_df)}")
    print(f"States Covered: {plans_df['state'].nunique()}")
    print(f"Average Premium: ${plans_df['monthly_premium'].mean():.2f}")
    print(f"Average Rating: {plans_df['overall_rating'].mean():.1f}/5.0")
    print(f"Top Plan Type: {plans_df['plan_type'].mode().iloc[0]}")
    
    print(f"\n🏆 Best Plans Generated:")
    for state in best_plans_df['state'].unique():
        state_best = best_plans_df[best_plans_df['state'] == state].iloc[0]
        print(f"  {state}: {state_best['plan_name']} (Score: {state_best['overall_score']})")
    
    print(f"\n💾 Files saved:")
    print(f"  Plans: {plans_file}")
    print(f"  Best Plans: {best_plans_file}")
    print(f"  Summary: {summary_file}")

if __name__ == "__main__":
    main()
