#!/usr/bin/env python3
"""
Very simple test to check if everything works
"""

print("🚀 Starting simple test...")

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
    
    import numpy as np
    print("✅ NumPy imported successfully")
    
    from sklearn.ensemble import RandomForestRegressor
    print("✅ Scikit-learn imported successfully")
    
    # Create simple data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [3, 6, 9, 12, 15]
    })
    
    print(f"✅ Created sample data: {data.shape}")
    
    # Train simple model
    X = data[['feature1', 'feature2']]
    y = data['target']
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Make prediction
    prediction = model.predict([[6, 12]])
    print(f"✅ Model prediction: {prediction[0]:.2f}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
