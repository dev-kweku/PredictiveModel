# src/prediction.py (updated with label encoding)
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

def get_season(month):
    """Determine season from month"""
    if month in [12, 1, 2]:
        return 'WINTER'
    elif month in [3, 4, 5]:
        return 'SPRING'
    elif month in [6, 7, 8]:
        return 'SUMMER'
    else:
        return 'FALL'

def predict_disaster(date_str, severity, location, model_path='models/disaster_model.pkl'):
    """
    Predict disaster type for given date, severity, and location
    
    Parameters:
    - date_str (str): Date string in format 'YYYY-MM-DD'
    - severity (int): Severity level (1, 2, or 3)
    - location (str): Location name
    - model_path (str): Path to trained model file
    
    Returns:
    - dict: Prediction results with disaster type and probabilities
    """
    # Load model and preprocessing objects
    model_objects = joblib.load(model_path)
    model = model_objects['model']
    label_encoder = model_objects['label_encoder']
    scaler = model_objects['scaler']
    feature_cols = model_objects['feature_cols']
    categorical_features = model_objects['categorical_features']
    
    # Parse date
    date_obj = pd.to_datetime(date_str)
    
    # Create features
    features = {
        'Year': date_obj.year,
        'Month': date_obj.month,
        'Quarter': (date_obj.month - 1) // 3 + 1,
        'DayOfWeek': date_obj.dayofweek,
        'DayOfYear': date_obj.dayofyear,
        'WeekOfYear': date_obj.isocalendar().week,
        'Season': get_season(date_obj.month),
        'Is_Weekend': date_obj.dayofweek >= 5,
        'Severity': severity,
        'Location_History_Count': 0,  # Default value
        'Disaster_History_Count': 0,  # Default value
        'Disasters_Last_30_Days': 0,  # Default value
        'Disasters_Last_90_Days': 0,  # Default value
        'Location_Disaster_History': 0  # Default value
    }
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    # One-hot encode categorical features
    features_encoded = pd.get_dummies(features_df, columns=categorical_features, drop_first=True)
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in features_encoded.columns and col not in categorical_features:
            features_encoded[col] = 0
    
    # Add missing columns from training
    missing_cols = set(feature_cols) - set(features_encoded.columns)
    for col in missing_cols:
        features_encoded[col] = 0
    
    # Ensure column order matches training
    features_encoded = features_encoded[feature_cols]
    
    # Scale features
    features_scaled = scaler.transform(features_encoded)
    
    # Make prediction (returns encoded labels)
    prediction_encoded = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Decode the prediction
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get class labels and probabilities
    classes_encoded = np.arange(len(label_encoder.classes_))
    classes = label_encoder.inverse_transform(classes_encoded)
    prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
    
    # Sort by probability
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'predicted_disaster': prediction,
        'probabilities': sorted_probs,
        'input_date': date_str,
        'input_severity': severity,
        'input_location': location
    }

if __name__ == "__main__":
    # Example usage
    result = predict_disaster('2023-06-15', 2, 'ACCRA')
    print(f"Predicted disaster: {result['predicted_disaster']}")
    print("\nProbabilities:")
    for disaster, prob in result['probabilities']:
        print(f"{disaster}: {prob:.4f}")