# fix_disaster_mapping.py
import joblib
import os

# Load the models
model = joblib.load('models/disaster_predictor.pkl')
disaster_encoder = joblib.load('models/disaster_encoder.pkl')

print("Model classes:", model.classes_)
print("Model classes length:", len(model.classes_))
print("Disaster encoder classes:", disaster_encoder.classes_)
print("Disaster encoder classes length:", len(disaster_encoder.classes_))

# Create a proper disaster mapping from the disaster encoder classes
# We'll map each integer index in the disaster encoder to the corresponding disaster type
disaster_mapping = {i: disaster_type for i, disaster_type in enumerate(disaster_encoder.classes_)}

print("Proper disaster mapping:", disaster_mapping)

# Save the correct disaster mapping
joblib.dump(disaster_mapping, 'models/disaster_mapping.pkl')

print("Fixed disaster mapping saved successfully!")

# Also update the forecast results if they exist
if os.path.exists('data/processed/forecast_results.csv'):
    import pandas as pd
    
    # Load forecast data
    forecast_df = pd.read_csv('data/processed/forecast_results.csv')
    
    # Convert disaster types using the new mapping
    # Only convert if the value is in the mapping keys
    forecast_df['disaster_type'] = forecast_df['disaster_type'].apply(
        lambda x: disaster_mapping.get(x, x) if pd.notna(x) and x in disaster_mapping else x
    )
    
    # Save updated forecast data
    forecast_df.to_csv('data/processed/forecast_results.csv', index=False)
    
    print("Updated forecast results with correct disaster types!")
    
    # Update location summary
    location_summary = pd.read_csv('data/processed/location_forecast_summary.csv')
    location_summary['most_likely_disaster'] = location_summary['most_likely_disaster'].apply(
        lambda x: disaster_mapping.get(x, x) if pd.notna(x) and x in disaster_mapping else x
    )
    location_summary.to_csv('data/processed/location_forecast_summary.csv', index=False)
    
    print("Updated location summary with correct disaster types!")