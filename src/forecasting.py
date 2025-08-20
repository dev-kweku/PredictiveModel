import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_future_forecast(historical_data, model, location_encoder, disaster_encoder, season_encoder, 
                            start_year=2024, end_year=2028):
    """
    Generate disaster forecasts for the next 5 years (2024-2028) for each month and location
    """
    # Create a manual season mapping dictionary
    season_mapping = {
        'Spring': 0,
        'Summer': 1,
        'Fall': 2,
        'Winter': 3
    }
    
    # Get unique locations
    locations = historical_data['location'].unique()
    
    # Create future dates
    future_dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            future_dates.append({
                'year': year,
                'month': month,
                'day': 15,  # Mid-month representation
                'date': pd.to_datetime(f'{year}-{month}-15')
            })
    
    # Create DataFrame for all location-date combinations
    forecast_data = []
    for date_info in future_dates:
        for location in locations:
            record = {
                'location': location,
                'year': date_info['year'],
                'month': date_info['month'],
                'day': date_info['day'],
                'date': date_info['date']
            }
            forecast_data.append(record)
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Add day of week
    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
    
    # Apply feature engineering
    forecast_df['location_encoded'] = location_encoder.transform(forecast_df['location'])
    
    # Create season
    forecast_df['season'] = forecast_df['month'].apply(lambda x: 
        'Spring' if x in [3,4,5] else
        'Summer' if x in [6,7,8] else
        'Fall' if x in [9,10,11] else 'Winter')
    
    # Encode season using manual mapping instead of the fitted encoder
    forecast_df['season_encoded'] = forecast_df['season'].map(season_mapping)
    
    # Calculate location risk from historical data
    location_risk = historical_data.groupby('location').size() / len(historical_data)
    forecast_df['location_risk'] = forecast_df['location'].map(location_risk)
    
    # Calculate disaster frequency from historical data
    disaster_freq = historical_data.groupby('disaster_type').size() / len(historical_data)
    # Use average disaster frequency for all predictions
    avg_disaster_freq = disaster_freq.mean()
    forecast_df['disaster_freq'] = avg_disaster_freq
    
    # Prepare features for prediction
    features = ['year', 'month', 'day', 'day_of_week', 'location_encoded', 
                'season_encoded', 'location_risk', 'disaster_freq']
    
    # Make predictions
    X_forecast = forecast_df[features]
    forecast_df['disaster_encoded'] = model.predict(X_forecast)
    
    # Get prediction probabilities
    proba = model.predict_proba(X_forecast)
    
    # Use model's classes_ attribute to get disaster classes
    disaster_classes = model.classes_
    
    # Add probability columns for each disaster type
    for i, disaster in enumerate(disaster_classes):
        if i < proba.shape[1]:  # Make sure we don't exceed the array bounds
            forecast_df[f'prob_{disaster}'] = proba[:, i]
        else:
            print(f"Warning: Skipping {disaster} due to index out of bounds")
    
    # Convert encoded disaster type back to string using the model's classes_
    # This ensures we get the actual disaster type names
    forecast_df['disaster_type'] = [disaster_classes[i] for i in forecast_df['disaster_encoded']]
    
    # Select relevant columns
    prob_cols = [f'prob_{disaster}' for disaster in disaster_classes if f'prob_{disaster}' in forecast_df.columns]
    result_df = forecast_df[['location', 'year', 'month', 'disaster_type'] + prob_cols]
    
    return result_df

def aggregate_forecast_by_location(forecast_df):
    """
    Aggregate forecast data by location to show most likely disasters
    """
    # Group by location and get most common disaster type
    location_summary = forecast_df.groupby('location')['disaster_type'].apply(
        lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
    ).reset_index(name='most_likely_disaster')
    
    # Calculate disaster type percentages by location
    disaster_pct = forecast_df.groupby(['location', 'disaster_type']).size() / \
                  forecast_df.groupby('location').size() * 100
    disaster_pct = disaster_pct.reset_index(name='percentage')
    
    # Pivot for easier visualization
    disaster_pivot = disaster_pct.pivot(index='location', columns='disaster_type', values='percentage').fillna(0)
    
    return location_summary, disaster_pivot

def save_forecast_results(forecast_df, location_summary, disaster_pivot, output_dir):
    """
    Save forecast results to CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    forecast_df.to_csv(os.path.join(output_dir, 'forecast_results.csv'), index=False)
    location_summary.to_csv(os.path.join(output_dir, 'location_forecast_summary.csv'), index=False)
    disaster_pivot.to_csv(os.path.join(output_dir, 'disaster_percentages.csv'), index=True)
    
    print("Forecast results saved successfully!")