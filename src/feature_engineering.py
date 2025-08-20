import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features(df):
    """
    Create features for machine learning model
    """
    # Check if required columns exist
    if 'location' not in df.columns:
        raise ValueError("DataFrame must contain 'location' column")
    if 'disaster_type' not in df.columns:
        raise ValueError("DataFrame must contain 'disaster_type' column")
    
    # Location encoding
    location_encoder = LabelEncoder()
    df['location_encoded'] = location_encoder.fit_transform(df['location'])
    
    # Disaster type encoding
    disaster_encoder = LabelEncoder()
    df['disaster_encoded'] = disaster_encoder.fit_transform(df['disaster_type'])
    
    # Season feature
    df['season'] = df['month'].apply(lambda x: 
        'Spring' if x in [3,4,5] else
        'Summer' if x in [6,7,8] else
        'Fall' if x in [9,10,11] else 'Winter')
    
    # Ensure all seasons are represented in the data
    all_seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    existing_seasons = df['season'].unique()
    
    missing_seasons = [s for s in all_seasons if s not in existing_seasons]
    if missing_seasons:
        print(f"Adding dummy rows for missing seasons: {missing_seasons}")
        for season in missing_seasons:
            # Create a dummy row for each missing season
            dummy_row = df.iloc[0].copy()
            dummy_row['season'] = season
            # Set month to a value corresponding to the season
            if season == 'Spring':
                dummy_row['month'] = 4
            elif season == 'Summer':
                dummy_row['month'] = 7
            elif season == 'Fall':
                dummy_row['month'] = 10
            else:  # Winter
                dummy_row['month'] = 1
            
            # Add the dummy row
            df = pd.concat([df, dummy_row.to_frame().T], ignore_index=True)
    
    # Season encoding
    season_encoder = LabelEncoder()
    df['season_encoded'] = season_encoder.fit_transform(df['season'])
    
    # Location disaster frequency (historical risk)
    location_risk = df.groupby('location').size() / len(df)
    df['location_risk'] = df['location'].map(location_risk)
    
    # Disaster type frequency
    disaster_freq = df.groupby('disaster_type').size() / len(df)
    df['disaster_freq'] = df['disaster_type'].map(disaster_freq)
    
    return df, location_encoder, disaster_encoder, season_encoder