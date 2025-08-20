import pandas as pd
import numpy as np
from datetime import datetime
import os

def preprocess_data(input_path, output_path):
    """
    Load and preprocess the disaster data
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Check if the first row contains column names
    if df.iloc[0, 0] == '../data/raw/Nadmo_cleaned_refined.csv':
        # The first row is not data but contains file info, skip it
        df = df.iloc[1:].reset_index(drop=True)
        print("Removed first row with file info")
    
    # Set column names based on the actual structure
    if len(df.columns) == 6:
        df.columns = ['id', 'date', 'disaster_type', 'value1', 'location', 'value2']
    elif len(df.columns) == 5:
        df.columns = ['id', 'date', 'disaster_type', 'location', 'value1']
    else:
        print(f"Unexpected number of columns: {len(df.columns)}")
    
    # Convert date to datetime
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT
    
    df['date'] = df['date'].apply(parse_date)
    
    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Standardize disaster types
    df['disaster_type'] = df['disaster_type'].str.upper().str.strip()
    
    # Create a comprehensive mapping dictionary
    disaster_mapping = {
        'RAINSTORM': 'RAIN_STORM',
        'RAINSTROM': 'RAIN_STORM',
        'FLOODING': 'FLOOD',
        'WINDSTORM': 'WIND_STORM',
        'DOMESTIC FIRE': 'FIRE',
        'COMMERCIAL FIRE': 'FIRE',
        'BUSH FIRE': 'FIRE',
        'MAN MADE': 'MAN_MADE',
        'MAN-MADE': 'MAN_MADE',
        'TIDAL WAVES': 'TIDAL_WAVE',
        'TIDAL WAVE': 'TIDAL_WAVE',
        'PEST&INSECT INFESTATION': 'PEST_INFESTATION',
        'INSECT/PESTICIDE': 'PEST_INFESTATION',
        'GAS EXPLOSION': 'EXPLOSION',
        'CHEEMICAL EXPLOSION': 'EXPLOSION',
        'ROAD ACCIDENT': 'ACCIDENT',
        'BUILDING COLLAPSE': 'COLLAPSE',
        'LANDSLIDE': 'LANDSLIDE',
        'EPIDEMICS (AVIAN FLU)': 'EPIDEMIC',
        'BIRD FLU': 'EPIDEMIC',
        'DROWNING': 'DROWNING',
        'GALAMSEY PIT COLLAPSE': 'COLLAPSE',
        'LIGHTENING': 'LIGHTNING',
        'RAINSTROM': 'RAIN_STORM',
        'FLOODING': 'FLOOD',
        'MAN MADE ': 'MAN_MADE',
        'DOMESTIC FIRE  ': 'FIRE',
        'RAINSTORM  ': 'RAIN_STORM',
        'FLOOD  ': 'FLOOD',
        'WINDSTORM  ': 'WIND_STORM',
        'TIDAL WAVES': 'TIDAL_WAVE',
        'PEST&INSECT INFESTATION': 'PEST_INFESTATION',
        'INSECT/PESTICIDE': 'PEST_INFESTATION',
        'GAS EXPLOSION': 'EXPLOSION',
        'CHEEMICAL EXPLOSION': 'EXPLOSION',
        'ROAD ACCIDENT': 'ACCIDENT',
        'BUILDING COLLAPSE': 'COLLAPSE',
        'LANDSLIDE': 'LANDSLIDE',
        'EPIDEMICS (AVIAN FLU)': 'EPIDEMIC',
        'BIRD FLU': 'EPIDEMIC',
        'DROWNING': 'DROWNING',
        'GALAMSEY PIT COLLAPSE': 'COLLAPSE',
        'drown': 'DROWNING',
        'domestic fire': 'FIRE',
        'Rainstorm': 'RAIN_STORM',
        'Rainstorm  ': 'RAIN_STORM',
        'Flood': 'FLOOD',
        'Flood  ': 'FLOOD',
        'domestic fire': 'FIRE',
        'MAN MADE(building collapse)': 'COLLAPSE',
        'MAN MADE(DROWNING)': 'DROWNING',
        'flood': 'FLOOD',
        'flooding': 'FLOOD',
        'domestic fire': 'FIRE',
        'rainstorm': 'RAIN_STORM',
        'windstorm': 'WIND_STORM',
        'tidal wave': 'TIDAL_WAVE',
        'man made': 'MAN_MADE',
        'building collapse': 'COLLAPSE',
        'landslide': 'LANDSLIDE',
        'epidemics (avian flu)': 'EPIDEMIC',
        'bird flu': 'EPIDEMIC',
        'drowning': 'DROWNING',
        'galamsey pit collapse': 'COLLAPSE',
        'road accident': 'ACCIDENT',
        'chemical explosion': 'EXPLOSION',
        'gas explosion': 'EXPLOSION',
        'pest&insect infestation': 'PEST_INFESTATION',
        'insect/pesticide': 'PEST_INFESTATION',
        'pest infestation': 'PEST_INFESTATION',
        'lightening': 'LIGHTNING',
        'bush fire': 'FIRE',
        'commercial fire': 'FIRE',
        'tidal waves': 'TIDAL_WAVE',
        'man-made': 'MAN_MADE',
        'man made': 'MAN_MADE'
    }
    
    # Apply the mapping
    df['disaster_type'] = df['disaster_type'].replace(disaster_mapping)
    
    # Handle missing locations
    df['location'] = df['location'].fillna('UNKNOWN')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Data cleaning complete and saved to {output_path}")
    
    return df