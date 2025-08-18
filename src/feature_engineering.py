# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_temporal_features(df):
    """Create time-based features"""
    df = df.copy()
    
    # Extract temporal components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Create season feature
    def get_season(month):
        if month in [12, 1, 2]:
            return 'WINTER'
        elif month in [3, 4, 5]:
            return 'SPRING'
        elif month in [6, 7, 8]:
            return 'SUMMER'
        else:
            return 'FALL'
    
    df['Season'] = df['Month'].apply(get_season)
    
    # Weekend indicator
    df['Is_Weekend'] = df['DayOfWeek'] >= 5
    
    return df

def create_historical_features(df):
    """Create historical frequency features"""
    df = df.copy()
    
    # Location history count
    location_counts = df['Location'].value_counts().to_dict()
    df['Location_History_Count'] = df['Location'].map(location_counts)
    
    # Disaster type history count
    disaster_counts = df['Disaster_Type'].value_counts().to_dict()
    df['Disaster_History_Count'] = df['Disaster_Type'].map(disaster_counts)
    
    # Location-disaster combination history
    location_disaster_counts = df.groupby(['Location', 'Disaster_Type']).size().to_dict()
    df['Location_Disaster_History'] = df.apply(
        lambda row: location_disaster_counts.get((row['Location'], row['Disaster_Type']), 0), 
        axis=1
    )
    
    # Create time-based historical features using a different approach
    # First, sort by date
    df = df.sort_values('Date')
    
    # Create a copy with date as index for time-based operations
    df_indexed = df.set_index('Date')
    
    # Initialize columns with zeros
    df['Disasters_Last_30_Days'] = 0
    df['Disasters_Last_90_Days'] = 0
    
    # For each disaster type, calculate rolling counts
    for disaster_type in df['Disaster_Type'].unique():
        # Filter for this disaster type
        mask = df['Disaster_Type'] == disaster_type
        
        # Get dates for this disaster type
        dates = df.loc[mask, 'Date']
        
        # For each date, count disasters in the last 30 and 90 days
        for i, date in enumerate(dates):
            # Calculate 30-day window
            start_date_30 = date - pd.Timedelta(days=30)
            # Count disasters in the 30 days before this date (not including this date)
            count_30 = ((dates >= start_date_30) & (dates < date)).sum()
            df.loc[mask & (df['Date'] == date), 'Disasters_Last_30_Days'] = count_30
            
            # Calculate 90-day window
            start_date_90 = date - pd.Timedelta(days=90)
            # Count disasters in the 90 days before this date (not including this date)
            count_90 = ((dates >= start_date_90) & (dates < date)).sum()
            df.loc[mask & (df['Date'] == date), 'Disasters_Last_90_Days'] = count_90
    
    return df

def encode_categorical_features(df, categorical_cols):
    """Encode categorical features using one-hot encoding"""
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def prepare_features(df, target_col='Disaster_Type'):
    """Prepare features for modeling"""
    # Create temporal features
    df = create_temporal_features(df)
    
    # Create historical features
    df = create_historical_features(df)
    
    # Define feature columns
    feature_cols = [
        'Year', 'Month', 'Quarter', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
        'Season', 'Is_Weekend', 'Severity', 'Location_History_Count',
        'Disaster_History_Count', 'Disasters_Last_30_Days', 
        'Disasters_Last_90_Days', 'Location_Disaster_History'
    ]
    
    # Select features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Encode categorical features
    categorical_features = ['Season']
    X_encoded = encode_categorical_features(X, categorical_features)
    
    return X_encoded, y

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    df = pd.read_csv('../data/processed_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    X, y = prepare_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")