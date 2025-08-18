import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from feature_engineering import create_temporal_features, create_historical_features, prepare_features

def test_create_temporal_features():
    # Create test data
    dates = pd.to_datetime(['2023-01-15', '2023-06-15', '2023-12-15'])
    df = pd.DataFrame({'Date': dates})
    
    # Create temporal features
    df_temporal = create_temporal_features(df)
    
    # Assertions
    assert 'Year' in df_temporal.columns
    assert 'Month' in df_temporal.columns
    assert 'Season' in df_temporal.columns
    assert 'Is_Weekend' in df_temporal.columns
    assert df_temporal['Season'].iloc[0] == 'WINTER'
    assert df_temporal['Season'].iloc[1] == 'SUMMER'
    assert df_temporal['Is_Weekend'].iloc[0] == False

def test_create_historical_features():
    # Create test data
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'Disaster_Type': ['FLOOD', 'FLOOD', 'STORM'],
        'Location': ['ACCRA', 'ACCRA', 'KUMASI'],
        'Severity': [1, 2, 1]
    }
    df = pd.DataFrame(data)
    
    # Create historical features
    df_historical = create_historical_features(df)
    
    # Assertions
    assert 'Location_History_Count' in df_historical.columns
    assert 'Disaster_History_Count' in df_historical.columns
    assert 'Location_Disaster_History' in df_historical.columns
    assert df_historical['Location_History_Count'].iloc[0] == 2
    assert df_historical['Disaster_History_Count'].iloc[0] == 2
    assert df_historical['Location_Disaster_History'].iloc[0] == 2

def test_prepare_features():
    # Create test data
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-06-15']),
        'Disaster_Type': ['FLOOD', 'STORM'],
        'Location': ['ACCRA', 'KUMASI'],
        'Severity': [1, 2]
    }
    df = pd.DataFrame(data)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Assertions
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert 'Season_FALL' in X.columns or 'Season_SPRING' in X.columns or 'Season_SUMMER' in X.columns or 'Season_WINTER' in X.columns