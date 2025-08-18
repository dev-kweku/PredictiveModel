import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_preprocessing import load_data, clean_data, parse_date

def test_parse_date():
    # Test various date formats
    assert parse_date('11/2/19') is not None
    assert parse_date('14/10/2019') is not None
    assert parse_date('8/13/19') is not None
    assert parse_date('22-01-2024') is not None
    assert parse_date('*') is None
    assert parse_date('invalid_date') is None

def test_load_data():
    # Create a temporary raw data file
    with open('temp_raw_data.txt', 'w') as f:
        f.write("2 | \n")
        f.write("3 | 11/2/19 | WINDSTORM | 1 | AGONA NKWANTA | \n")
        f.write("4 | 21/03/19 | WINDSTORM | 1 | MPATASE | \n")
    
    # Test loading
    df = load_data('temp_raw_data.txt')
    assert len(df) == 2
    assert list(df.columns) == ['Date', 'Disaster_Type', 'Severity', 'Location']
    
    # Clean up
    os.remove('temp_raw_data.txt')

def test_clean_data():
    # Create test data
    data = {
        'Date': ['11/2/19', '14/10/2019', '*', 'invalid'],
        'Disaster_Type': ['WINDSTORM', 'FLOOD', 'RAINSTROM', 'FLOODING'],
        'Severity': ['1', '2', '3', 'invalid'],
        'Location': ['AGONA NKWANTA', 'MPATASE', '*', 'ACCRA']
    }
    df = pd.DataFrame(data)
    
    # Clean data
    clean_df = clean_data(df)
    
    # Assertions
    assert clean_df['Disaster_Type'].iloc[2] == 'RAINSTORM'
    assert clean_df['Disaster_Type'].iloc[3] == 'FLOOD'
    assert clean_df['Location'].iloc[2] == 'UNKNOWN'
    assert pd.isna(clean_df['Date'].iloc[3])
    assert clean_df['Severity'].iloc[3] == 1  # Default value