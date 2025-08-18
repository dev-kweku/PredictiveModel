# src/data_preprocessing.py (updated with correct column mapping)
import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load raw disaster data from Excel file with known column structure"""
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Based on debug output, we know the structure:
        # Column 1 (index 1): Date
        # Column 2 (index 2): Disaster Type
        # Column 3 (index 3): Severity
        # Column 4 (index 4): Location
        
        # Create a new DataFrame with the correct columns
        new_df = pd.DataFrame()
        new_df['Date'] = df.iloc[:, 1]  # Column 1: Date
        new_df['Disaster_Type'] = df.iloc[:, 2]  # Column 2: Disaster Type
        new_df['Severity'] = df.iloc[:, 3]  # Column 3: Severity
        new_df['Location'] = df.iloc[:, 4]  # Column 4: Location
        
        logger.info("Successfully mapped columns from known structure")
        return new_df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def parse_date(date_val):
    """Parse various date formats into datetime object"""
    if pd.isna(date_val) or date_val == '*' or date_val == 'nan' or date_val == '':
        return None
    
    # If it's already a datetime object, return it
    if isinstance(date_val, pd.Timestamp):
        return date_val
    
    date_str = str(date_val).strip()
    
    # Handle special cases
    if date_str == '11/319':
        return pd.to_datetime('11/03/19', format='%d/%m/%y')
    elif date_str == '5/8/202':
        return pd.to_datetime('05/08/2020', format='%d/%m/%Y')
    elif date_str == '31/2/2020':
        # February 31 doesn't exist, use February 28
        return pd.to_datetime('28/02/2020', format='%d/%m/%Y')
    elif date_str == '12/10/2022`':
        return pd.to_datetime('12/10/2022', format='%d/%m/%Y')
    elif date_str == '8+A197+B195+B195:Y196':
        return None  # This appears to be an error in the data
    elif date_str == '15/9/222':
        return pd.to_datetime('15/09/2022', format='%d/%m/%Y')
    elif date_str == '23-30/7/2023':
        return pd.to_datetime('23/07/2023', format='%d/%m/%Y')  # Use start date
    elif date_str == '1':
        return None  # Just a number, not a date
    elif date_str == '21-24/6/2023':
        return pd.to_datetime('21/06/2023', format='%d/%m/%Y')  # Use start date
    elif date_str == '10-21-24/24/06/203':
        return pd.to_datetime('24/06/2023', format='%d/%m/%Y')  # Extract valid part
    
    date_formats = [
        '%d/%m/%y', '%d/%m/%Y',
        '%m/%d/%y', '%m/%d/%Y',
        '%d-%m-%Y', '%Y-%m-%d',
        '%d-%b-%y', '%d-%b-%Y',
        '%d/%b/%y', '%d/%b/%Y',
        '%b %d, %Y', '%d %b %Y',
        '%Y%m%d', '%Y-%m-%d %H:%M:%S'
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    # Try pandas automatic parsing as last resort
    try:
        return pd.to_datetime(date_str)
    except:
        logger.warning(f"Could not parse date: {date_str}")
        return None

def clean_data(df):
    """Clean and preprocess the disaster data"""
    if df is None:
        logger.error("No data to clean")
        return None
        
    logger.info("Cleaning data...")
    
    # Ensure we have the required columns
    required_columns = ['Date', 'Disaster_Type', 'Severity', 'Location']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return None
    
    # Apply date parsing
    df['Date'] = df['Date'].apply(parse_date)
    
    # Log how many dates were successfully parsed
    valid_dates = df['Date'].notna().sum()
    logger.info(f"Successfully parsed {valid_dates} out of {len(df)} dates")
    
    # Standardize disaster types
    df['Disaster_Type'] = df['Disaster_Type'].str.upper().str.strip()
    df['Disaster_Type'] = df['Disaster_Type'].replace({
        'RAINSTROM': 'RAINSTORM',
        'FLOODING': 'FLOOD',
        'MAN-MADE': 'MAN MADE',
        'DOMESTIC FIRE': 'DOMESTIC FIRE',
        'COMMERCIAL FIRE': 'COMMERCIAL FIRE',
        'BUSH FIRE': 'BUSH FIRE',
        'PEST&INSECT INFESTATION': 'PEST INFESTATION',
        'INSECT/PESTICIDE': 'PEST INFESTATION',
        'GAS EXPLOSION': 'EXPLOSION',
        'CHEEMICAL EXPLOSION': 'EXPLOSION',
        'LIGHTENING': 'LIGHTNING',
        'TIDAL WAVE': 'TIDAL WAVES',
        'LANDSLIDE': 'LANDSLIDE',
        'DROWNING': 'DROWNING',
        'BIRD FLU': 'EPIDEMICS',
        'EPIDEMICS (AVIAN FLU)': 'EPIDEMICS',
        'ROAD ACCIDENT': 'ROAD ACCIDENT',
        'BUILDING COLLAPSE': 'BUILDING COLLAPSE',
        'GALAMSEY PIT COLLAPSE': 'BUILDING COLLAPSE',
        'INDUSTRIAL FIRE': 'INDUSTRIAL FIRE'
    })
    
    # Handle severity with robust conversion
    def convert_severity(sev_str):
        try:
            sev = float(sev_str)
            if np.isnan(sev):
                return 1
            return int(sev)
        except:
            return 1  # Default severity
    
    df['Severity'] = df['Severity'].apply(convert_severity)
    
    # Clean location
    df['Location'] = df['Location'].str.upper().str.strip()
    df['Location'] = df['Location'].replace('*', 'UNKNOWN')
    df['Location'] = df['Location'].replace('NAN', 'UNKNOWN')
    
    # Log data before filtering
    logger.info(f"Data before filtering: {len(df)} rows")
    logger.info(f"Missing dates: {df['Date'].isna().sum()}")
    logger.info(f"Missing disaster types: {df['Disaster_Type'].isna().sum()}")
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['Date', 'Disaster_Type'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Cleaned data has {len(df)} valid records")
    
    return df

def save_processed_data(df, output_path):
    """Save processed data to CSV"""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

if __name__ == "__main__":
    # Example usage
    raw_data = load_data('data/Nadmo.new.xlsx')
    if raw_data is not None:
        clean_data = clean_data(raw_data)
        if clean_data is not None:
            save_processed_data(clean_data, 'data/processed_data.csv')