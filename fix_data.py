import pandas as pd
import numpy as np
from datetime import datetime

def clean_disaster_data(file_path, output_path):
    """
    Clean and fix disaster data by converting all text to uppercase, 
    removing duplicates, and standardizing date format
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Convert all string columns to uppercase
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.upper()
    
    # Clean specific columns
    # Remove any leading/trailing whitespace
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    # Handle date column - remove '00:00:00' time format and standardize
    if 'Date' in df.columns:
        df['Date'] = df['Date'].str.strip()
        
        # Remove '00:00:00' time format
        df['Date'] = df['Date'].str.replace(r'\s+00:00:00$', '', regex=True)
        df['Date'] = df['Date'].str.replace(r'^\d{4}-\d{2}-\d{2}\s+00:00:00$', '', regex=True)
        
        # Standardize date formats (this is a simple approach - you might need more complex parsing)
        # Convert various date formats to a standard format
        def standardize_date(date_str):
            if pd.isna(date_str) or date_str == 'NAN' or date_str == '':
                return date_str
            
            # Remove any remaining time components
            date_str = str(date_str).split(' ')[0]
            
            # Try to parse common date formats
            try:
                # Handle DD/MM/YYYY format
                if '/' in date_str and len(date_str.split('/')[0]) <= 2:
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    return date_obj.strftime('%Y-%m-%d')
                
                # Handle YYYY-MM-DD format
                elif '-' in date_str and len(date_str.split('-')[0]) == 4:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    return date_obj.strftime('%Y-%m-%d')
                
                # Handle other formats if needed
                else:
                    return date_str
                    
            except ValueError:
                # Return original if parsing fails
                return date_str
        
        # Apply date standardization
        df['Date'] = df['Date'].apply(standardize_date)
    
    # Handle disaster_type column - clean any inconsistencies
    if 'disaster_type' in df.columns:
        df['disaster_type'] = df['disaster_type'].str.strip()
        # Remove any extra spaces and standardize
        df['disaster_type'] = df['disaster_type'].str.replace(r'\s+', ' ', regex=True)
    
    # Handle location column - clean any inconsistencies
    if 'Location' in df.columns:
        df['Location'] = df['Location'].str.strip()
        # Remove extra quotes if they exist
        df['Location'] = df['Location'].str.replace('"', '').str.replace("'", "")
        # Remove extra spaces
        df['Location'] = df['Location'].str.replace(r'\s+', ' ', regex=True)
    
    # Remove duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    
    print(f"Removed {initial_count - final_count} duplicate rows")
    print(f"Final data shape: {df.shape}")
    
    # Display sample of cleaned data
    print("\nSample of cleaned data:")
    print(df.head(10))
    
    # Display value counts for disaster types
    if 'disaster_type' in df.columns:
        print("\nDisaster type counts:")
        print(df['disaster_type'].value_counts())
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    return df

def analyze_dates(df):
    """Analyze and display information about date formats"""
    if 'Date' in df.columns:
        print("\n" + "="*50)
        print("DATE ANALYSIS")
        print("="*50)
        
        # Count different date formats
        date_formats = df['Date'].apply(lambda x: 'YYYY-MM-DD' if '-' in str(x) and len(str(x).split('-')[0]) == 4 else 
                                    'DD/MM/YYYY' if '/' in str(x) and len(str(x).split('/')[0]) <= 2 else
                                    'OTHER' if pd.notna(x) and str(x) != 'NAN' else 'MISSING')
        
        print("Date format distribution:")
        print(date_formats.value_counts())
        
        print(f"\nUnique dates: {df['Date'].nunique()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Show samples of different date formats
        print("\nSample dates:")
        for date_format in date_formats.unique():
            if date_format != 'MISSING':
                sample_dates = df[date_formats == date_format]['Date'].head(3).tolist()
                print(f"{date_format}: {sample_dates}")

# File paths
input_file = "./data/raw/Nadmo_cleaned_refined.csv"
output_file = "Nadmo_cleaned_final.csv"

# Run the cleaning function
cleaned_df = clean_disaster_data(input_file, output_file)

# Additional analysis
print("\n" + "="*50)
print("DATA QUALITY REPORT")
print("="*50)

# Check for missing values
print("\nMissing values per column:")
print(cleaned_df.isnull().sum())

# Check data types
print("\nData types:")
print(cleaned_df.dtypes)

# Check unique values in each column
print("\nNumber of unique values per column:")
for col in cleaned_df.columns:
    print(f"{col}: {cleaned_df[col].nunique()}")

# Analyze dates
analyze_dates(cleaned_df)

# Display some examples of cleaned dates
if 'Date' in cleaned_df.columns:
    print("\n" + "="*50)
    print("CLEANED DATE EXAMPLES")
    print("="*50)
    date_samples = cleaned_df['Date'].dropna().head(10).tolist()
    for i, date in enumerate(date_samples, 1):
        print(f"Example {i}: {date}")