# inspect_excel.py
import pandas as pd
import sys

def inspect_excel(file_path):
    """Inspect the structure of an Excel file"""
    try:
        # Read with headers
        df_headers = pd.read_excel(file_path)
        print("\n=== Reading with headers ===")
        print(f"Columns: {list(df_headers.columns)}")
        print(f"Shape: {df_headers.shape}")
        print("\nFirst 5 rows:")
        print(df_headers.head())
        
        # Read without headers
        df_no_headers = pd.read_excel(file_path, header=None)
        print("\n=== Reading without headers ===")
        print(f"Shape: {df_no_headers.shape}")
        print("\nFirst 10 rows:")
        print(df_no_headers.head(10))
        
        # Show data types
        print("\n=== Data types (with headers) ===")
        print(df_headers.dtypes)
        
        # Show sample of each column
        print("\n=== Sample of each column (first 5 values) ===")
        for col in df_headers.columns:
            print(f"\nColumn '{col}':")
            print(df_headers[col].head(5).tolist())
            
    except Exception as e:
        print(f"Error inspecting file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_excel(sys.argv[1])
    else:
        print("Please provide the Excel file path as an argument")