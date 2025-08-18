# src/utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Using categorical units to plot a list of strings")

def plot_disaster_distribution(df, save_path=None):
    """Plot distribution of disaster types"""
    plt.figure(figsize=(12, 6))
    disaster_counts = df['Disaster_Type'].value_counts()
    sns.barplot(x=disaster_counts.values, y=disaster_counts.index)
    plt.title('Disaster Type Distribution')
    plt.xlabel('Count')
    plt.ylabel('Disaster Type')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_temporal_patterns(df, save_path=None):
    """Plot temporal patterns in disaster data"""
    # Yearly distribution
    df['Year'] = df['Date'].dt.year
    yearly_disasters = df.groupby(['Year', 'Disaster_Type']).size().unstack().fillna(0)
    
    plt.figure(figsize=(14, 8))
    yearly_disasters.plot(kind='bar', stacked=True)
    plt.title('Yearly Disaster Distribution')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_yearly.png'))
    else:
        plt.show()
    
    # Monthly distribution
    df['Month'] = df['Date'].dt.month
    monthly_disasters = df.groupby(['Month', 'Disaster_Type']).size().unstack().fillna(0)
    
    plt.figure(figsize=(14, 8))
    monthly_disasters.plot(kind='bar', stacked=True)
    plt.title('Monthly Disaster Distribution')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.xticks(np.arange(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_monthly.png'))
    else:
        plt.show()

def plot_location_distribution(df, top_n=20, save_path=None):
    """Plot distribution of disasters by location"""
    top_locations = df['Location'].value_counts().head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_locations.values, y=top_locations.index)
    plt.title(f'Top {top_n} Disaster Locations')
    plt.xlabel('Count')
    plt.ylabel('Location')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_severity_distribution(df, save_path=None):
    """Plot distribution of disaster severity"""
    severity_counts = df['Severity'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=severity_counts.index, y=severity_counts.values)
    plt.title('Disaster Severity Distribution')
    plt.xlabel('Severity Level')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2], ['Low', 'Medium', 'High'])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def generate_summary_report(df, save_path=None):
    """Generate a summary report of the disaster data"""
    report = f"""
    Disaster Data Summary Report
    ============================
    
    Data Overview:
    - Total records: {len(df)}
    - Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
    - Unique disaster types: {df['Disaster_Type'].nunique()}
    - Unique locations: {df['Location'].nunique()}
    
    Disaster Type Distribution:
    {df['Disaster_Type'].value_counts().to_string()}
    
    Top 5 Locations:
    {df['Location'].value_counts().head(5).to_string()}
    
    Severity Distribution:
    {df['Severity'].value_counts().sort_index().to_string()}
    
    Yearly Distribution:
    {df['Date'].dt.year.value_counts().sort_index().to_string()}
    """
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    else:
        print(report)
    
    return report

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('data/processed_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Generate plots
    plot_disaster_distribution(df, 'reports/disaster_distribution.png')
    plot_temporal_patterns(df, 'reports/temporal_patterns.png')
    plot_location_distribution(df, save_path='reports/location_distribution.png')
    plot_severity_distribution(df, save_path='reports/severity_distribution.png')
    
    # Generate summary report
    generate_summary_report(df, 'reports/data_summary.txt')