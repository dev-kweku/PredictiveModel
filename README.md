

# Ghana Natural Disaster Analysis and Forecasting Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Objectives](#project-objectives)
3. [File Structure](#file-structure)
4. [Setup Instructions](#setup-instructions)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Machine Learning Model](#machine-learning-model)
7. [Forecasting System](#forecasting-system)
8. [Dashboard Features](#dashboard-features)
9. [How to Run the Application](#how-to-run-the-application)
10. [Troubleshooting](#troubleshooting)
11. [Future Enhancements](#future-enhancements)
12. [Conclusion](#conclusion)

---

## Project Overview

The Ghana Natural Disaster Analysis and Forecasting Project is a comprehensive system designed to analyze historical disaster data in Ghana, identify patterns, predict future occurrences, and provide a 5-year forecast (2024-2028) for various disaster types. The project leverages machine learning techniques to build predictive models and provides an interactive dashboard for data visualization and prediction.

### Key Features
- Historical disaster analysis with visualizations
- 5-year disaster forecasting (2024-2028)
- Interactive prediction tool for specific locations and dates
- Comprehensive dashboard with multiple analysis views
- Probability-based predictions for all disaster types

### Technologies Used
- **Programming Language**: Python
- **Machine Learning**: Scikit-learn (Random Forest Classifier)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Data Storage**: CSV files

---

## Project Objectives

1. **Analyze Historical Disaster Data**: Examine patterns and trends in disaster occurrences across Ghana.
2. **Build Predictive Models**: Develop machine learning models to predict disaster types based on location and temporal features.
3. **Generate 5-Year Forecasts**: Create monthly disaster predictions for all locations in Ghana from 2024 to 2028.
4. **Develop Interactive Dashboard**: Provide a user-friendly interface for exploring historical data, viewing forecasts, and making predictions.
5. **Identify High-Risk Areas**: Determine locations most susceptible to specific disaster types for better resource allocation.

---

## File Structure

```
disaster_project/
│
├── data/
│   ├── raw/
│   │   └── Nadmo_cleaned_refined.csv
│   └── processed/
│       ├── cleaned_disaster_data.csv
│       ├── forecast_results.csv
│       ├── location_forecast_summary.csv
│       └── disaster_percentages.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_forecasting.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── forecasting.py
│
├── app/
│   ├── streamlit_dashboard.py
│   └── assets/
│       └── style.css
│
├── models/
│   ├── disaster_predictor.pkl
│   ├── location_encoder.pkl
│   ├── disaster_encoder.pkl
│   ├── season_encoder.pkl
│   ├── model_classes.pkl
│   └── disaster_mapping.pkl
│
├── requirements.txt
├── README.md
└── check_raw_data.py




┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Processing│───▶│  ML Model      │
│ (CSV Files)     │    │   & Cleaning    │    │  Training      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Forecasting   │◀───│   Model         │    │  Dashboard     │
│  Engine        │    │  Evaluation    │    │  (Streamlit)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Directory Descriptions

- **data/**: Contains raw and processed disaster data
  - **raw/**: Original disaster data CSV file
  - **processed/**: Cleaned data and forecast results
- **notebooks/**: Jupyter notebooks for data analysis and model training
- **src/**: Python modules for core functionality
- **app/**: Streamlit dashboard application
- **models/**: Trained machine learning models and encoders
- **requirements.txt**: Python dependencies
- **README.md**: Project documentation
- **check_raw_data.py**: Utility script to examine raw data structure

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the Repository**

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
  
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import pandas, numpy, sklearn, streamlit, plotly; print('All dependencies installed successfully')"
   ```

---

## Data Processing Pipeline

### 1. Data Preprocessing (`notebooks/01_data_preprocessing.ipynb`)

**Purpose**: Clean and prepare raw disaster data for analysis.

**Steps**:
1. Load raw CSV data with `header=None` to handle unnamed columns
2. Skip first row if it contains file information
3. Assign column names based on data structure:
   - Column 0: date
   - Column 1: disaster_type
   - Column 2: id
   - Column 3: location
   - Column 4: severity_index
4. Convert date strings to datetime objects
5. Extract temporal features (year, month, day, day_of_week)
6. Standardize disaster type names using mapping dictionary
7. Handle missing values in location field
8. Save cleaned data to `data/processed/cleaned_disaster_data.csv`

**Key Functions**:
- `parse_date()`: Converts various date string formats to datetime objects
- Disaster type mapping: Standardizes disaster type names (e.g., "DOMESTIC FIRE" → "FIRE")

### 2. Exploratory Analysis (`notebooks/02_exploratory_analysis.ipynb`)

**Purpose**: Analyze historical disaster patterns and generate visualizations.

**Analyses Performed**:
- Disaster type distribution (pie chart)
- Top affected locations (bar chart)
- Monthly disaster patterns
- Yearly trends
- Location-specific analysis

**Visualizations Generated**:
- Disaster distribution pie chart
- Location frequency bar chart
- Monthly trend line charts
- Yearly trend analysis

### 3. Feature Engineering (`src/feature_engineering.py`)

**Purpose**: Create features for machine learning model.

**Features Created**:
- **Location Encoding**: Numerical representation of location names
- **Disaster Type Encoding**: Numerical representation of disaster types
- **Season Feature**: Categorize months into seasons (Spring, Summer, Fall, Winter)
- **Season Encoding**: Numerical representation of seasons
- **Location Risk**: Historical frequency of disasters at each location
- **Disaster Frequency**: Overall frequency of each disaster type

**Key Function**:
```python
def create_features(df):
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
    
    # Additional processing...
```

---

## Machine Learning Model

### Model Selection and Training (`notebooks/03_modeling.ipynb`)

**Algorithm**: Random Forest Classifier
- **Why Random Forest?**
  - Handles both categorical and numerical features well
  - Provides feature importance
  - Resistant to overfitting
  - Handles non-linear relationships

**Model Architecture**:
- **Number of Estimators**: 100
- **Random State**: 42 (for reproducibility)
- **Test Size**: 20% of data

**Features Used**:
1. Year
2. Month
3. Day
4. Day of week
5. Location encoded
6. Season encoded
7. Location risk
8. Disaster frequency

**Training Process**:
1. Split data into training and testing sets
2. Train Random Forest classifier on training set
3. Evaluate model on test set
4. Save trained model and encoders

**Model Performance**:
- **Accuracy**: ~75% (varies based on data split)
- **Evaluation Metrics**: Classification report, confusion matrix

**Model Persistence**:
- Saved as `models/disaster_predictor.pkl`
- Encoders saved for future use:
  - `models/location_encoder.pkl`
  - `models/disaster_encoder.pkl`
  - `models/season_encoder.pkl`
  - `models/disaster_mapping.pkl`

---

## Forecasting System

### 5-Year Forecast Generation (`notebooks/04_forecasting.ipynb`)

**Purpose**: Generate disaster forecasts for 2024-2028 for all locations.

**Forecasting Process**:
1. **Generate Future Dates**: Create monthly dates from 2024 to 2028
2. **Create Location-Date Combinations**: For each location and date combination
3. **Feature Engineering**: Apply same feature engineering as training data
4. **Make Predictions**: Use trained model to predict disaster types
5. **Calculate Probabilities**: Get prediction probabilities for all disaster types
6. **Aggregate Results**: Summarize forecasts by location

**Key Function**:
```python
def generate_future_forecast(historical_data, model, location_encoder, 
                             disaster_encoder, season_encoder, 
                             start_year=2024, end_year=2028):
    # Generate future dates
    future_dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            future_dates.append({
                'year': year,
                'month': month,
                'day': 15,
                'date': pd.to_datetime(f'{year}-{month}-15')
            })
    
    # Create DataFrame for all location-date combinations
    forecast_df = pd.DataFrame(forecast_data)
    
    # Apply feature engineering
    # ... (same as training data)
    
    # Make predictions
    X_forecast = forecast_df[features]
    forecast_df['disaster_encoded'] = model.predict(X_forecast)
    
    # Get probabilities
    proba = model.predict_proba(X_forecast)
    
    # Convert encoded predictions to disaster types
    forecast_df['disaster_type'] = forecast_df['disaster_encoded'].map(disaster_mapping)
    
    return forecast_df
```

**Output Files**:
- `forecast_results.csv`: Monthly predictions for all locations
- `location_forecast_summary.csv`: Most likely disaster by location
- `disaster_percentages.csv`: Probability matrix of disasters by location

---

## Dashboard Features

The Streamlit dashboard (`app/streamlit_dashboard.py`) provides four main sections:

### 1. Overview Page
- **Key Metrics**: Total disasters, affected locations, disaster types, forecast years
- **Disaster Distribution**: Pie chart showing historical disaster type distribution
- **Top Affected Locations**: Bar chart of most disaster-prone locations
- **Forecast Summary**: Most likely disasters by location
- **Disaster Probability Heatmap**: Heatmap showing disaster probabilities by location

### 2. Historical Analysis Page
- **Location Selector**: Choose any location to analyze
- **Yearly Trend**: Line chart showing disaster frequency over years
- **Monthly Pattern**: Bar chart showing seasonal patterns
- **Disaster Types**: Breakdown of disaster types for selected location

### 3. Forecast Page
- **Location and Year Selectors**: Choose location and year for forecast
- **Monthly Predictions**: Table showing predicted disasters by month
- **Probability Chart**: Stacked bar chart of disaster probabilities
- **5-Year Trend**: Line chart showing most likely disaster trend over time

### 4. Prediction Tool Page
- **Input Form**: Select location, year, month, and day
- **Prediction Result**: Shows predicted disaster type
- **Probability Breakdown**: Bar chart showing probabilities for all disaster types
- **Probability Table**: Detailed probability percentages

### Dashboard Styling
- **Custom CSS**: Located at `app/assets/style.css`
- **Responsive Design**: Works on various screen sizes
- **Color Scheme**: Professional blue-based color scheme
- **Interactive Elements**: All charts and selectors are fully interactive

---

## How to Run the Application

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Processing
Run the notebooks in sequence:

1. **Data Preprocessing**:
   ```bash
   jupyter notebook notebooks/01_data_preprocessing.ipynb
   ```
   - Execute all cells to clean and prepare data

2. **Exploratory Analysis**:
   ```bash
   jupyter notebook notebooks/02_exploratory_analysis.ipynb
   ```
   - Execute all cells to analyze historical patterns

3. **Model Training**:
   ```bash
   jupyter notebook notebooks/03_modeling.ipynb
   ```
   - Execute all cells to train the machine learning model

4. **Forecast Generation**:
   ```bash
   jupyter notebook notebooks/04_forecasting.ipynb
   ```
   - Execute all cells to generate 5-year forecasts

### Step 3: Start Dashboard
```bash
streamlit run app/streamlit_dashboard.py
```

### Step 4: Access Dashboard
- Open the provided URL (usually http://localhost:8501)
- Navigate through the four main sections using the sidebar
- Interact with charts and selectors to explore data

### Alternative: Automated Script

```python
import subprocess
import os

def run_notebook(notebook_path):
    print(f"Running {notebook_path}...")
    subprocess.run([
        "jupyter", "nbconvert", "--to", "script", "--execute",
        "--ExecutePreprocessor.timeout=600", notebook_path
    ], check=True)
    print(f"Finished {notebook_path}")

def main():
    # Run notebooks in order
    notebooks = [
        "notebooks/01_data_preprocessing.ipynb",
        "notebooks/02_exploratory_analysis.ipynb",
        "notebooks/03_modeling.ipynb",
        "notebooks/04_forecasting.ipynb"
    ]
    
    for notebook in notebooks:
        run_notebook(notebook)
    
    # Start Streamlit app
    print("Starting Streamlit dashboard...")
    subprocess.run(["streamlit", "run", "app/streamlit_dashboard.py"])

if __name__ == "__main__":
    main()
```


## Conclusion

The Ghana Natural Disaster Analysis and Forecasting Project provides a comprehensive solution for understanding historical disaster patterns and predicting future occurrences. By leveraging machine learning techniques and interactive visualizations, the system offers valuable insights for disaster preparedness and resource allocation.

### Key Achievements
1. **Comprehensive Data Processing**: Successfully cleaned and standardized disaster data from various sources
2. **Accurate Predictive Model**: Developed a Random Forest classifier with ~75% accuracy
3. **5-Year Forecasting System**: Generated monthly predictions for all locations in Ghana
4. **Interactive Dashboard**: Created a user-friendly interface for data exploration and prediction
5. **Scalable Architecture**: Designed modular system that can be easily extended

### Impact
- **Disaster Preparedness**: Enables better planning and resource allocation
- **Risk Assessment**: Identifies high-risk areas for targeted interventions
- **Policy Making**: Provides data-driven insights for policy decisions
- **Public Awareness**: Increases disaster awareness through accessible visualizations

### Lessons Learned
1. **Data Quality is Crucial**: Garbage in, garbage out - proper data cleaning is essential
2. **Feature Engineering Matters**: Well-designed features significantly improve model performance
3. **User Experience is Key**: Complex models need simple, intuitive interfaces
4. **Iterative Development**: Continuous improvement based on feedback is necessary
5. **Documentation is Vital**: Comprehensive documentation ensures project sustainability

This project serves as a foundation for future disaster management systems in Ghana and can be adapted for similar applications in other regions. The combination of historical analysis, predictive modeling, and interactive visualization provides a powerful tool for disaster risk reduction and management.

---



## Acknowledgments

- **National Disaster Management Organization (NADMO)**: For providing the disaster data
- **University Faculty**: For guidance and support throughout the project
- **Open Source Community**: For the tools and libraries that made this project possible

---
