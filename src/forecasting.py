# src/forecasting.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class DisasterForecaster:
    def __init__(self, model_path='models/disaster_model.pkl'):
        """Initialize the forecaster with the trained classification model"""
        self.model_objects = joblib.load(model_path)
        self.model = self.model_objects['model']
        self.label_encoder = self.model_objects['label_encoder']
        self.scaler = self.model_objects['scaler']
        self.feature_cols = self.model_objects['feature_cols']
        self.categorical_features = self.model_objects['categorical_features']
        
        # Initialize time series models
        self.count_models = {}
        self.seasonal_models = {}
        
    def get_season(self, month):
        """Determine season from month"""
        if month in [12, 1, 2]:
            return 'WINTER'
        elif month in [3, 4, 5]:
            return 'SPRING'
        elif month in [6, 7, 8]:
            return 'SUMMER'
        else:
            return 'FALL'
    
    def create_features_for_date(self, date_obj, severity, location):
        """Create features for a given date, severity, and location"""
        features = {
            'Year': date_obj.year,
            'Month': date_obj.month,
            'Quarter': (date_obj.month - 1) // 3 + 1,
            'DayOfWeek': date_obj.dayofweek,
            'DayOfYear': date_obj.dayofyear,
            'WeekOfYear': date_obj.isocalendar().week,
            'Season': self.get_season(date_obj.month),
            'Is_Weekend': date_obj.dayofweek >= 5,
            'Severity': severity,
            'Location_History_Count': 0,  # Default value
            'Disaster_History_Count': 0,  # Default value
            'Disasters_Last_30_Days': 0,  # Default value
            'Disasters_Last_90_Days': 0,  # Default value
            'Location_Disaster_History': 0  # Default value
        }
        return features
    
    def predict_disaster_for_date(self, date_obj, severity, location):
        """Predict disaster type for a given date, severity, and location"""
        # Create features
        features = self.create_features_for_date(date_obj, severity, location)
        features_df = pd.DataFrame([features])
        
        # One-hot encode categorical features
        features_encoded = pd.get_dummies(features_df, columns=self.categorical_features, drop_first=True)
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in features_encoded.columns and col not in self.categorical_features:
                features_encoded[col] = 0
        
        # Add missing columns from training
        missing_cols = set(self.feature_cols) - set(features_encoded.columns)
        for col in missing_cols:
            features_encoded[col] = 0
        
        # Ensure column order matches training
        features_encoded = features_encoded[self.feature_cols]
        
        # Scale features
        features_scaled = self.scaler.transform(features_encoded)
        
        # Make prediction
        prediction_encoded = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Decode the prediction
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get class labels and probabilities
        classes_encoded = np.arange(len(self.label_encoder.classes_))
        classes = self.label_encoder.inverse_transform(classes_encoded)
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
        
        return {
            'predicted_disaster': prediction,
            'probabilities': prob_dict,
            'date': date_obj,
            'severity': severity,
            'location': location
        }
    
    def prepare_time_series_data(self, df):
        """Prepare data for time series forecasting"""
        # Ensure date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create time-based features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        
        # Create lag features
        df = df.sort_values('Date')
        df['Disasters_Last_7_Days'] = df.groupby('Disaster_Type')['Date'].transform(
            lambda x: x.rolling('7D', closed='left').count()
        ).fillna(0)
        
        df['Disasters_Last_30_Days'] = df.groupby('Disaster_Type')['Date'].transform(
            lambda x: x.rolling('30D', closed='left').count()
        ).fillna(0)
        
        # Create moving averages
        df['MA_7'] = df.groupby('Disaster_Type')['Disasters_Last_7_Days'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        df['MA_30'] = df.groupby('Disaster_Type')['Disasters_Last_30_Days'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        return df
    
    def train_time_series_models(self, df):
        """Train time series models for disaster count prediction"""
        df = self.prepare_time_series_data(df)
        
        # Aggregate data by date and disaster type
        daily_counts = df.groupby(['Date', 'Disaster_Type']).size().reset_index(name='Count')
        
        # Merge with features
        features_df = df[['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 
                        'WeekOfYear', 'Quarter', 'Disasters_Last_7_Days', 
                        'Disasters_Last_30_Days', 'MA_7', 'MA_30']].drop_duplicates()
        
        daily_data = pd.merge(daily_counts, features_df, on='Date', how='left')
        
        # Train a model for each disaster type
        for disaster_type in daily_data['Disaster_Type'].unique():
            disaster_data = daily_data[daily_data['Disaster_Type'] == disaster_type].copy()
            
            # Prepare features
            feature_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 
                        'WeekOfYear', 'Quarter', 'Disasters_Last_7_Days', 
                        'Disasters_Last_30_Days', 'MA_7', 'MA_30']
            
            X = disaster_data[feature_cols]
            y = disaster_data['Count']
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            self.count_models[disaster_type] = {
                'model': model,
                'feature_cols': feature_cols
            }
        
        # Train seasonal pattern models
        monthly_patterns = df.groupby(['Month', 'Disaster_Type']).size().reset_index(name='Count')
        
        for disaster_type in monthly_patterns['Disaster_Type'].unique():
            disaster_data = monthly_patterns[monthly_patterns['Disaster_Type'] == disaster_type]
            
            # Create seasonal features
            disaster_data['Month_Sin'] = np.sin(2 * np.pi * disaster_data['Month'] / 12)
            disaster_data['Month_Cos'] = np.cos(2 * np.pi * disaster_data['Month'] / 12)
            
            X = disaster_data[['Month_Sin', 'Month_Cos']]
            y = disaster_data['Count']
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            self.seasonal_models[disaster_type] = model
    
    def forecast_disaster_counts(self, start_date, end_date, locations):
        """Forecast disaster counts for a date range and locations"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        forecasts = []
        
        for date in date_range:
            for location in locations:
                # Get base prediction for each severity level
                for severity in [1, 2, 3]:
                    result = self.predict_disaster_for_date(date, severity, location)
                    
                    # Get count prediction
                    disaster_type = result['predicted_disaster']
                    probability = result['probabilities'][disaster_type]
                    
                    # Predict count using time series model
                    if disaster_type in self.count_models:
                        # Create features for count prediction
                        count_features = {
                            'Year': date.year,
                            'Month': date.month,
                            'Day': date.day,
                            'DayOfWeek': date.dayofweek,
                            'DayOfYear': date.dayofyear,
                            'WeekOfYear': date.isocalendar().week,
                            'Quarter': (date.month - 1) // 3 + 1,
                            'Disasters_Last_7_Days': 0,  # Default
                            'Disasters_Last_30_Days': 0,  # Default
                            'MA_7': 0,  # Default
                            'MA_30': 0  # Default
                        }
                        
                        count_df = pd.DataFrame([count_features])
                        predicted_count = self.count_models[disaster_type]['model'].predict(
                            count_df[self.count_models[disaster_type]['feature_cols']]
                        )[0]
                        
                        # Adjust count by probability
                        adjusted_count = max(1, int(predicted_count * probability))
                    else:
                        adjusted_count = 1
                    
                    forecasts.append({
                        'Date': date,
                        'Location': location,
                        'Severity': severity,
                        'Predicted_Disaster': disaster_type,
                        'Probability': probability,
                        'Predicted_Count': adjusted_count
                    })
        
        return pd.DataFrame(forecasts)
    
    def forecast_high_risk_periods(self, forecast_df, risk_threshold=0.7):
        """Identify high-risk periods based on forecast"""
        # Calculate risk score
        forecast_df['Risk_Score'] = forecast_df['Probability'] * forecast_df['Predicted_Count']
        
        # Identify high-risk periods
        high_risk = forecast_df[forecast_df['Risk_Score'] >= risk_threshold]
        
        # Aggregate by date
        daily_risk = high_risk.groupby('Date').agg({
            'Risk_Score': 'sum',
            'Predicted_Count': 'sum',
            'Predicted_Disaster': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        }).reset_index()
        
        # Sort by risk score
        daily_risk = daily_risk.sort_values('Risk_Score', ascending=False)
        
        return daily_risk
    
    def generate_forecast_report(self, start_date, end_date, locations):
        """Generate a comprehensive forecast report"""
        # Get forecasts
        forecast_df = self.forecast_disaster_counts(start_date, end_date, locations)
        
        # Get high-risk periods
        high_risk_periods = self.forecast_high_risk_periods(forecast_df)
        
        # Aggregate by disaster type
        disaster_summary = forecast_df.groupby('Predicted_Disaster').agg({
            'Predicted_Count': 'sum',
            'Probability': 'mean'
        }).reset_index()
        
        # Aggregate by location
        location_summary = forecast_df.groupby('Location').agg({
            'Predicted_Count': 'sum',
            'Risk_Score': 'sum'
        }).reset_index()
        
        # Aggregate by month
        forecast_df['Month'] = forecast_df['Date'].dt.month
        monthly_summary = forecast_df.groupby('Month').agg({
            'Predicted_Count': 'sum',
            'Risk_Score': 'sum'
        }).reset_index()
        
        return {
            'forecast_df': forecast_df,
            'high_risk_periods': high_risk_periods,
            'disaster_summary': disaster_summary,
            'location_summary': location_summary,
            'monthly_summary': monthly_summary
        }