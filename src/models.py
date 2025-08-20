import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_predictive_model(X, y):
    """
    Train a predictive model using Random Forest
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm

def save_model(model, location_encoder, disaster_encoder, season_encoder, model_dir):
    """
    Save the trained model and encoders
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and encoders
    joblib.dump(model, os.path.join(model_dir, 'disaster_predictor.pkl'))
    joblib.dump(location_encoder, os.path.join(model_dir, 'location_encoder.pkl'))
    joblib.dump(disaster_encoder, os.path.join(model_dir, 'disaster_encoder.pkl'))
    joblib.dump(season_encoder, os.path.join(model_dir, 'season_encoder.pkl'))
    
    print("Model and encoders saved successfully!")

def load_model(model_dir):
    """
    Load the trained model and encoders
    """
    model = joblib.load(os.path.join(model_dir, '../models/disaster_predictor.pkl'))
    location_encoder = joblib.load(os.path.join(model_dir, '../models/location_encoder.pkl'))
    disaster_encoder = joblib.load(os.path.join(model_dir, '../models/disaster_encoder.pkl'))
    season_encoder = joblib.load(os.path.join(model_dir, '../models/season_encoder.pkl'))
    
    return model, location_encoder, disaster_encoder, season_encoder