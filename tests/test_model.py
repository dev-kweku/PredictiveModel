import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_training import handle_imbalanced_data, train_models, evaluate_models

def test_handle_imbalanced_data():
    # Create imbalanced test data
    data = {
        'Date': pd.to_datetime(['2023-01-01'] * 10 + ['2023-01-02'] * 2),
        'Disaster_Type': ['FLOOD'] * 10 + ['STORM'] * 2,
        'Location': ['ACCRA'] * 10 + ['KUMASI'] * 2,
        'Severity': [1] * 12
    }
    df = pd.DataFrame(data)
    
    # Handle imbalance
    df_balanced = handle_imbalanced_data(df)
    
    # Assertions
    assert len(df_balanced) > len(df)
    assert df_balanced['Disaster_Type'].value_counts().min() >= 5

def test_train_models():
    # Create test data
    X_train = np.random.rand(100, 10)
    y_train = np.random.choice(['FLOOD', 'STORM', 'FIRE'], size=100)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Assertions
    assert len(models) == 3
    assert 'Random Forest' in models
    assert 'XGBoost' in models
    assert 'LightGBM' in models

def test_evaluate_models():
    # Create test data
    X_test = np.random.rand(20, 10)
    y_test = np.random.choice(['FLOOD', 'STORM', 'FIRE'], size=20)
    
    # Create mock models
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(np.random.rand(100, 10), np.random.choice(['FLOOD', 'STORM', 'FIRE'], size=100))
    
    models = {'Random Forest': model}
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Assertions
    assert 'Random Forest' in results
    assert 'accuracy' in results['Random Forest']
    assert 'precision' in results['Random Forest']
    assert 'recall' in results['Random Forest']
    assert 'f1-score' in results['Random Forest']