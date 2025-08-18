# main.py
import pandas as pd
import numpy as np
import sys
import os
from src.data_preprocessing import load_data, clean_data, save_processed_data
from src.feature_engineering import prepare_features
from src.model_training import (
    handle_imbalanced_data, train_models, evaluate_models, 
    select_best_model, time_series_cv, save_model, plot_feature_importance
)
from src.utils import (
    plot_disaster_distribution, plot_temporal_patterns, 
    plot_location_distribution, plot_severity_distribution, 
    generate_summary_report
)
from src.prediction import predict_disaster
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Ghana Disaster Prediction Model - Training")
    print("=" * 40)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    raw_data = load_data('data/Nadmo.new.xlsx')
    
    if raw_data is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    clean_df = clean_data(raw_data)
    
    if clean_df is None or len(clean_df) == 0:
        print("No valid data after cleaning. Exiting.")
        sys.exit(1)
    
    save_processed_data(clean_df, 'data/processed_data.csv')
    print(f"Processed data saved with {len(clean_df)} records")
    
    # Step 2: Exploratory Data Analysis
    print("\n2. Performing exploratory data analysis...")
    clean_df['Date'] = pd.to_datetime(clean_df['Date'])
    
    # Generate plots
    plot_disaster_distribution(clean_df, 'reports/disaster_distribution.png')
    plot_temporal_patterns(clean_df, 'reports/temporal_patterns.png')
    plot_location_distribution(clean_df, save_path='reports/location_distribution.png')
    plot_severity_distribution(clean_df, save_path='reports/severity_distribution.png')
    
    # Generate summary report
    generate_summary_report(clean_df, 'reports/data_summary.txt')
    print("EDA complete. Visualizations saved to reports/ directory")
    
    # Step 3: Feature engineering
    print("\n3. Engineering features...")
    X, y = prepare_features(clean_df)
    print(f"Features prepared: {X.shape[1]} features for {len(X)} records")
    
    # Step 4: Handle imbalanced data
    print("\n4. Handling imbalanced data...")
    df_upsampled = handle_imbalanced_data(clean_df)
    X_upsampled, y_upsampled = prepare_features(df_upsampled)
    print(f"Data after resampling: {len(df_upsampled)} records")
    
    # Step 5: Split data
    print("\n5. Splitting data into train and test sets...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_upsampled, y_upsampled, test_size=0.2, random_state=42, stratify=y_upsampled
    )
    print(f"Train set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")
    
    # Step 6: Scale features
    print("\n6. Scaling features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 7: Train models
    print("\n7. Training models...")
    models = train_models(X_train_scaled, y_train)
    
    # Step 8: Evaluate models
    print("\n8. Evaluating models...")
    results = evaluate_models(models, X_test_scaled, y_test)
    
    # Step 9: Select best model
    print("\n9. Selecting best model...")
    best_model_name, best_model, best_score = select_best_model(results)
    
    # Step 10: Time series cross-validation
    print("\n10. Performing time series cross-validation...")
    cv_scores = time_series_cv(best_model, X, y)
    
    # Step 11: Feature importance
    print("\n11. Analyzing feature importance...")
    importance_df = plot_feature_importance(best_model, X_train.columns)
    if importance_df is not None:
        importance_df.to_csv('reports/feature_importance.csv', index=False)
        print("Feature importance saved to reports/feature_importance.csv")
    
    # Step 12: Save model
    print("\n12. Saving model...")
    save_model(
        best_model, scaler, X_train.columns, ['Season'], 
        'models/disaster_model.pkl'
    )
    
    print("\n" + "=" * 40)
    print("Model training completed successfully!")
    print(f"Best model: {best_model_name} with F1-score: {best_score:.4f}")
    print(f"Cross-validation score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nTo run the dashboard, use: streamlit run app.py")

if __name__ == "__main__":
    main()