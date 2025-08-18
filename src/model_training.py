# src/model_training.py (updated with label encoding)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def handle_imbalanced_data(df, target_col='Disaster_Type'):
    """Handle imbalanced data using resampling"""
    # Get class distribution
    class_distribution = df[target_col].value_counts()
    
    # Separate majority and minority classes
    df_majority = df[df[target_col] == class_distribution.index[0]]
    df_minority_list = []
    
    for disaster_type in class_distribution.index:
        if disaster_type != class_distribution.index[0]:
            df_minority = df[df[target_col] == disaster_type]
            df_minority_list.append(df_minority)
    
    # Upsample minority classes
    df_minority_upsampled_list = []
    for df_minority in df_minority_list:
        n_samples = len(df_majority) // 2
        df_minority_upsampled = resample(
            df_minority, 
            replace=True,
            n_samples=n_samples,
            random_state=42
        )
        df_minority_upsampled_list.append(df_minority_upsampled)
    
    # Combine majority and upsampled minority classes
    df_upsampled = pd.concat([df_majority] + df_minority_upsampled_list)
    
    return df_upsampled

def train_models(X_train, y_train):
    """Train multiple models and return the best one"""
    # Initialize label encoder
    label_encoder = LabelEncoder()
    
    # Fit and transform the target variable
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train_encoded)
        trained_models[name] = {
            'model': model,
            'label_encoder': label_encoder
        }
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return performance metrics"""
    # Encode the test labels
    label_encoder = list(models.values())[0]['label_encoder']
    y_test_encoded = label_encoder.transform(y_test)
    
    results = {}
    
    for name, model_dict in models.items():
        model = model_dict['model']
        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score'],
            'model': model_dict['model'],
            'label_encoder': label_encoder
        }
        
        # Print classification report
        print(f"\n{name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'reports/{name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()
    
    return results

def select_best_model(results):
    """Select the best model based on F1-score"""
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1-score'])
    best_model_dict = results[best_model_name]
    best_score = results[best_model_name]['f1-score']
    
    print(f"\nBest Model: {best_model_name} with F1-score: {best_score:.4f}")
    
    return best_model_name, best_model_dict, best_score

def time_series_cv(model_dict, X, y):
    """Perform time series cross-validation"""
    model = model_dict['model']
    label_encoder = model_dict['label_encoder']
    y_encoded = label_encoder.transform(y)
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y_encoded, cv=tscv, scoring='f1_weighted')
    
    print(f"\nTime Series Cross-Validation Results:")
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f}")
    print(f"CV Score Std Dev: {cv_scores.std():.4f}")
    
    return cv_scores

def save_model(model_dict, scaler, feature_cols, categorical_features, output_path):
    """Save the trained model and preprocessing objects"""
    model_objects = {
        'model': model_dict['model'],
        'label_encoder': model_dict['label_encoder'],
        'scaler': scaler,
        'feature_cols': feature_cols,
        'categorical_features': categorical_features
    }
    
    joblib.dump(model_objects, output_path)
    print(f"\nModel saved to {output_path}")

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('reports/feature_importance.png')
        plt.close()
        
        return importance_df
    else:
        print("Model does not support feature importance analysis")
        return None

if __name__ == "__main__":
    # Example usage
    from src.feature_engineering import prepare_features
    
    # Load processed data
    df = pd.read_csv('data/processed_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Handle imbalanced data
    df_upsampled = handle_imbalanced_data(df)
    X_upsampled, y_upsampled = prepare_features(df_upsampled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_upsampled, y_upsampled, test_size=0.2, random_state=42, stratify=y_upsampled
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = train_models(X_train_scaled, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test_scaled, y_test)
    
    # Select best model
    best_model_name, best_model_dict, best_score = select_best_model(results)
    
    # Time series cross-validation
    cv_scores = time_series_cv(best_model_dict, X, y)
    
    # Plot feature importance
    importance_df = plot_feature_importance(best_model_dict['model'], X_train.columns)
    
    # Save model
    save_model(
        best_model_dict, scaler, X_train.columns, ['Season'], 
        'models/disaster_model.pkl'
    )