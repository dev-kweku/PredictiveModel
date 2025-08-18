# app.py (updated to handle label encoder)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import prepare_features
from src.model_training import handle_imbalanced_data, train_models, evaluate_models, select_best_model, save_model
from src.prediction import predict_disaster
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ghana Disaster Prediction Dashboard",
    page_icon="🌪️",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        model_objects = joblib.load('models/disaster_model.pkl')
        return model_objects
    except:
        return None

# Load and preprocess data
@st.cache_data
def load_and_process_data():
    raw_data = load_data('data/Nadmo.new.xlsx')
    if raw_data is not None:
        clean_df = clean_data(raw_data)
        return clean_df
    return None

# Train model if not exists
def train_model_if_needed():
    try:
        # Try to load the model
        model_objects = joblib.load('models/disaster_model.pkl')
        return model_objects
    except:
        # Model doesn't exist, train it
        st.info("Training model... This may take a few minutes.")
        
        # Load and preprocess data
        raw_data = load_data('data/Nadmo.new.xlsx')
        clean_df = clean_data(raw_data)
        
        if clean_df is None or len(clean_df) == 0:
            st.error("Failed to load or process data.")
            return None
        
        # Prepare features
        X, y = prepare_features(clean_df)
        
        # Handle imbalanced data
        df_upsampled = handle_imbalanced_data(clean_df)
        X_upsampled, y_upsampled = prepare_features(df_upsampled)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_upsampled, y_upsampled, test_size=0.2, random_state=42, stratify=y_upsampled
        )
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = train_models(X_train_scaled, y_train)
        
        # Evaluate models
        results = evaluate_models(models, X_test_scaled, y_test)
        
        # Select best model
        best_model_name, best_model_dict, best_score = select_best_model(results)
        
        # Save model
        save_model(
            best_model_dict, scaler, X_train.columns, ['Season'], 
            'models/disaster_model.pkl'
        )
        
        st.success(f"Model trained successfully! Best model: {best_model_name} with F1-score: {best_score:.4f}")
        
        return best_model_dict

# Main function
def main():
    st.title("🌪️ Ghana Disaster Prediction Dashboard")
    st.markdown("This dashboard predicts disaster types in Ghana based on historical data.")
    
    # Load data
    df = load_and_process_data()
    
    if df is None:
        st.error("Failed to load or process data. Please check the data file.")
        return
    
    # Train or load model
    model_objects = train_model_if_needed()
    
    if model_objects is None:
        st.error("Failed to train or load model.")
        return
    
    # Sidebar for user inputs
    st.sidebar.header("Prediction Parameters")
    
    # Date input
    date_input = st.sidebar.date_input("Select Date")
    
    # Severity input
    severity_input = st.sidebar.slider("Severity Level", 1, 3, 2)
    
    # Location input
    locations = sorted(df['Location'].unique())
    location_input = st.sidebar.selectbox("Select Location", locations)
    
    # Predict button
    if st.sidebar.button("Predict Disaster"):
        # Make prediction
        result = predict_disaster(
            date_input.strftime('%Y-%m-%d'), 
            severity_input, 
            location_input,
            model_objects
        )
        
        # Display prediction
        st.subheader("Prediction Results")
        st.write(f"**Predicted Disaster Type:** {result['predicted_disaster']}")
        
        # Display probabilities
        st.write("**Probabilities:**")
        prob_df = pd.DataFrame(result['probabilities'], columns=['Disaster Type', 'Probability'])
        fig = px.bar(prob_df, x='Probability', y='Disaster Type', orientation='h', 
                    color='Probability', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Data visualization section
    st.header("Disaster Data Analysis")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Disaster Types", "Temporal Patterns", "Locations", "Severity"])
    
    with tab1:
        st.subheader("Disaster Type Distribution")
        disaster_counts = df['Disaster_Type'].value_counts().reset_index()
        disaster_counts.columns = ['Disaster Type', 'Count']
        fig = px.bar(disaster_counts, x='Count', y='Disaster Type', orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Temporal Patterns")
        
        # Extract year and month
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Yearly distribution
        yearly_counts = df.groupby('Year').size().reset_index(name='Count')
        fig1 = px.line(yearly_counts, x='Year', y='Count', title='Yearly Disaster Count')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Monthly distribution
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_counts = df.groupby('Month').size().reset_index(name='Count')
        monthly_counts['Month'] = monthly_counts['Month'].apply(lambda x: month_names[x-1])
        fig2 = px.bar(monthly_counts, x='Month', y='Count', title='Monthly Disaster Count')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Top Disaster Locations")
        location_counts = df['Location'].value_counts().head(20).reset_index()
        location_counts.columns = ['Location', 'Count']
        fig = px.bar(location_counts, x='Count', y='Location', orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Severity Distribution")
        severity_counts = df['Severity'].value_counts().reset_index()
        severity_counts.columns = ['Severity', 'Count']
        severity_counts['Severity'] = severity_counts['Severity'].apply(lambda x: f"Level {x}")
        fig = px.pie(severity_counts, values='Count', names='Severity')
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.header("Raw Data")
    st.dataframe(df.head(100))

if __name__ == "__main__":
    main()