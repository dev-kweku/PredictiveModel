import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import os
# Set page config
st.set_page_config(
    page_title="Ghana Natural Disaster Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Suppress pandas warnings
pd.set_option('mode.chained_assignment', None)
# Custom CSS
def load_css():
    css_path = os.path.join('app', 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()
# Load data
@st.cache_data
def load_data():
    historical_df = pd.read_csv('data/processed/cleaned_disaster_data.csv')
    forecast_df = pd.read_csv('data/processed/forecast_results.csv')
    location_summary = pd.read_csv('data/processed/location_forecast_summary.csv')
    disaster_pct = pd.read_csv('data/processed/disaster_percentages.csv', index_col=0)
    return historical_df, forecast_df, location_summary, disaster_pct
# Load models
@st.cache_resource
def load_models():
    model = joblib.load('models/disaster_predictor.pkl')
    location_encoder = joblib.load('models/location_encoder.pkl')
    disaster_encoder = joblib.load('models/disaster_encoder.pkl')
    season_encoder = joblib.load('models/season_encoder.pkl')
    
    # Try to load disaster mapping, if it doesn't exist, create it from model classes
    try:
        disaster_mapping = joblib.load('models/disaster_mapping.pkl')
    except FileNotFoundError:
        disaster_mapping = {i: disaster_type for i, disaster_type in enumerate(model.classes_)}
        joblib.dump(disaster_mapping, 'models/disaster_mapping.pkl')
    
    return model, location_encoder, disaster_encoder, season_encoder, disaster_mapping
# Feature engineering function for prediction
def create_prediction_features(input_data, historical_data, location_encoder, season_encoder):
    # Create a copy to avoid SettingWithCopyWarning
    input_data = input_data.copy()
    
    # Add day of week
    input_data['day_of_week'] = input_data['date'].dt.dayofweek
    
    # Apply feature engineering
    input_data['location_encoded'] = location_encoder.transform(input_data['location'])
    
    # Create season
    input_data['season'] = input_data['month'].apply(lambda x: 
        'Spring' if x in [3,4,5] else
        'Summer' if x in [6,7,8] else
        'Fall' if x in [9,10,11] else 'Winter')
    
    # Use manual season mapping instead of the fitted encoder
    season_mapping = {
        'Spring': 0,
        'Summer': 1,
        'Fall': 2,
        'Winter': 3
    }
    input_data['season_encoded'] = input_data['season'].map(season_mapping)
    
    # Calculate location risk
    location_risk = historical_data.groupby('location').size() / len(historical_data)
    input_data['location_risk'] = input_data['location'].map(location_risk)
    
    # Calculate disaster frequency
    disaster_freq = historical_data.groupby('disaster_type').size() / len(historical_data)
    avg_disaster_freq = disaster_freq.mean()
    input_data['disaster_freq'] = avg_disaster_freq
    
    return input_data
# Load data and models
historical_df, forecast_df, location_summary, disaster_pct = load_data()
model, location_encoder, disaster_encoder, season_encoder, disaster_mapping = load_models()
# Debug: Check what we're working with
if st.checkbox("Show Debug Info"):
    st.write("Model classes:", model.classes_)
    st.write("Disaster mapping:", disaster_mapping)
    st.write("Forecast data types:")
    st.write(forecast_df.dtypes)
    st.write("Forecast data sample:")
    st.write(forecast_df.head())
    st.write("Location summary sample:")
    st.write(location_summary.head())
    st.write("Disaster pct sample:")
    st.write(disaster_pct.head())
# Apply disaster type mapping to forecast_df
if 'disaster_type' in forecast_df.columns:
    if forecast_df['disaster_type'].dtype in [np.int64, np.int32, int, float]:
        forecast_df['disaster_type'] = forecast_df['disaster_type'].map(disaster_mapping)
# Apply disaster type mapping to location_summary
if 'most_likely_disaster' in location_summary.columns:
    # Handle different data types in the column
    if location_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int, float]:
        location_summary['most_likely_disaster'] = location_summary['most_likely_disaster'].map(disaster_mapping)
    else:
        # If it's object type, try to convert to numeric first
        try:
            numeric_vals = pd.to_numeric(location_summary['most_likely_disaster'], errors='coerce')
            location_summary['most_likely_disaster'] = numeric_vals.map(disaster_mapping).fillna(location_summary['most_likely_disaster'])
        except:
            # If conversion fails, try direct mapping
            location_summary['most_likely_disaster'] = location_summary['most_likely_disaster'].map(disaster_mapping)
# Update disaster_pct columns if they are numeric
if disaster_pct.columns.dtype in [np.int64, np.int32, int, float]:
    new_columns = []
    for col in disaster_pct.columns:
        try:
            # Try to convert to int and map
            col_int = int(col)
            new_columns.append(disaster_mapping.get(col_int, str(col)))
        except (ValueError, TypeError):
            # If conversion fails, keep original
            new_columns.append(str(col))
    disaster_pct.columns = new_columns
# Sidebar
st.sidebar.title("🌪️ Disaster Dashboard")
st.sidebar.markdown("### Navigation")
# UPDATED: Changed navigation order to Overview, Prediction, Forecast, Historical Analysis
page = st.sidebar.radio("Go to", ["Overview", "Prediction", "Forecast", "Historical Analysis"])
# Overview Page
if page == "Overview":
    st.title("Ghana Natural Disaster Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Disasters", f"{historical_df.shape[0]:,}")
    with col2:
        st.metric("Affected Locations", f"{historical_df['location'].nunique():,}")
    with col3:
        st.metric("Disaster Types", f"{historical_df['disaster_type'].nunique():,}")
    with col4:
        st.metric("Forecast Years", "4 (2025-2028)")  # Updated from 5 to 4 years
    
    # Disaster distribution comparison
    st.subheader("Historical vs Forecasted Disaster Distribution")
    
    # Historical distribution
    hist_disaster_counts = historical_df['disaster_type'].value_counts().reset_index()
    hist_disaster_counts['source'] = 'Historical'
    
    # Forecast distribution - filter out 2024
    forecast_df_filtered = forecast_df[forecast_df['year'] != 2024]
    forecast_disaster_counts = forecast_df_filtered['disaster_type'].value_counts().reset_index()
    forecast_disaster_counts['source'] = 'Forecast'
    
    # Combine for comparison
    combined_disaster_counts = pd.concat([hist_disaster_counts, forecast_disaster_counts])
    
    # Create comparison chart
    fig_comparison = px.bar(
        combined_disaster_counts,
        x='disaster_type',
        y='count',
        color='source',
        title="Historical vs Forecasted Disaster Distribution",
        barmode='group'
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Top affected locations comparison
    st.subheader("Top 10 Most Affected Locations: Historical vs Forecast")
    
    # Historical top locations
    hist_top_locations = historical_df['location'].value_counts().head(10).reset_index()
    hist_top_locations['source'] = 'Historical'
    
    # Forecast top locations (using location summary) - filter out 2024
    forecast_top_locations = location_summary.copy()
    forecast_top_locations['count'] = forecast_top_locations.groupby('most_likely_disaster')['location'].transform('count')
    forecast_top_locations = forecast_top_locations[['location', 'count']].drop_duplicates().head(10)
    forecast_top_locations['source'] = 'Forecast'
    
    # Combine for comparison
    combined_top_locations = pd.concat([
        hist_top_locations.rename(columns={'count': 'value'}),
        forecast_top_locations.rename(columns={'count': 'value'})
    ])
    
    # Create comparison chart
    fig_locations = px.bar(
        combined_top_locations,
        x='value',
        y='location',
        orientation='h',
        color='source',
        title="Historical vs Forecasted Disaster Frequency by Location",
        barmode='group'
    )
    st.plotly_chart(fig_locations, use_container_width=True)
    
    # Forecast summary
    st.subheader("4-Year Forecast Summary")  # Updated from 5-Year to 4-Year
    st.markdown("### Most Likely Disasters by Location")
    
    # Ensure location summary displays disaster names, not codes
    display_summary = location_summary.copy()
    if 'most_likely_disaster' in display_summary.columns:
        # Check if we still have numeric values
        if display_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int, float]:
            display_summary['most_likely_disaster'] = display_summary['most_likely_disaster'].map(disaster_mapping)
        # Fill any remaining NaN values
        display_summary['most_likely_disaster'] = display_summary['most_likely_disaster'].fillna('Unknown')
    
    st.dataframe(display_summary.sort_values('location'))
    
    st.markdown("### Disaster Probability Heatmap")
    fig_heatmap = px.imshow(
        disaster_pct,
        labels=dict(x="Disaster Type", y="Location", color="Probability (%)"),
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
# Prediction Page (moved to second position)
elif page == "Prediction":
    st.title("Disaster Prediction")
    
    st.markdown("""
    Use this tool to predict the most likely disaster type for a specific location and time.
    The model uses historical patterns to make predictions.
    """)
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.selectbox(
                "Location",
                options=historical_df['location'].unique(),
                index=0
            )
            
            year = st.number_input(
                "Year",
                min_value=2025,  # Changed from 2024 to 2025
                max_value=2028,
                value=2025,  # Changed default from 2024 to 2025
                step=1
            )
        
        with col2:
            month = st.selectbox(
                "Month",
                options=range(1, 13),
                format_func=lambda x: datetime(2024, x, 1).strftime('%B'),
                index=6
            )
            
            day = st.number_input(
                "Day",
                min_value=1,
                max_value=31,
                value=15,
                step=1
            )
        
        # UPDATED: Changed button color to green using type="primary"
        submit = st.form_submit_button("Predict Disaster", type="primary")
    
    if submit:
        # Prepare input data
        input_data = pd.DataFrame({
            'location': [location],
            'year': [year],
            'month': [month],
            'day': [day],
            'date': [pd.to_datetime(f'{year}-{month}-{day}')]
        })
        
        # Apply feature engineering
        input_data = create_prediction_features(input_data, historical_df, location_encoder, season_encoder)
        
        # Prepare features
        features = ['year', 'month', 'day', 'day_of_week', 'location_encoded', 
                    'season_encoded', 'location_risk', 'disaster_freq']
        
        # Make prediction
        prediction = model.predict(input_data[features])[0]
        
        # Get probabilities
        proba = model.predict_proba(input_data[features])[0]
        
        # Use model's classes_ attribute to get disaster classes
        disaster_classes = model.classes_
        
        # Convert prediction to disaster type with bounds checking
        try:
            if 0 <= prediction < len(disaster_classes):
                disaster_type = disaster_mapping.get(prediction, f"Unknown Disaster (Code: {prediction})")
            else:
                # If prediction is out of bounds, use disaster mapping
                disaster_type = disaster_mapping.get(prediction, f"Unknown Disaster (Code: {prediction})")
        except Exception as e:
            st.error(f"Error converting prediction to disaster type: {e}")
            disaster_type = f"Error (Code: {prediction})"
        
        # Display results
        st.success(f"### Predicted Disaster: **{disaster_type}**")
        
        st.subheader("Prediction Probabilities")
        
        # Create a list of disaster names by mapping the numeric classes
        disaster_names = [disaster_mapping.get(cls, f"Unknown ({cls})") for cls in disaster_classes]
        
        # Create probability dataframe with disaster names
        prob_df = pd.DataFrame({
            'Disaster Type': disaster_names,
            'Probability': proba * 100
        })
        
        # UPDATED: Ensure the predicted disaster comes first, then sort the rest by probability descending
        # Create a priority column: 0 for the predicted disaster, 1 for others
        prob_df['priority'] = prob_df['Disaster Type'].apply(lambda x: 0 if x == disaster_type else 1)
        
        # Sort by priority (ascending) and then by probability (descending)
        prob_df = prob_df.sort_values(by=['priority', 'Probability'], ascending=[True, False])
        
        # Remove the priority column
        prob_df = prob_df.drop(columns=['priority'])
        
        fig_proba = px.bar(
            prob_df,
            x='Probability',
            y='Disaster Type',
            orientation='h',
            title="Disaster Type Probabilities",
            color='Probability',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_proba, use_container_width=True)
        
        st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}))
# Forecast Page (moved to third position)
elif page == "Forecast":
    st.title("4-Year Disaster Forecast (2025-2028)")  # Updated title
    
    # Filter out 2024 from forecast data
    forecast_df_filtered = forecast_df[forecast_df['year'] != 2024]
    
    # Location selector
    forecast_location = st.selectbox(
        "Select Location for Forecast",
        options=sorted(forecast_df_filtered['location'].unique()),
        index=0
    )
    
    # Year selector - only show 2025-2028
    available_years = sorted(forecast_df_filtered['year'].unique())
    if available_years:
        forecast_year = st.selectbox(
            "Select Year",
            options=available_years,
            index=0
        )
    else:
        st.error("No forecast data available for any year.")
        st.stop()
    
    # Filter forecast data and create a copy
    location_forecast = forecast_df_filtered[
        (forecast_df_filtered['location'] == forecast_location) & 
        (forecast_df_filtered['year'] == forecast_year)
    ].copy()
    
    # Check if we have data for the selected location and year
    if location_forecast.empty:
        st.warning(f"No forecast data available for {forecast_location} in {forecast_year}.")
    else:
        # Get historical data for comparison
        historical_location_data = historical_df[historical_df['location'] == forecast_location].copy()
        historical_location_data['date'] = pd.to_datetime(historical_location_data['date'])
        historical_location_data['year'] = historical_location_data['date'].dt.year
        historical_location_data['month'] = historical_location_data['date'].dt.month
        
        # Display forecast
        st.subheader(f"Disaster Forecast for {forecast_location} in {forecast_year}")
        
        # Create forecast table
        forecast_display = location_forecast[['month', 'disaster_type']].copy()
        forecast_display['month'] = forecast_display['month'].apply(
            lambda x: datetime(2024, x, 1).strftime('%B')
        )
        forecast_display.columns = ['Month', 'Predicted Disaster']
        st.dataframe(forecast_display)
        
        # Historical vs Forecast comparison
        st.subheader(f"Historical vs Forecasted Disaster Patterns for {forecast_location}")
        
        # Get historical monthly pattern
        hist_monthly = historical_location_data.groupby('month').size().reset_index(name='count')
        hist_monthly['source'] = 'Historical'
        
        # Get forecast monthly pattern
        forecast_monthly = location_forecast.groupby('month').size().reset_index(name='count')
        forecast_monthly['source'] = 'Forecast'
        
        # Combine for comparison
        combined_monthly = pd.concat([hist_monthly, forecast_monthly])
        
        # Create comparison chart
        fig_monthly_comparison = px.bar(
            combined_monthly,
            x='month',
            y='count',
            color='source',
            title=f"Historical vs Forecasted Monthly Disaster Pattern for {forecast_location}",
            barmode='group'
        )
        st.plotly_chart(fig_monthly_comparison, use_container_width=True)
        
        # Probability chart
        st.subheader("Disaster Probability by Month")
        
        # Get probability columns
        prob_cols = [col for col in location_forecast.columns if col.startswith('prob_')]
        if prob_cols:
            prob_data = location_forecast[['month'] + prob_cols].copy()
            
            # Melt for plotting
            prob_melted = prob_data.melt(
                id_vars=['month'],
                var_name='disaster_type',
                value_name='probability'
            )
            prob_melted['disaster_type'] = prob_melted['disaster_type'].str.replace('prob_', '')
            
            # Map disaster codes to names if needed
            if prob_melted['disaster_type'].dtype in [np.int64, np.int32, int, float]:
                # Try to convert to int and map
                try:
                    prob_melted['disaster_type'] = prob_melted['disaster_type'].astype(int).map(disaster_mapping)
                except (ValueError, TypeError):
                    # If conversion fails, try to map directly
                    prob_melted['disaster_type'] = prob_melted['disaster_type'].map(disaster_mapping)
            
            prob_melted['month'] = prob_melted['month'].apply(
                lambda x: datetime(2024, x, 1).strftime('%B')
            )
            
            fig_prob = px.bar(
                prob_melted,
                x='month',
                y='probability',
                color='disaster_type',
                title=f"Disaster Probabilities for {forecast_location} in {forecast_year}",
                barmode='stack'
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("No probability data available for this forecast.")
        
        # 4-year trend (updated from 5-year)
        st.subheader(f"4-Year Disaster Trend for {forecast_location}")
        
        # Get most likely disaster each year - filter out 2024
        yearly_forecast = forecast_df_filtered[forecast_df_filtered['location'] == forecast_location].copy()
        if not yearly_forecast.empty:
            yearly_summary = yearly_forecast.groupby('year')['disaster_type'].apply(
                lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
            ).reset_index(name='most_likely_disaster')
            
            # Map disaster codes to names if needed
            if yearly_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int, float]:
                # Try to convert to int and map
                try:
                    yearly_summary['most_likely_disaster'] = yearly_summary['most_likely_disaster'].astype(int).map(disaster_mapping)
                except (ValueError, TypeError):
                    # If conversion fails, try to map directly
                    yearly_summary['most_likely_disaster'] = yearly_summary['most_likely_disaster'].map(disaster_mapping)
            
            # Get historical yearly trend for comparison
            hist_yearly = historical_location_data.groupby('year').size().reset_index(name='count')
            
            # Create figure with secondary y-axis
            fig_trend = go.Figure()
            
            # Add historical count
            fig_trend.add_trace(
                go.Scatter(
                    x=hist_yearly['year'],
                    y=hist_yearly['count'],
                    name='Historical Count',
                    mode='markers+lines',
                    yaxis='y'
                )
            )
            
            # Add forecast disaster type
            fig_trend.add_trace(
                go.Scatter(
                    x=yearly_summary['year'],
                    y=yearly_summary['most_likely_disaster'],
                    name='Forecast Disaster Type',
                    mode='markers+lines',
                    yaxis='y2'
                )
            )
            
            # Create layout with two y-axes
            fig_trend.update_layout(
                title=f"Historical Count vs Forecast Disaster Type for {forecast_location}",
                xaxis=dict(title='Year'),
                yaxis=dict(
                    title=dict(text='Historical Disaster Count', font=dict(color='blue')),
                    tickfont=dict(color='blue')
                ),
                yaxis2=dict(
                    title=dict(text='Forecast Disaster Type', font=dict(color='red')),
                    tickfont=dict(color='red'),
                    anchor='x',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No yearly trend data available for this location.")
# Historical Analysis Page (moved to fourth position)
elif page == "Historical Analysis":
    st.title("Historical Disaster Analysis")
    
    # Location selector
    selected_location = st.selectbox(
        "Select Location",
        options=sorted(historical_df['location'].unique()),
        index=0
    )
    
    # Filter data and create a copy to avoid SettingWithCopyWarning
    location_data = historical_df[historical_df['location'] == selected_location].copy()
    
    # Convert date
    location_data['date'] = pd.to_datetime(location_data['date'])
    location_data['year'] = location_data['date'].dt.year
    location_data['month'] = location_data['date'].dt.month
    
    # Yearly trend
    st.subheader(f"Yearly Disaster Trend in {selected_location}")
    yearly_counts = location_data.groupby('year').size().reset_index(name='count')
    fig_yearly = px.line(
        yearly_counts,
        x='year',
        y='count',
        title=f"Disaster Frequency in {selected_location}",
        markers=True
    )
    st.plotly_chart(fig_yearly, use_container_width=True)
    
    # Monthly pattern
    st.subheader(f"Monthly Disaster Pattern in {selected_location}")
    monthly_counts = location_data.groupby('month').size().reset_index(name='count')
    fig_monthly = px.bar(
        monthly_counts,
        x='month',
        y='count',
        title=f"Monthly Disaster Frequency in {selected_location}",
        color='count',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Disaster types
    st.subheader(f"Disaster Types in {selected_location}")
    disaster_types = location_data['disaster_type'].value_counts().reset_index()
    fig_types = px.bar(
        disaster_types,
        x='count',
        y='disaster_type',
        orientation='h',
        title=f"Disaster Types in {selected_location}",
        color='count',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_types, use_container_width=True)
# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This dashboard analyzes natural disasters in Ghana and provides forecasts for the next 4 years.")
st.sidebar.markdown("Developed as a Final Year Project")