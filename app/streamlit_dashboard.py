# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import joblib
# import os

# # Set page config
# st.set_page_config(
#     page_title="Ghana Natural Disaster Dashboard",
#     page_icon="🌪️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Suppress pandas warnings
# pd.set_option('mode.chained_assignment', None)

# # Custom CSS
# def load_css():
#     css_path = os.path.join('app', 'assets', 'style.css')
#     if os.path.exists(css_path):
#         with open(css_path) as f:
#             st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# load_css()

# # Load data
# @st.cache_data
# def load_data():
#     historical_df = pd.read_csv('data/processed/cleaned_disaster_data.csv')
#     forecast_df = pd.read_csv('data/processed/forecast_results.csv')
#     location_summary = pd.read_csv('data/processed/location_forecast_summary.csv')
#     disaster_pct = pd.read_csv('data/processed/disaster_percentages.csv', index_col=0)
#     return historical_df, forecast_df, location_summary, disaster_pct

# # Load models
# @st.cache_resource
# def load_models():
#     model = joblib.load('models/disaster_predictor.pkl')
#     location_encoder = joblib.load('models/location_encoder.pkl')
#     disaster_encoder = joblib.load('models/disaster_encoder.pkl')
#     season_encoder = joblib.load('models/season_encoder.pkl')
    
#     # Try to load disaster mapping, if it doesn't exist, create it from model classes
#     try:
#         disaster_mapping = joblib.load('models/disaster_mapping.pkl')
#     except FileNotFoundError:
#         disaster_mapping = {i: disaster_type for i, disaster_type in enumerate(model.classes_)}
#         joblib.dump(disaster_mapping, 'models/disaster_mapping.pkl')
    
#     return model, location_encoder, disaster_encoder, season_encoder, disaster_mapping

# # Feature engineering function for prediction
# def create_prediction_features(input_data, historical_data, location_encoder, season_encoder):
#     # Create a copy to avoid SettingWithCopyWarning
#     input_data = input_data.copy()
    
#     # Add day of week
#     input_data['day_of_week'] = input_data['date'].dt.dayofweek
    
#     # Apply feature engineering
#     input_data['location_encoded'] = location_encoder.transform(input_data['location'])
    
#     # Create season
#     input_data['season'] = input_data['month'].apply(lambda x: 
#         'Spring' if x in [3,4,5] else
#         'Summer' if x in [6,7,8] else
#         'Fall' if x in [9,10,11] else 'Winter')
    
#     # Use manual season mapping instead of the fitted encoder
#     season_mapping = {
#         'Spring': 0,
#         'Summer': 1,
#         'Fall': 2,
#         'Winter': 3
#     }
#     input_data['season_encoded'] = input_data['season'].map(season_mapping)
    
#     # Calculate location risk
#     location_risk = historical_data.groupby('location').size() / len(historical_data)
#     input_data['location_risk'] = input_data['location'].map(location_risk)
    
#     # Calculate disaster frequency
#     disaster_freq = historical_data.groupby('disaster_type').size() / len(historical_data)
#     avg_disaster_freq = disaster_freq.mean()
#     input_data['disaster_freq'] = avg_disaster_freq
    
#     return input_data

# # Load data and models
# historical_df, forecast_df, location_summary, disaster_pct = load_data()
# model, location_encoder, disaster_encoder, season_encoder, disaster_mapping = load_models()

# # Convert disaster type numbers to names in forecast data if needed
# if 'disaster_type' in forecast_df.columns and forecast_df['disaster_type'].dtype in [np.int64, np.int32, int]:
#     # Create mapping from model's classes
#     disaster_mapping = {i: disaster_type for i, disaster_type in enumerate(model.classes_)}
#     forecast_df['disaster_type'] = forecast_df['disaster_type'].map(disaster_mapping)
    
#     # Update location summary if it contains numeric disaster types
#     if 'most_likely_disaster' in location_summary.columns and location_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int]:
#         location_summary['most_likely_disaster'] = location_summary['most_likely_disaster'].map(disaster_mapping)
    
#     # Update disaster_pct columns if they are numeric
#     if disaster_pct.columns.dtype in [np.int64, np.int32, int]:
#         new_columns = [disaster_mapping.get(col, str(col)) for col in disaster_pct.columns]
#         disaster_pct.columns = new_columns

# # Debug information
# if st.checkbox("Show Debug Info"):
#     st.write("Model classes:", model.classes_)
#     st.write("Model classes length:", len(model.classes_))
#     st.write("Disaster mapping:", disaster_mapping)
#     st.write("Sample forecast data types:")
#     st.write(forecast_df.dtypes)
#     st.write("Sample forecast data:")
#     st.write(forecast_df.head())

# # Sidebar
# st.sidebar.title("🌪️ Disaster Dashboard")
# st.sidebar.markdown("### Navigation")
# page = st.sidebar.radio("Go to", ["Overview", "Historical Analysis", "Forecast", "Prediction Tool"])

# # Overview Page
# if page == "Overview":
#     st.title("Ghana Natural Disaster Overview")
    
#     # Key metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Disasters", f"{historical_df.shape[0]:,}")
#     with col2:
#         st.metric("Affected Locations", f"{historical_df['location'].nunique():,}")
#     with col3:
#         st.metric("Disaster Types", f"{historical_df['disaster_type'].nunique():,}")
#     with col4:
#         st.metric("Forecast Years", "5 (2024-2028)")
    
#     # Disaster distribution
#     st.subheader("Disaster Type Distribution")
#     disaster_counts = historical_df['disaster_type'].value_counts().reset_index()
#     fig_disaster = px.pie(
#         disaster_counts, 
#         values='count', 
#         names='disaster_type',
#         title="Historical Disaster Distribution",
#         hole=0.4
#     )
#     st.plotly_chart(fig_disaster, use_container_width=True)
    
#     # Top affected locations
#     st.subheader("Top 10 Most Affected Locations")
#     top_locations = historical_df['location'].value_counts().head(10).reset_index()
#     fig_locations = px.bar(
#         top_locations,
#         x='count',
#         y='location',
#         orientation='h',
#         title="Historical Disaster Frequency by Location",
#         color='count',
#         color_continuous_scale='Viridis'
#     )
#     st.plotly_chart(fig_locations, use_container_width=True)
    
#     # Forecast summary
#     st.subheader("5-Year Forecast Summary")
#     st.markdown("### Most Likely Disasters by Location")
#     st.dataframe(location_summary.sort_values('location'))
    
#     st.markdown("### Disaster Probability Heatmap")
#     fig_heatmap = px.imshow(
#         disaster_pct,
#         labels=dict(x="Disaster Type", y="Location", color="Probability (%)"),
#         color_continuous_scale='Viridis',
#         aspect="auto"
#     )
#     st.plotly_chart(fig_heatmap, use_container_width=True)

# # Historical Analysis Page
# elif page == "Historical Analysis":
#     st.title("Historical Disaster Analysis")
    
#     # Location selector
#     selected_location = st.selectbox(
#         "Select Location",
#         options=historical_df['location'].unique(),
#         index=0
#     )
    
#     # Filter data and create a copy to avoid SettingWithCopyWarning
#     location_data = historical_df[historical_df['location'] == selected_location].copy()
    
#     # Convert date
#     location_data['date'] = pd.to_datetime(location_data['date'])
#     location_data['year'] = location_data['date'].dt.year
#     location_data['month'] = location_data['date'].dt.month
    
#     # Yearly trend
#     st.subheader(f"Yearly Disaster Trend in {selected_location}")
#     yearly_counts = location_data.groupby('year').size().reset_index(name='count')
#     fig_yearly = px.line(
#         yearly_counts,
#         x='year',
#         y='count',
#         title=f"Disaster Frequency in {selected_location}",
#         markers=True
#     )
#     st.plotly_chart(fig_yearly, use_container_width=True)
    
#     # Monthly pattern
#     st.subheader(f"Monthly Disaster Pattern in {selected_location}")
#     monthly_counts = location_data.groupby('month').size().reset_index(name='count')
#     fig_monthly = px.bar(
#         monthly_counts,
#         x='month',
#         y='count',
#         title=f"Monthly Disaster Frequency in {selected_location}",
#         color='count',
#         color_continuous_scale='Blues'
#     )
#     st.plotly_chart(fig_monthly, use_container_width=True)
    
#     # Disaster types
#     st.subheader(f"Disaster Types in {selected_location}")
#     disaster_types = location_data['disaster_type'].value_counts().reset_index()
#     fig_types = px.bar(
#         disaster_types,
#         x='count',
#         y='disaster_type',
#         orientation='h',
#         title=f"Disaster Types in {selected_location}",
#         color='count',
#         color_continuous_scale='Reds'
#     )
#     st.plotly_chart(fig_types, use_container_width=True)

# # Forecast Page
# elif page == "Forecast":
#     st.title("5-Year Disaster Forecast (2024-2028)")
    
#     # Location selector
#     forecast_location = st.selectbox(
#         "Select Location for Forecast",
#         options=forecast_df['location'].unique(),
#         index=0
#     )
    
#     # Year selector
#     forecast_year = st.selectbox(
#         "Select Year",
#         options=sorted(forecast_df['year'].unique()),
#         index=0
#     )
    
#     # Filter forecast data and create a copy
#     location_forecast = forecast_df[
#         (forecast_df['location'] == forecast_location) & 
#         (forecast_df['year'] == forecast_year)
#     ].copy()
    
#     # Display forecast
#     st.subheader(f"Disaster Forecast for {forecast_location} in {forecast_year}")
    
#     # Create forecast table
#     forecast_display = location_forecast[['month', 'disaster_type']].copy()
#     forecast_display['month'] = forecast_display['month'].apply(
#         lambda x: datetime(2024, x, 1).strftime('%B')
#     )
#     forecast_display.columns = ['Month', 'Predicted Disaster']
#     st.dataframe(forecast_display)
    
#     # Probability chart
#     st.subheader("Disaster Probability by Month")
    
#     # Get probability columns
#     prob_cols = [col for col in location_forecast.columns if col.startswith('prob_')]
#     prob_data = location_forecast[['month'] + prob_cols].copy()
    
#     # Melt for plotting
#     prob_melted = prob_data.melt(
#         id_vars=['month'],
#         var_name='disaster_type',
#         value_name='probability'
#     )
#     prob_melted['disaster_type'] = prob_melted['disaster_type'].str.replace('prob_', '')
#     prob_melted['month'] = prob_melted['month'].apply(
#         lambda x: datetime(2024, x, 1).strftime('%B')
#     )
    
#     fig_prob = px.bar(
#         prob_melted,
#         x='month',
#         y='probability',
#         color='disaster_type',
#         title=f"Disaster Probabilities for {forecast_location} in {forecast_year}",
#         barmode='stack'
#     )
#     st.plotly_chart(fig_prob, use_container_width=True)
    
#     # 5-year trend
#     st.subheader(f"5-Year Disaster Trend for {forecast_location}")
    
#     # Get most likely disaster each year
#     yearly_forecast = forecast_df[forecast_df['location'] == forecast_location].copy()
#     yearly_summary = yearly_forecast.groupby('year')['disaster_type'].apply(
#         lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
#     ).reset_index(name='most_likely_disaster')
    
#     fig_trend = px.line(
#         yearly_summary,
#         x='year',
#         y='most_likely_disaster',
#         title=f"Most Likely Disaster Trend for {forecast_location}",
#         markers=True,
#         line_shape='linear'
#     )
#     fig_trend.update_traces(mode='markers+lines')
#     st.plotly_chart(fig_trend, use_container_width=True)

# # Prediction Tool Page
# elif page == "Prediction Tool":
#     st.title("Disaster Prediction Tool")
    
#     st.markdown("""
#     Use this tool to predict the most likely disaster type for a specific location and time.
#     The model uses historical patterns to make predictions.
#     """)
    
#     # Input form
#     with st.form("prediction_form"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             location = st.selectbox(
#                 "Location",
#                 options=historical_df['location'].unique(),
#                 index=0
#             )
            
#             year = st.number_input(
#                 "Year",
#                 min_value=2024,
#                 max_value=2028,
#                 value=2024,
#                 step=1
#             )
        
#         with col2:
#             month = st.selectbox(
#                 "Month",
#                 options=range(1, 13),
#                 format_func=lambda x: datetime(2024, x, 1).strftime('%B'),
#                 index=6
#             )
            
#             day = st.number_input(
#                 "Day",
#                 min_value=1,
#                 max_value=31,
#                 value=15,
#                 step=1
#             )
        
#         submit = st.form_submit_button("Predict Disaster")
    
#     if submit:
#         # Prepare input data
#         input_data = pd.DataFrame({
#             'location': [location],
#             'year': [year],
#             'month': [month],
#             'day': [day],
#             'date': [pd.to_datetime(f'{year}-{month}-{day}')]
#         })
        
#         # Apply feature engineering
#         input_data = create_prediction_features(input_data, historical_df, location_encoder, season_encoder)
        
#         # Prepare features
#         features = ['year', 'month', 'day', 'day_of_week', 'location_encoded', 
#                     'season_encoded', 'location_risk', 'disaster_freq']
        
#         # Make prediction
#         prediction = model.predict(input_data[features])[0]
        
#         # Get probabilities
#         proba = model.predict_proba(input_data[features])[0]
        
#         # Use model's classes_ attribute to get disaster classes
#         disaster_classes = model.classes_
        
#         # Convert prediction to disaster type with bounds checking
#         try:
#             if 0 <= prediction < len(disaster_classes):
#                 disaster_type = disaster_classes[prediction]
#             else:
#                 # If prediction is out of bounds, use disaster mapping
#                 disaster_type = disaster_mapping.get(prediction, f"Unknown Disaster (Code: {prediction})")
#         except Exception as e:
#             st.error(f"Error converting prediction to disaster type: {e}")
#             disaster_type = f"Error (Code: {prediction})"
        
#         # Display results
#         st.success(f"### Predicted Disaster: **{disaster_type}**")
        
#         st.subheader("Prediction Probabilities")
#         prob_df = pd.DataFrame({
#             'Disaster Type': disaster_classes,
#             'Probability': proba * 100
#         }).sort_values('Probability', ascending=False)
        
#         fig_proba = px.bar(
#             prob_df,
#             x='Probability',
#             y='Disaster Type',
#             orientation='h',
#             title="Disaster Type Probabilities",
#             color='Probability',
#             color_continuous_scale='Viridis'
#         )
#         st.plotly_chart(fig_proba, use_container_width=True)
        
#         st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}))

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("### About")
# st.sidebar.markdown("This dashboard analyzes natural disasters in Ghana and provides forecasts for the next 5 years.")
# st.sidebar.markdown("Developed as a Final Year Project")




# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import joblib
# import os

# # Set page config
# st.set_page_config(
#     page_title="Ghana Natural Disaster Dashboard",
#     page_icon="🌪️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Suppress pandas warnings
# pd.set_option('mode.chained_assignment', None)

# # Custom CSS
# def load_css():
#     css_path = os.path.join('app', 'assets', 'style.css')
#     if os.path.exists(css_path):
#         with open(css_path) as f:
#             st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# load_css()

# # Load data
# @st.cache_data
# def load_data():
#     historical_df = pd.read_csv('data/processed/cleaned_disaster_data.csv')
#     forecast_df = pd.read_csv('data/processed/forecast_results.csv')
#     location_summary = pd.read_csv('data/processed/location_forecast_summary.csv')
#     disaster_pct = pd.read_csv('data/processed/disaster_percentages.csv', index_col=0)
#     return historical_df, forecast_df, location_summary, disaster_pct

# # Load models
# @st.cache_resource
# def load_models():
#     model = joblib.load('models/disaster_predictor.pkl')
#     location_encoder = joblib.load('models/location_encoder.pkl')
#     disaster_encoder = joblib.load('models/disaster_encoder.pkl')
#     season_encoder = joblib.load('models/season_encoder.pkl')
    
#     # Try to load disaster mapping, if it doesn't exist, create it from model classes
#     try:
#         disaster_mapping = joblib.load('models/disaster_mapping.pkl')
#     except FileNotFoundError:
#         disaster_mapping = {i: disaster_type for i, disaster_type in enumerate(model.classes_)}
#         joblib.dump(disaster_mapping, 'models/disaster_mapping.pkl')
    
#     return model, location_encoder, disaster_encoder, season_encoder, disaster_mapping

# # Feature engineering function for prediction
# def create_prediction_features(input_data, historical_data, location_encoder, season_encoder):
#     # Create a copy to avoid SettingWithCopyWarning
#     input_data = input_data.copy()
    
#     # Add day of week
#     input_data['day_of_week'] = input_data['date'].dt.dayofweek
    
#     # Apply feature engineering
#     input_data['location_encoded'] = location_encoder.transform(input_data['location'])
    
#     # Create season
#     input_data['season'] = input_data['month'].apply(lambda x: 
#         'Spring' if x in [3,4,5] else
#         'Summer' if x in [6,7,8] else
#         'Fall' if x in [9,10,11] else 'Winter')
    
#     # Use manual season mapping instead of the fitted encoder
#     season_mapping = {
#         'Spring': 0,
#         'Summer': 1,
#         'Fall': 2,
#         'Winter': 3
#     }
#     input_data['season_encoded'] = input_data['season'].map(season_mapping)
    
#     # Calculate location risk
#     location_risk = historical_data.groupby('location').size() / len(historical_data)
#     input_data['location_risk'] = input_data['location'].map(location_risk)
    
#     # Calculate disaster frequency
#     disaster_freq = historical_data.groupby('disaster_type').size() / len(historical_data)
#     avg_disaster_freq = disaster_freq.mean()
#     input_data['disaster_freq'] = avg_disaster_freq
    
#     return input_data

# # Load data and models
# historical_df, forecast_df, location_summary, disaster_pct = load_data()
# model, location_encoder, disaster_encoder, season_encoder, disaster_mapping = load_models()

# # Convert disaster type numbers to names in forecast data if needed
# if 'disaster_type' in forecast_df.columns and forecast_df['disaster_type'].dtype in [np.int64, np.int32, int]:
#     # Create mapping from model's classes
#     disaster_mapping = {i: disaster_type for i, disaster_type in enumerate(model.classes_)}
#     forecast_df['disaster_type'] = forecast_df['disaster_type'].map(disaster_mapping)
    
#     # Update location summary if it contains numeric disaster types
#     if 'most_likely_disaster' in location_summary.columns and location_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int]:
#         location_summary['most_likely_disaster'] = location_summary['most_likely_disaster'].map(disaster_mapping)
    
#     # Update disaster_pct columns if they are numeric
#     if disaster_pct.columns.dtype in [np.int64, np.int32, int]:
#         new_columns = [disaster_mapping.get(col, str(col)) for col in disaster_pct.columns]
#         disaster_pct.columns = new_columns

# # Debug information
# if st.checkbox("Show Debug Info"):
#     st.write("Model classes:", model.classes_)
#     st.write("Model classes length:", len(model.classes_))
#     st.write("Disaster mapping:", disaster_mapping)
#     st.write("Sample forecast data types:")
#     st.write(forecast_df.dtypes)
#     st.write("Sample forecast data:")
#     st.write(forecast_df.head())

# # Sidebar
# st.sidebar.title("🌪️ Disaster Dashboard")
# st.sidebar.markdown("### Navigation")
# page = st.sidebar.radio("Go to", ["Overview", "Historical Analysis", "Forecast", "Prediction Tool"])

# # Overview Page
# if page == "Overview":
#     st.title("Ghana Natural Disaster Overview")
    
#     # Key metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Disasters", f"{historical_df.shape[0]:,}")
#     with col2:
#         st.metric("Affected Locations", f"{historical_df['location'].nunique():,}")
#     with col3:
#         st.metric("Disaster Types", f"{historical_df['disaster_type'].nunique():,}")
#     with col4:
#         st.metric("Forecast Years", "5 (2024-2028)")
    
#     # Disaster distribution
#     st.subheader("Disaster Type Distribution")
#     disaster_counts = historical_df['disaster_type'].value_counts().reset_index()
#     fig_disaster = px.pie(
#         disaster_counts, 
#         values='count', 
#         names='disaster_type',
#         title="Historical Disaster Distribution",
#         hole=0.4
#     )
#     st.plotly_chart(fig_disaster, use_container_width=True)
    
#     # Top affected locations
#     st.subheader("Top 10 Most Affected Locations")
#     top_locations = historical_df['location'].value_counts().head(10).reset_index()
#     fig_locations = px.bar(
#         top_locations,
#         x='count',
#         y='location',
#         orientation='h',
#         title="Historical Disaster Frequency by Location",
#         color='count',
#         color_continuous_scale='Viridis'
#     )
#     st.plotly_chart(fig_locations, use_container_width=True)
    
#     # Forecast summary
#     st.subheader("5-Year Forecast Summary")
#     st.markdown("### Most Likely Disasters by Location")
    
#     # Ensure location summary displays disaster names, not codes
#     display_summary = location_summary.copy()
#     if 'most_likely_disaster' in display_summary.columns and display_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int]:
#         display_summary['most_likely_disaster'] = display_summary['most_likely_disaster'].map(disaster_mapping)
    
#     st.dataframe(display_summary.sort_values('location'))
    
#     st.markdown("### Disaster Probability Heatmap")
    
#     # Ensure disaster_pct has proper column names
#     display_disaster_pct = disaster_pct.copy()
#     if display_disaster_pct.columns.dtype in [np.int64, np.int32, int]:
#         new_columns = [disaster_mapping.get(col, str(col)) for col in display_disaster_pct.columns]
#         display_disaster_pct.columns = new_columns
    
#     fig_heatmap = px.imshow(
#         display_disaster_pct,
#         labels=dict(x="Disaster Type", y="Location", color="Probability (%)"),
#         color_continuous_scale='Viridis',
#         aspect="auto"
#     )
#     st.plotly_chart(fig_heatmap, use_container_width=True)

# # Historical Analysis Page
# elif page == "Historical Analysis":
#     st.title("Historical Disaster Analysis")
    
#     # Location selector
#     selected_location = st.selectbox(
#         "Select Location",
#         options=historical_df['location'].unique(),
#         index=0
#     )
    
#     # Filter data and create a copy to avoid SettingWithCopyWarning
#     location_data = historical_df[historical_df['location'] == selected_location].copy()
    
#     # Convert date
#     location_data['date'] = pd.to_datetime(location_data['date'])
#     location_data['year'] = location_data['date'].dt.year
#     location_data['month'] = location_data['date'].dt.month
    
#     # Yearly trend
#     st.subheader(f"Yearly Disaster Trend in {selected_location}")
#     yearly_counts = location_data.groupby('year').size().reset_index(name='count')
#     fig_yearly = px.line(
#         yearly_counts,
#         x='year',
#         y='count',
#         title=f"Disaster Frequency in {selected_location}",
#         markers=True
#     )
#     st.plotly_chart(fig_yearly, use_container_width=True)
    
#     # Monthly pattern
#     st.subheader(f"Monthly Disaster Pattern in {selected_location}")
#     monthly_counts = location_data.groupby('month').size().reset_index(name='count')
#     fig_monthly = px.bar(
#         monthly_counts,
#         x='month',
#         y='count',
#         title=f"Monthly Disaster Frequency in {selected_location}",
#         color='count',
#         color_continuous_scale='Blues'
#     )
#     st.plotly_chart(fig_monthly, use_container_width=True)
    
#     # Disaster types
#     st.subheader(f"Disaster Types in {selected_location}")
#     disaster_types = location_data['disaster_type'].value_counts().reset_index()
#     fig_types = px.bar(
#         disaster_types,
#         x='count',
#         y='disaster_type',
#         orientation='h',
#         title=f"Disaster Types in {selected_location}",
#         color='count',
#         color_continuous_scale='Reds'
#     )
#     st.plotly_chart(fig_types, use_container_width=True)

# # Forecast Page
# elif page == "Forecast":
#     st.title("5-Year Disaster Forecast (2024-2028)")
    
#     # Location selector
#     forecast_location = st.selectbox(
#         "Select Location for Forecast",
#         options=forecast_df['location'].unique(),
#         index=0
#     )
    
#     # Year selector
#     forecast_year = st.selectbox(
#         "Select Year",
#         options=sorted(forecast_df['year'].unique()),
#         index=0
#     )
    
#     # Filter forecast data and create a copy
#     location_forecast = forecast_df[
#         (forecast_df['location'] == forecast_location) & 
#         (forecast_df['year'] == forecast_year)
#     ].copy()
    
#     # Display forecast
#     st.subheader(f"Disaster Forecast for {forecast_location} in {forecast_year}")
    
#     # Create forecast table
#     forecast_display = location_forecast[['month', 'disaster_type']].copy()
#     forecast_display['month'] = forecast_display['month'].apply(
#         lambda x: datetime(2024, x, 1).strftime('%B')
#     )
#     forecast_display.columns = ['Month', 'Predicted Disaster']
#     st.dataframe(forecast_display)
    
#     # Probability chart
#     st.subheader("Disaster Probability by Month")
    
#     # Get probability columns
#     prob_cols = [col for col in location_forecast.columns if col.startswith('prob_')]
#     prob_data = location_forecast[['month'] + prob_cols].copy()
    
#     # Melt for plotting
#     prob_melted = prob_data.melt(
#         id_vars=['month'],
#         var_name='disaster_type',
#         value_name='probability'
#     )
#     prob_melted['disaster_type'] = prob_melted['disaster_type'].str.replace('prob_', '')
    
#     # Map disaster codes to names if needed
#     if prob_melted['disaster_type'].dtype in [np.int64, np.int32, int]:
#         prob_melted['disaster_type'] = prob_melted['disaster_type'].map(disaster_mapping)
    
#     prob_melted['month'] = prob_melted['month'].apply(
#         lambda x: datetime(2024, x, 1).strftime('%B')
#     )
    
#     fig_prob = px.bar(
#         prob_melted,
#         x='month',
#         y='probability',
#         color='disaster_type',
#         title=f"Disaster Probabilities for {forecast_location} in {forecast_year}",
#         barmode='stack'
#     )
#     st.plotly_chart(fig_prob, use_container_width=True)
    
#     # 5-year trend
#     st.subheader(f"5-Year Disaster Trend for {forecast_location}")
    
#     # Get most likely disaster each year
#     yearly_forecast = forecast_df[forecast_df['location'] == forecast_location].copy()
#     yearly_summary = yearly_forecast.groupby('year')['disaster_type'].apply(
#         lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
#     ).reset_index(name='most_likely_disaster')
    
#     # Map disaster codes to names if needed
#     if yearly_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int]:
#         yearly_summary['most_likely_disaster'] = yearly_summary['most_likely_disaster'].map(disaster_mapping)
    
#     fig_trend = px.line(
#         yearly_summary,
#         x='year',
#         y='most_likely_disaster',
#         title=f"Most Likely Disaster Trend for {forecast_location}",
#         markers=True,
#         line_shape='linear'
#     )
#     fig_trend.update_traces(mode='markers+lines')
#     st.plotly_chart(fig_trend, use_container_width=True)

# # Prediction Tool Page
# elif page == "Prediction Tool":
#     st.title("Disaster Prediction Tool")
    
#     st.markdown("""
#     Use this tool to predict the most likely disaster type for a specific location and time.
#     The model uses historical patterns to make predictions.
#     """)
    
#     # Input form
#     with st.form("prediction_form"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             location = st.selectbox(
#                 "Location",
#                 options=historical_df['location'].unique(),
#                 index=0
#             )
            
#             year = st.number_input(
#                 "Year",
#                 min_value=2024,
#                 max_value=2028,
#                 value=2024,
#                 step=1
#             )
        
#         with col2:
#             month = st.selectbox(
#                 "Month",
#                 options=range(1, 13),
#                 format_func=lambda x: datetime(2024, x, 1).strftime('%B'),
#                 index=6
#             )
            
#             day = st.number_input(
#                 "Day",
#                 min_value=1,
#                 max_value=31,
#                 value=15,
#                 step=1
#             )
        
#         submit = st.form_submit_button("Predict Disaster")
    
#     if submit:
#         # Prepare input data
#         input_data = pd.DataFrame({
#             'location': [location],
#             'year': [year],
#             'month': [month],
#             'day': [day],
#             'date': [pd.to_datetime(f'{year}-{month}-{day}')]
#         })
        
#         # Apply feature engineering
#         input_data = create_prediction_features(input_data, historical_df, location_encoder, season_encoder)
        
#         # Prepare features
#         features = ['year', 'month', 'day', 'day_of_week', 'location_encoded', 
#                     'season_encoded', 'location_risk', 'disaster_freq']
        
#         # Make prediction
#         prediction = model.predict(input_data[features])[0]
        
#         # Get probabilities
#         proba = model.predict_proba(input_data[features])[0]
        
#         # Use model's classes_ attribute to get disaster classes
#         disaster_classes = model.classes_
        
#         # Convert prediction to disaster type with bounds checking
#         try:
#             if 0 <= prediction < len(disaster_classes):
#                 disaster_type = disaster_classes[prediction]
#             else:
#                 # If prediction is out of bounds, use disaster mapping
#                 disaster_type = disaster_mapping.get(prediction, f"Unknown Disaster (Code: {prediction})")
#         except Exception as e:
#             st.error(f"Error converting prediction to disaster type: {e}")
#             disaster_type = f"Error (Code: {prediction})"
        
#         # Display results
#         st.success(f"### Predicted Disaster: **{disaster_type}**")
        
#         st.subheader("Prediction Probabilities")
#         prob_df = pd.DataFrame({
#             'Disaster Type': disaster_classes,
#             'Probability': proba * 100
#         }).sort_values('Probability', ascending=False)
        
#         fig_proba = px.bar(
#             prob_df,
#             x='Probability',
#             y='Disaster Type',
#             orientation='h',
#             title="Disaster Type Probabilities",
#             color='Probability',
#             color_continuous_scale='Viridis'
#         )
#         st.plotly_chart(fig_proba, use_container_width=True)
        
#         st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}))

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("### About")
# st.sidebar.markdown("This dashboard analyzes natural disasters in Ghana and provides forecasts for the next 5 years.")
# st.sidebar.markdown("Developed as a Final Year Project")








# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import joblib
# import os

# # Set page config
# st.set_page_config(
#     page_title="Ghana Natural Disaster Dashboard",
#     page_icon="🌪️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Suppress pandas warnings
# pd.set_option('mode.chained_assignment', None)

# # Custom CSS
# def load_css():
#     css_path = os.path.join('app', 'assets', 'style.css')
#     if os.path.exists(css_path):
#         with open(css_path) as f:
#             st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# load_css()

# # Load data
# @st.cache_data
# def load_data():
#     historical_df = pd.read_csv('data/processed/cleaned_disaster_data.csv')
#     forecast_df = pd.read_csv('data/processed/forecast_results.csv')
#     location_summary = pd.read_csv('data/processed/location_forecast_summary.csv')
#     disaster_pct = pd.read_csv('data/processed/disaster_percentages.csv', index_col=0)
#     return historical_df, forecast_df, location_summary, disaster_pct

# # Load models
# @st.cache_resource
# def load_models():
#     model = joblib.load('models/disaster_predictor.pkl')
#     location_encoder = joblib.load('models/location_encoder.pkl')
#     disaster_encoder = joblib.load('models/disaster_encoder.pkl')
#     season_encoder = joblib.load('models/season_encoder.pkl')
    
#     # Try to load disaster mapping, if it doesn't exist, create it from model classes
#     try:
#         disaster_mapping = joblib.load('models/disaster_mapping.pkl')
#     except FileNotFoundError:
#         disaster_mapping = {i: disaster_type for i, disaster_type in enumerate(model.classes_)}
#         joblib.dump(disaster_mapping, 'models/disaster_mapping.pkl')
    
#     return model, location_encoder, disaster_encoder, season_encoder, disaster_mapping

# # Feature engineering function for prediction
# def create_prediction_features(input_data, historical_data, location_encoder, season_encoder):
#     # Create a copy to avoid SettingWithCopyWarning
#     input_data = input_data.copy()
    
#     # Add day of week
#     input_data['day_of_week'] = input_data['date'].dt.dayofweek
    
#     # Apply feature engineering
#     input_data['location_encoded'] = location_encoder.transform(input_data['location'])
    
#     # Create season
#     input_data['season'] = input_data['month'].apply(lambda x: 
#         'Spring' if x in [3,4,5] else
#         'Summer' if x in [6,7,8] else
#         'Fall' if x in [9,10,11] else 'Winter')
    
#     # Use manual season mapping instead of the fitted encoder
#     season_mapping = {
#         'Spring': 0,
#         'Summer': 1,
#         'Fall': 2,
#         'Winter': 3
#     }
#     input_data['season_encoded'] = input_data['season'].map(season_mapping)
    
#     # Calculate location risk
#     location_risk = historical_data.groupby('location').size() / len(historical_data)
#     input_data['location_risk'] = input_data['location'].map(location_risk)
    
#     # Calculate disaster frequency
#     disaster_freq = historical_data.groupby('disaster_type').size() / len(historical_data)
#     avg_disaster_freq = disaster_freq.mean()
#     input_data['disaster_freq'] = avg_disaster_freq
    
#     return input_data

# # Function to map disaster codes to names
# def map_disaster_types(data, mapping, column_name='disaster_type'):
#     """
#     Map disaster type codes to their names using the provided mapping.
    
#     Parameters:
#     data: DataFrame or Series containing disaster type codes
#     mapping: Dictionary mapping codes to names
#     column_name: Name of the column containing disaster type codes
    
#     Returns:
#     DataFrame or Series with mapped disaster type names
#     """
#     if isinstance(data, pd.DataFrame) and column_name in data.columns:
#         # Check if the column contains numeric values that need mapping
#         if data[column_name].dtype in [np.int64, np.int32, int, float]:
#             data[column_name] = data[column_name].map(mapping).fillna(data[column_name])
#         return data
#     elif isinstance(data, pd.Series):
#         if data.dtype in [np.int64, np.int32, int, float]:
#             return data.map(mapping).fillna(data)
#         return data
#     return data

# # Load data and models
# historical_df, forecast_df, location_summary, disaster_pct = load_data()
# model, location_encoder, disaster_encoder, season_encoder, disaster_mapping = load_models()

# # Apply disaster type mapping to all dataframes
# historical_df = map_disaster_types(historical_df, disaster_mapping)
# forecast_df = map_disaster_types(forecast_df, disaster_mapping)

# # Update location summary if it contains numeric disaster types
# if 'most_likely_disaster' in location_summary.columns:
#     location_summary = map_disaster_types(location_summary, disaster_mapping, 'most_likely_disaster')

# # Update disaster_pct columns if they are numeric
# if disaster_pct.columns.dtype in [np.int64, np.int32, int, float]:
#     new_columns = []
#     for col in disaster_pct.columns:
#         if col in disaster_mapping.values():  # Already a name
#             new_columns.append(col)
#         else:
#             # Try to convert to int first, then map
#             try:
#                 col_int = int(col)
#                 new_columns.append(disaster_mapping.get(col_int, str(col)))
#             except (ValueError, TypeError):
#                 new_columns.append(str(col))
#     disaster_pct.columns = new_columns

# # Debug information
# if st.checkbox("Show Debug Info"):
#     st.write("Model classes:", model.classes_)
#     st.write("Model classes length:", len(model.classes_))
#     st.write("Disaster mapping:", disaster_mapping)
#     st.write("Sample forecast data types:")
#     st.write(forecast_df.dtypes)
#     st.write("Sample forecast data:")
#     st.write(forecast_df.head())
#     st.write("Historical data disaster types:")
#     st.write(historical_df['disaster_type'].unique())
#     st.write("Forecast data disaster types:")
#     st.write(forecast_df['disaster_type'].unique())

# # Sidebar
# st.sidebar.title("🌪️ Disaster Dashboard")
# st.sidebar.markdown("### Navigation")
# page = st.sidebar.radio("Go to", ["Overview", "Historical Analysis", "Forecast", "Prediction Tool"])

# # Overview Page
# if page == "Overview":
#     st.title("Ghana Natural Disaster Overview")
    
#     # Key metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Disasters", f"{historical_df.shape[0]:,}")
#     with col2:
#         st.metric("Affected Locations", f"{historical_df['location'].nunique():,}")
#     with col3:
#         st.metric("Disaster Types", f"{historical_df['disaster_type'].nunique():,}")
#     with col4:
#         st.metric("Forecast Years", "5 (2024-2028)")
    
#     # Disaster distribution
#     st.subheader("Disaster Type Distribution")
#     disaster_counts = historical_df['disaster_type'].value_counts().reset_index()
#     fig_disaster = px.pie(
#         disaster_counts, 
#         values='count', 
#         names='disaster_type',
#         title="Historical Disaster Distribution",
#         hole=0.4
#     )
#     st.plotly_chart(fig_disaster, use_container_width=True)
    
#     # Top affected locations
#     st.subheader("Top 10 Most Affected Locations")
#     top_locations = historical_df['location'].value_counts().head(10).reset_index()
#     fig_locations = px.bar(
#         top_locations,
#         x='count',
#         y='location',
#         orientation='h',
#         title="Historical Disaster Frequency by Location",
#         color='count',
#         color_continuous_scale='Viridis'
#     )
#     st.plotly_chart(fig_locations, use_container_width=True)
    
#     # Forecast summary
#     st.subheader("5-Year Forecast Summary")
#     st.markdown("### Most Likely Disasters by Location")
    
#     # Ensure location summary displays disaster names, not codes
#     display_summary = location_summary.copy()
#     if 'most_likely_disaster' in display_summary.columns:
#         display_summary = map_disaster_types(display_summary, disaster_mapping, 'most_likely_disaster')
    
#     st.dataframe(display_summary.sort_values('location'))
    
#     st.markdown("### Disaster Probability Heatmap")
    
#     # Ensure disaster_pct has proper column names
#     display_disaster_pct = disaster_pct.copy()
#     if display_disaster_pct.columns.dtype in [np.int64, np.int32, int, float]:
#         new_columns = []
#         for col in display_disaster_pct.columns:
#             if col in disaster_mapping.values():  # Already a name
#                 new_columns.append(col)
#             else:
#                 # Try to convert to int first, then map
#                 try:
#                     col_int = int(col)
#                     new_columns.append(disaster_mapping.get(col_int, str(col)))
#                 except (ValueError, TypeError):
#                     new_columns.append(str(col))
#         display_disaster_pct.columns = new_columns
    
#     fig_heatmap = px.imshow(
#         display_disaster_pct,
#         labels=dict(x="Disaster Type", y="Location", color="Probability (%)"),
#         color_continuous_scale='Viridis',
#         aspect="auto"
#     )
#     st.plotly_chart(fig_heatmap, use_container_width=True)

# # Historical Analysis Page
# elif page == "Historical Analysis":
#     st.title("Historical Disaster Analysis")
    
#     # Location selector
#     selected_location = st.selectbox(
#         "Select Location",
#         options=historical_df['location'].unique(),
#         index=0
#     )
    
#     # Filter data and create a copy to avoid SettingWithCopyWarning
#     location_data = historical_df[historical_df['location'] == selected_location].copy()
    
#     # Convert date
#     location_data['date'] = pd.to_datetime(location_data['date'])
#     location_data['year'] = location_data['date'].dt.year
#     location_data['month'] = location_data['date'].dt.month
    
#     # Yearly trend
#     st.subheader(f"Yearly Disaster Trend in {selected_location}")
#     yearly_counts = location_data.groupby('year').size().reset_index(name='count')
#     fig_yearly = px.line(
#         yearly_counts,
#         x='year',
#         y='count',
#         title=f"Disaster Frequency in {selected_location}",
#         markers=True
#     )
#     st.plotly_chart(fig_yearly, use_container_width=True)
    
#     # Monthly pattern
#     st.subheader(f"Monthly Disaster Pattern in {selected_location}")
#     monthly_counts = location_data.groupby('month').size().reset_index(name='count')
#     fig_monthly = px.bar(
#         monthly_counts,
#         x='month',
#         y='count',
#         title=f"Monthly Disaster Frequency in {selected_location}",
#         color='count',
#         color_continuous_scale='Blues'
#     )
#     st.plotly_chart(fig_monthly, use_container_width=True)
    
#     # Disaster types
#     st.subheader(f"Disaster Types in {selected_location}")
#     disaster_types = location_data['disaster_type'].value_counts().reset_index()
#     fig_types = px.bar(
#         disaster_types,
#         x='count',
#         y='disaster_type',
#         orientation='h',
#         title=f"Disaster Types in {selected_location}",
#         color='count',
#         color_continuous_scale='Reds'
#     )
#     st.plotly_chart(fig_types, use_container_width=True)

# # Forecast Page
# elif page == "Forecast":
#     st.title("5-Year Disaster Forecast (2024-2028)")
    
#     # Location selector
#     forecast_location = st.selectbox(
#         "Select Location for Forecast",
#         options=forecast_df['location'].unique(),
#         index=0
#     )
    
#     # Year selector
#     forecast_year = st.selectbox(
#         "Select Year",
#         options=sorted(forecast_df['year'].unique()),
#         index=0
#     )
    
#     # Filter forecast data and create a copy
#     location_forecast = forecast_df[
#         (forecast_df['location'] == forecast_location) & 
#         (forecast_df['year'] == forecast_year)
#     ].copy()
    
#     # Display forecast
#     st.subheader(f"Disaster Forecast for {forecast_location} in {forecast_year}")
    
#     # Create forecast table
#     forecast_display = location_forecast[['month', 'disaster_type']].copy()
#     forecast_display['month'] = forecast_display['month'].apply(
#         lambda x: datetime(2024, x, 1).strftime('%B')
#     )
#     forecast_display.columns = ['Month', 'Predicted Disaster']
#     st.dataframe(forecast_display)
    
#     # Probability chart
#     st.subheader("Disaster Probability by Month")
    
#     # Get probability columns
#     prob_cols = [col for col in location_forecast.columns if col.startswith('prob_')]
#     prob_data = location_forecast[['month'] + prob_cols].copy()
    
#     # Melt for plotting
#     prob_melted = prob_data.melt(
#         id_vars=['month'],
#         var_name='disaster_type',
#         value_name='probability'
#     )
#     prob_melted['disaster_type'] = prob_melted['disaster_type'].str.replace('prob_', '')
    
#     # Map disaster codes to names if needed
#     if prob_melted['disaster_type'].dtype in [np.int64, np.int32, int, float]:
#         # Try to convert to int first, then map
#         try:
#             prob_melted['disaster_type'] = prob_melted['disaster_type'].astype(int).map(disaster_mapping)
#         except (ValueError, TypeError):
#             # If conversion fails, try to map directly
#             prob_melted['disaster_type'] = prob_melted['disaster_type'].map(disaster_mapping)
    
#     prob_melted['month'] = prob_melted['month'].apply(
#         lambda x: datetime(2024, x, 1).strftime('%B')
#     )
    
#     fig_prob = px.bar(
#         prob_melted,
#         x='month',
#         y='probability',
#         color='disaster_type',
#         title=f"Disaster Probabilities for {forecast_location} in {forecast_year}",
#         barmode='stack'
#     )
#     st.plotly_chart(fig_prob, use_container_width=True)
    
#     # 5-year trend
#     st.subheader(f"5-Year Disaster Trend for {forecast_location}")
    
#     # Get most likely disaster each year
#     yearly_forecast = forecast_df[forecast_df['location'] == forecast_location].copy()
#     yearly_summary = yearly_forecast.groupby('year')['disaster_type'].apply(
#         lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
#     ).reset_index(name='most_likely_disaster')
    
#     # Map disaster codes to names if needed
#     if yearly_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int, float]:
#         # Try to convert to int first, then map
#         try:
#             yearly_summary['most_likely_disaster'] = yearly_summary['most_likely_disaster'].astype(int).map(disaster_mapping)
#         except (ValueError, TypeError):
#             # If conversion fails, try to map directly
#             yearly_summary['most_likely_disaster'] = yearly_summary['most_likely_disaster'].map(disaster_mapping)
    
#     fig_trend = px.line(
#         yearly_summary,
#         x='year',
#         y='most_likely_disaster',
#         title=f"Most Likely Disaster Trend for {forecast_location}",
#         markers=True,
#         line_shape='linear'
#     )
#     fig_trend.update_traces(mode='markers+lines')
#     st.plotly_chart(fig_trend, use_container_width=True)

# # Prediction Tool Page
# elif page == "Prediction Tool":
#     st.title("Disaster Prediction Tool")
    
#     st.markdown("""
#     Use this tool to predict the most likely disaster type for a specific location and time.
#     The model uses historical patterns to make predictions.
#     """)
    
#     # Input form
#     with st.form("prediction_form"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             location = st.selectbox(
#                 "Location",
#                 options=historical_df['location'].unique(),
#                 index=0
#             )
            
#             year = st.number_input(
#                 "Year",
#                 min_value=2024,
#                 max_value=2028,
#                 value=2024,
#                 step=1
#             )
        
#         with col2:
#             month = st.selectbox(
#                 "Month",
#                 options=range(1, 13),
#                 format_func=lambda x: datetime(2024, x, 1).strftime('%B'),
#                 index=6
#             )
            
#             day = st.number_input(
#                 "Day",
#                 min_value=1,
#                 max_value=31,
#                 value=15,
#                 step=1
#             )
        
#         submit = st.form_submit_button("Predict Disaster")
    
#     if submit:
#         # Prepare input data
#         input_data = pd.DataFrame({
#             'location': [location],
#             'year': [year],
#             'month': [month],
#             'day': [day],
#             'date': [pd.to_datetime(f'{year}-{month}-{day}')]
#         })
        
#         # Apply feature engineering
#         input_data = create_prediction_features(input_data, historical_df, location_encoder, season_encoder)
        
#         # Prepare features
#         features = ['year', 'month', 'day', 'day_of_week', 'location_encoded', 
#                     'season_encoded', 'location_risk', 'disaster_freq']
        
#         # Make prediction
#         prediction = model.predict(input_data[features])[0]
        
#         # Get probabilities
#         proba = model.predict_proba(input_data[features])[0]
        
#         # Use model's classes_ attribute to get disaster classes
#         disaster_classes = model.classes_
        
#         # Convert prediction to disaster type with bounds checking
#         try:
#             if 0 <= prediction < len(disaster_classes):
#                 disaster_type = disaster_classes[prediction]
#             else:
#                 # If prediction is out of bounds, use disaster mapping
#                 disaster_type = disaster_mapping.get(prediction, f"Unknown Disaster (Code: {prediction})")
#         except Exception as e:
#             st.error(f"Error converting prediction to disaster type: {e}")
#             disaster_type = f"Error (Code: {prediction})"
        
#         # Display results
#         st.success(f"### Predicted Disaster: **{disaster_type}**")
        
#         st.subheader("Prediction Probabilities")
#         prob_df = pd.DataFrame({
#             'Disaster Type': disaster_classes,
#             'Probability': proba * 100
#         }).sort_values('Probability', ascending=False)
        
#         fig_proba = px.bar(
#             prob_df,
#             x='Probability',
#             y='Disaster Type',
#             orientation='h',
#             title="Disaster Type Probabilities",
#             color='Probability',
#             color_continuous_scale='Viridis'
#         )
#         st.plotly_chart(fig_proba, use_container_width=True)
        
#         st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}))

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("### About")
# st.sidebar.markdown("This dashboard analyzes natural disasters in Ghana and provides forecasts for the next 5 years.")
# st.sidebar.markdown("Developed as a Final Year Project")



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
        # Create mapping from the model's classes
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

# Function to map disaster codes to names
def map_disaster_types(data, mapping, column_name='disaster_type'):
    """
    Map disaster type codes to their names using the provided mapping.
    
    Parameters:
    data: DataFrame or Series containing disaster type codes
    mapping: Dictionary mapping codes to names
    column_name: Name of the column containing disaster type codes
    
    Returns:
    DataFrame or Series with mapped disaster type names
    """
    if isinstance(data, pd.DataFrame) and column_name in data.columns:
        # Check if the column contains numeric values that need mapping
        if data[column_name].dtype in [np.int64, np.int32, int, float]:
            data[column_name] = data[column_name].map(mapping).fillna(data[column_name])
        return data
    elif isinstance(data, pd.Series):
        if data.dtype in [np.int64, np.int32, int, float]:
            return data.map(mapping).fillna(data)
        return data
    return data

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

# Apply disaster type mapping to all dataframes
# First check if forecast_df has numeric disaster types
if 'disaster_type' in forecast_df.columns and forecast_df['disaster_type'].dtype in [np.int64, np.int32, int, float]:
    forecast_df['disaster_type'] = forecast_df['disaster_type'].map(disaster_mapping)

# Update location summary if it contains numeric disaster types
if 'most_likely_disaster' in location_summary.columns and location_summary['most_likely_disaster'].dtype in [np.int64, np.int32, int, float]:
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
page = st.sidebar.radio("Go to", ["Overview", "Historical Analysis", "Forecast", "Prediction Tool"])

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
        st.metric("Forecast Years", "5 (2024-2028)")
    
    # Disaster distribution
    st.subheader("Disaster Type Distribution")
    disaster_counts = historical_df['disaster_type'].value_counts().reset_index()
    fig_disaster = px.pie(
        disaster_counts, 
        values='count', 
        names='disaster_type',
        title="Historical Disaster Distribution",
        hole=0.4
    )
    st.plotly_chart(fig_disaster, use_container_width=True)
    
    # Top affected locations
    st.subheader("Top 10 Most Affected Locations")
    top_locations = historical_df['location'].value_counts().head(10).reset_index()
    fig_locations = px.bar(
        top_locations,
        x='count',
        y='location',
        orientation='h',
        title="Historical Disaster Frequency by Location",
        color='count',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_locations, use_container_width=True)
    
    # Forecast summary
    st.subheader("5-Year Forecast Summary")
    st.markdown("### Most Likely Disasters by Location")
    st.dataframe(location_summary.sort_values('location'))
    
    st.markdown("### Disaster Probability Heatmap")
    fig_heatmap = px.imshow(
        disaster_pct,
        labels=dict(x="Disaster Type", y="Location", color="Probability (%)"),
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Historical Analysis Page
elif page == "Historical Analysis":
    st.title("Historical Disaster Analysis")
    
    # Location selector
    selected_location = st.selectbox(
        "Select Location",
        options=historical_df['location'].unique(),
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

# Forecast Page
elif page == "Forecast":
    st.title("5-Year Disaster Forecast (2024-2028)")
    
    # Location selector
    forecast_location = st.selectbox(
        "Select Location for Forecast",
        options=forecast_df['location'].unique(),
        index=0
    )
    
    # Year selector
    forecast_year = st.selectbox(
        "Select Year",
        options=sorted(forecast_df['year'].unique()),
        index=0
    )
    
    # Filter forecast data and create a copy
    location_forecast = forecast_df[
        (forecast_df['location'] == forecast_location) & 
        (forecast_df['year'] == forecast_year)
    ].copy()
    
    # Display forecast
    st.subheader(f"Disaster Forecast for {forecast_location} in {forecast_year}")
    
    # Create forecast table
    forecast_display = location_forecast[['month', 'disaster_type']].copy()
    forecast_display['month'] = forecast_display['month'].apply(
        lambda x: datetime(2024, x, 1).strftime('%B')
    )
    forecast_display.columns = ['Month', 'Predicted Disaster']
    st.dataframe(forecast_display)
    
    # Probability chart
    st.subheader("Disaster Probability by Month")
    
    # Get probability columns
    prob_cols = [col for col in location_forecast.columns if col.startswith('prob_')]
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
    
    # 5-year trend
    st.subheader(f"5-Year Disaster Trend for {forecast_location}")
    
    # Get most likely disaster each year
    yearly_forecast = forecast_df[forecast_df['location'] == forecast_location].copy()
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
    
    fig_trend = px.line(
        yearly_summary,
        x='year',
        y='most_likely_disaster',
        title=f"Most Likely Disaster Trend for {forecast_location}",
        markers=True,
        line_shape='linear'
    )
    fig_trend.update_traces(mode='markers+lines')
    st.plotly_chart(fig_trend, use_container_width=True)

# Prediction Tool Page
elif page == "Prediction Tool":
    st.title("Disaster Prediction Tool")
    
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
                min_value=2024,
                max_value=2028,
                value=2024,
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
        
        submit = st.form_submit_button("Predict Disaster")
    
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
                disaster_type = disaster_classes[prediction]
            else:
                # If prediction is out of bounds, use disaster mapping
                disaster_type = disaster_mapping.get(prediction, f"Unknown Disaster (Code: {prediction})")
        except Exception as e:
            st.error(f"Error converting prediction to disaster type: {e}")
            disaster_type = f"Error (Code: {prediction})"
        
        # Display results
        st.success(f"### Predicted Disaster: **{disaster_type}**")
        
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Disaster Type': disaster_classes,
            'Probability': proba * 100
        }).sort_values('Probability', ascending=False)
        
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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This dashboard analyzes natural disasters in Ghana and provides forecasts for the next 5 years.")
st.sidebar.markdown("Developed as a Final Year Project")