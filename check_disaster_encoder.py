# check_disaster_encoder.py
import joblib

# Load the disaster encoder
disaster_encoder = joblib.load('models/disaster_encoder.pkl')

print("Disaster encoder classes:")
for i, disaster_type in enumerate(disaster_encoder.classes_):
    print(f"{i}: {disaster_type}")