# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("music_popularity_model.pkl")
scaler = joblib.load("music_scaler.pkl")

# Streamlit App UI
st.set_page_config(page_title="Music Popularity Predictor", layout="centered")
st.title("🎵 Music Popularity Predictor")
st.write("Enter song features to predict its popularity score")

# Input fields
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
key = st.slider("Key (Pitch Class)", 0, 11, 5)
loudness = st.slider("Loudness (dB)", -60.0, 0.0, -12.0)
mode = st.selectbox("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0)
duration_ms = st.slider("Duration (ms)", 60000, 300000, 180000)
explicit = st.selectbox("Explicit", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
release_year = st.slider("Release Year", 2000, 2025, 2024)
release_month = st.slider("Release Month", 1, 12, 6)
track_encoded = st.number_input("Track Name Encoded (optional)", value=0)
artist_encoded = st.number_input("Artist Encoded (optional)", value=0)

# Prediction
if st.button("Predict Popularity"):
    input_data = np.array([[track_encoded, artist_encoded, duration_ms, explicit, danceability,
                            energy, key, loudness, mode, speechiness, acousticness,
                            instrumentalness, liveness, valence, tempo, release_year, release_month]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"🎯 Predicted Popularity Score: **{prediction[0]:.2f}**")
