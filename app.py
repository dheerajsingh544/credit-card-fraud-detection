import streamlit as st
import numpy as np
import joblib
import os
import tensorflow as tf

# Load the trained Keras model
model_path = 'models/best_model.pkl'

# Check if model exists
if not os.path.exists(model_path):
    st.error("❌ Trained model not found! Please ensure 'models/best_model.pkl' exists.")
    st.stop()

# Load model
model = joblib.load(model_path)

# Page config
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Title section
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    "🔐 Predict whether a credit card transaction is **fraudulent or legitimate** using a trained machine learning model."
)

# Input section
st.subheader("🧾 Transaction Features (V1 - V28) + Amount")
v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0, format="%.6f")
    v_features.append(val)

amount = st.number_input("💰 Transaction Amount", value=0.0, format="%.2f")

# Prepare input
input_data = np.array([v_features + [amount]])

# Predict button
if st.button("🔍 Predict"):
    try:
        proba = model.predict(input_data)[0][0]  # Keras returns [[0.123]]
        if proba > 0.5:
            st.error(f"⚠️ Fraud Detected! (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {1 - proba:.2%})")
    except Exception as e:
        st.exception(f"Prediction Error: {e}")

# Footer
st.markdown("<hr style='border:1px solid #444;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "🚀 Built with ❤️ by <b>Dheeraj Singh</b> | "
    "💳 Credit Card Fraud Detection App | "
    "</div>",
    unsafe_allow_html=True
)
