import streamlit as st
import pandas as pd
import pickle

# Load model
with open("random_forest.pkl", "rb") as f:
    rf = pickle.load(f)

# Load training column names and feature means
with open("x_train_columns.pkl", "rb") as f:
    all_columns = pickle.load(f)

with open("x_train_mean.pkl", "rb") as f:
    feature_means = pickle.load(f)

# Function to predict
def predict_value_rf(user_input: dict):
    input_data = feature_means.copy()  # Start with means
    input_data.update(user_input)      # Update with user input
    df = pd.DataFrame([input_data])[all_columns]  # Ensure column order
    return rf.predict(df)[0]

# UI
st.title("⚽ Player Market Value Predictor")

st.markdown("Enter the player stats:")

# Collect user input
player_features = {
    'Best overall (BOV)': st.slider("BOV", 40, 100, 85),
    'AGE': st.slider("AGE", 16, 40, 19),
    'GROWTH': st.slider("GROWTH", 0, 20, 9),
    'FOOT': st.selectbox("Preferred Foot", [0, 1], format_func=lambda x: "Left" if x == 0 else "Right"),
    'GROWTH_POTENTIAL': st.slider("Growth Potential", 0, 20, 6),
    'Overall Rating (OVA)': st.slider("OVA", 30, 100, 66),
    'FINISHING': st.slider("Finishing", 0, 100, 60),
    'REACTIONS': st.slider("Reactions", 0, 100, 75),
    'LONG_PASSING': st.slider("Long Passing", 0, 100, 42),
    'BALL_CONTROL': st.slider("Ball Control", 0, 100, 90)
}

if st.button("Predict Market Value"):
    prediction = predict_value_rf(player_features)
    st.success(f"💰 Predicted Market Value: €{prediction:,.2f}")
