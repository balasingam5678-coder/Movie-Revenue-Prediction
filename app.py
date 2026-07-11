import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Movie Revenue Prediction",
    page_icon="🎬",
    layout="centered"
)

# ----------------------------
# Load Trained Model
# ----------------------------
with open("movie_revenue_model.pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------------
# Title
# ----------------------------
st.title("🎬 Movie Revenue Prediction")
st.write("Enter the movie details below to predict the expected revenue.")

# ----------------------------
# User Inputs
# ----------------------------
budget = st.number_input(
    "Budget (in Million $)",
    min_value=1.0,
    value=50.0
)

runtime = st.number_input(
    "Runtime (minutes)",
    min_value=60,
    max_value=240,
    value=120
)

genre = st.selectbox(
    "Genre",
    [
        "Action",
        "Comedy",
        "Drama",
        "Horror",
        "Romance",
        "Sci-Fi",
        "Thriller"
    ]
)

director_rating = st.slider(
    "Director Rating",
    1.0,
    10.0,
    7.0
)

star_power = st.slider(
    "Star Power",
    1.0,
    10.0,
    7.0
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Revenue"):

    input_df = pd.DataFrame({
        "Budget": [budget],
        "Runtime": [runtime],
        "Genre": [genre],
        "Director_Rating": [director_rating],
        "Star_Power": [star_power]
    })

    # One-hot encoding
    input_df = pd.get_dummies(input_df, columns=["Genre"])

    # Feature names used during training
    training_columns = [
        'Budget',
        'Runtime',
        'Director_Rating',
        'Star_Power',
        'Genre_Action',
        'Genre_Comedy',
        'Genre_Drama',
        'Genre_Horror',
        'Genre_Romance',
        'Genre_Sci-Fi',
        'Genre_Thriller'
    ]

    # Add missing columns
    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in correct order
    input_df = input_df[training_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"🎉 Predicted Revenue: ${prediction:,.2f} Million")
