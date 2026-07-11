import streamlit as st
import pandas as pd
import pickle

# Page configuration (must come first)
st.set_page_config(
    page_title="Movie Revenue Prediction",
    page_icon="🎬",
    layout="centered"
)

# Load trained model
with open("movie_revenue_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv("movies.csv")

# Title
st.title("🎬 Movie Revenue Prediction")
st.write("Enter the movie details below to predict movie revenue.")

# Input fields
budget = st.number_input(
    "Budget ($)",
    min_value=0,
    value=int(df["budget"].median()),
    step=100000
)

runtime = st.number_input(
    "Runtime (minutes)",
    min_value=1,
    value=int(df["runtime"].median()),
    step=1
)

popularity = st.number_input(
    "Popularity",
    min_value=0.0,
    value=float(df["popularity"].median()),
    step=0.1
)

vote_average = st.number_input(
    "Vote Average",
    min_value=0.0,
    max_value=10.0,
    value=float(df["vote_average"].median()),
    step=0.1
)

vote_count = st.number_input(
    "Vote Count",
    min_value=0,
    value=int(df["vote_count"].median()),
    step=1
)

# Create input dataframe
input_data = pd.DataFrame(
    [[budget, runtime, popularity, vote_average, vote_count]],
    columns=[
        "budget",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count"
    ]
)

# Prediction
if st.button("Predict Revenue"):
    prediction = model.predict(input_data)[0]

    st.success(f"🎉 Predicted Movie Revenue: ${prediction:,.2f}")

    st.write("### Input Summary")
    st.dataframe(input_data)
