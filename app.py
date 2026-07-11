import streamlit as st
import pandas as pd
import pickle

# Load model
with open("movie_revenue_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv("movies.csv")

st.set_page_config(page_title="Movie Revenue Prediction", page_icon="🎬")

st.title("🎬 Movie Revenue Prediction")
st.write("Enter the movie details below.")

# User Inputs
budget = st.number_input(
    "Budget ($)",
    min_value=0,
    value=int(df["budget"].median())
)

runtime = st.number_input(
    "Runtime (minutes)",
    min_value=1,
    value=int(df["runtime"].median())
)

popularity = st.number_input(
    "Popularity",
    min_value=0.0,
    value=float(df["popularity"].median())
)

vote_average = st.number_input(
    "Vote Average",
    min_value=0.0,
    max_value=10.0,
    value=float(df["vote_average"].median())
)

vote_count = st.number_input(
    "Vote Count",
    min_value=0,
    value=int(df["vote_count"].median())
)

if st.button("Predict Revenue"):

    # Create input dictionary
    user_data = {
        "budget": budget,
        "runtime": runtime,
        "popularity": popularity,
        "vote_average": vote_average,
        "vote_count": vote_count
    }

    # Create DataFrame
    input_data = pd.DataFrame([user_data])

    # Reorder columns to match model training
    if hasattr(model, "feature_names_in_"):
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)

    st.success(f"🎉 Predicted Revenue: ${prediction[0]:,.2f}")
