import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Movie Revenue Prediction",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Revenue Prediction")
st.write("Predict the expected movie revenue using a trained Random Forest model.")

# -------------------------------
# Load Model
# -------------------------------
with open("movie_revenue_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# User Inputs
# -------------------------------
budget = st.number_input(
    "Budget (in Millions)",
    min_value=1.0,
    max_value=500.0,
    value=100.0
)

runtime = st.number_input(
    "Runtime (minutes)",
    min_value=60,
    max_value=240,
    value=120
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

# -------------------------------
# Create Input Data
# -------------------------------
input_df = pd.DataFrame({
    "Budget": [budget],
    "Runtime": [runtime],
    "Director_Rating": [director_rating],
    "Star_Power": [star_power],
    "Genre": [genre]
})

# Apply one-hot encoding
input_df = pd.get_dummies(input_df, columns=["Genre"])

# Model feature order
model_features = [
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
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Arrange columns in correct order
input_df = input_df[model_features]

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Revenue"):
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Revenue: ${prediction:,.2f} Million")

    st.balloons()
with open("movie_revenue_model.pkl", "rb") as f:
    model = pickle.load(f)

print(model.feature_names_in_)
