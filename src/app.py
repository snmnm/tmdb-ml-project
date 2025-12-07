import streamlit as st
import joblib
import pandas as pd
from src.utils import preprocess
from src.config import MODEL_PATH


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Movie Rating Predictor")
st.write("Film özelliklerini girerek tahmini IMDb puanını görebilirsiniz.")

budget = st.number_input("Budget", min_value=0.0)
revenue = st.number_input("Revenue", min_value=0.0)
runtime = st.number_input("Runtime (min)", min_value=0.0)
popularity = st.number_input("Popularity", min_value=0.0)
vote_count = st.number_input("Vote Count", min_value=0.0)
release_year = st.number_input("Release Year", min_value=1900, max_value=2030)
genre_count = st.number_input("Genre Count", min_value=0, max_value=20)
keyword_count = st.number_input("Keyword Count", min_value=0, max_value=20)

if st.button("Predict Rating"):
    df = pd.DataFrame([{
        "budget": budget,
        "revenue": revenue,
        "runtime": runtime,
        "popularity": popularity,
        "vote_count": vote_count,
        "release_year": release_year,
        "genre_count": genre_count,
        "keyword_count": keyword_count
    }])

    # preprocess dataframe (X, y) formatında döner, y yoksa sorun değil
    X, _ = preprocess(df)

    model = load_model()
    pred = model.predict(X)[0]

    st.subheader(f"Tahmini IMDb Rating: {pred:.2f}")
