# src/predict.py

import joblib
import pandas as pd

from src.utils import preprocess
from src.config import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict_from_dataframe(df):
    model = load_model()
    X, _ = preprocess(df)
    return model.predict(X)


if __name__ == "__main__":
    df = pd.read_csv("data/new_movies.csv")
    preds = predict_from_dataframe(df)
    print(preds)
