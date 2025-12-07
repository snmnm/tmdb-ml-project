# src/utils.py

import pandas as pd
import ast

def load_data(path):
    """CSV dosyasını yükler."""
    return pd.read_csv(path)

def parse_json_column(column):
    """['Action', 'Drama'] gibi json listelerini parse eder."""
    if pd.isna(column):
        return []
    try:
        return [item['name'] for item in ast.literal_eval(column)]
    except:
        return []

def preprocess(df):
    """
    Notebook’taki feature engineering steps:
    - genres -> list -> genre_count
    - keywords -> list -> keyword_count
    - select features
    """
    # JSON kolonlarını parse et
    df["genres_list"] = df["genres"].apply(parse_json_column)
    df["keywords_list"] = df["keywords"].apply(parse_json_column)

    # Feature engineering
    df["genre_count"] = df["genres_list"].apply(len)
    df["keyword_count"] = df["keywords_list"].apply(len)

    # Release year extraction (zaten notebook’ta vardı)
    df["release_year"] = pd.to_datetime(df["release_date"]).dt.year

    # Feature seçimi
    feature_cols = [
        "budget", "revenue", "runtime", "popularity",
        "vote_count", "release_year", "genre_count", "keyword_count"
    ]

    X = df[feature_cols]
    y = df["vote_average"]

    return X, y
