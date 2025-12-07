# src/train.py

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS
from src.utils import load_data, preprocess


def train():
    df = load_data(DATA_PATH)

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("MAE:", round(mae, 4))
    print("R2 Score:", round(r2, 4))

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train()
