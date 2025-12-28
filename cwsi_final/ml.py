import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


# -------------------------------
# ML Feature Export
# -------------------------------
def export_ml_features_from_field_stats(df, out_csv="ml_features.csv"):
    d = df.copy()

    # Derived stress indicators
    d["stress_ratio"] = d["p90_cwsi"] / (d["mean_cwsi"] + 1e-6)
    d["log_pixels"] = np.log1p(d["n_pixels"])

    d.to_csv(out_csv, index=False)
    return out_csv


# -------------------------------
# Train XGBoost Model
# -------------------------------
def train_xgboost(features_csv, target_col="yield", model_out="xgb_model.joblib"):
    df = pd.read_csv(features_csv)

    # If no yield column exists → simulate one
    if target_col not in df.columns:
        np.random.seed(0)
        df[target_col] = (
            5.0 - 3.0 * df["mean_cwsi"] + 0.4 * np.random.randn(len(df))
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Ensure no NaNs
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    joblib.dump(model, model_out)

    return {
        "model_path": model_out,
        "mse": mse,
        "r2": r2
    }
