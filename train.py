import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

def train_demand_models():
    df = pd.read_csv("parking_pricing_data.csv")
    
    # Feature Engineering
    df["price_to_competitor_ratio"] = df["price"] / (df["competitor_price"] + 1e-5)
    df["is_peak_hour"] = df["hour_of_day"].apply(lambda h: 1 if 8 <= h <= 19 else 0)
    
    features = [
        "hour_of_day", "is_weekend", "traffic_index", 
        "competitor_price", "occupancy_ratio", "price",
        "price_to_competitor_ratio", "is_peak_hour"
    ]
    target = "demand"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Baseline Model (Ridge Regression)
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    rmse_base = root_mean_squared_error(y_test, y_pred_base)
    
    # 2. LightGBM Regressor
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42, 
        verbosity=-1
    )
    lgbm_model.fit(X_train, y_train)
    y_pred_lgbm = lgbm_model.predict(X_test)
    rmse_lgbm = root_mean_squared_error(y_test, y_pred_lgbm)
    mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    
    improvement = ((rmse_base - rmse_lgbm) / rmse_base) * 100
    
    print("--- Benchmark Results ---")
    print(f"Baseline (Ridge) RMSE : {rmse_base:.3f}")
    print(f"LightGBM RMSE         : {rmse_lgbm:.3f} ({improvement:.1f}% improvement)")
    print(f"LightGBM MAE          : {mae_lgbm:.3f}")
    print(f"LightGBM R2 Score     : {r2_lgbm:.3f}")
    
    # Save best model
    joblib.dump(lgbm_model, "best_demand_model.pkl")
    print("Model serialized to 'best_demand_model.pkl'.")

if __name__ == "__main__":
    train_demand_models()
